import asyncio
from dataclasses import dataclass
from typing import List, Tuple

import torch
from ray import serve

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.moge_model import MoGeModel, MoGeModelReconstructOutput
from tools.apis.reconstructor import (
    GeometricReconstructor, 
    GeometricReconstructionOutput
)


__ALL__ = ['MetricScaleOutput', 'MetricScaleEstimator']


@dataclass
class MetricScaleOutput(AgentContext):
    """
    scale_factor (float): The scaling factor to convert relative units from a reconstruction to metric units (meters). Multiplying any distance or coordinate from the relative reconstruction by this factor yields its metric equivalent.
    """
    scale_factor: float
    
    # not exposed to Planner
    _selected_index: List[int]

    def to_message_content(self) -> str:
        return (
            f'Successfully estimated metric scale factor: {self.scale_factor:.4f}, '
            f'derived from image at index {self._selected_index}.'
        )


@serve.deployment
class MetricScaleEstimator(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 4

    TOP_N = 3
    RELATIVE_CONF_PERCENTILE = 0.5
    
    def __init__(self, moge: MoGeModel, reconstructor: GeometricReconstructor):
        super().__init__()
        self.moge = moge
        self.reconstructor = reconstructor

    @AgentTool.document_output_class(MetricScaleOutput)
    async def estimate_scale(
        self,
        reconstruction: GeometricReconstructionOutput,
    ) -> AgentToolOutput:
        """
        Estimates the metric scale factor for a relative 3D reconstruction. It **MUST** be called when user's question requires real-world, metric-scale measurements (e.g., "how many meters", "calculate the length in feet", "what is the real-world size").
        Args:
            reconstruction (GeometricReconstructionOutput): The reconstruction output from `GeometricReconstructor.reconstruct`, which has a relative scale.
        """
        num_images = len(reconstruction._image)
        if num_images < 1:
            return self.error(msg='No images available in reconstruction')

        # Step 1: Sort images by reconstruction._depth_conf (Top 50%)
        depth_conf_scores = []
        for i in range(num_images):
            depth_conf_map = reconstruction._depth_conf[i].squeeze()  # Shape (H, W)
            if depth_conf_map.numel() == 0:
                depth_conf_scores.append((0., i))
                continue

            threshold = torch.quantile(depth_conf_map.float(), self.RELATIVE_CONF_PERCENTILE)
            mask_relative = depth_conf_map > threshold
            score = depth_conf_map[mask_relative].sum().item()
            depth_conf_scores.append((score, i))

        depth_conf_scores.sort(key=lambda x: x[0], reverse=True)

        # Step 2: Pick Top N images for scale estimation
        n = min(self.TOP_N, num_images)
        top_n_indices = [idx for _, idx in depth_conf_scores[:n]]

        # Step 3: Metric reconstruction with MoGE
        tasks = [
            self.moge.reconstruct.remote(image_source=reconstruction._image[idx])
            for idx in top_n_indices
        ]
        moge_results = await asyncio.gather(*tasks)

        valid_moge_results: List[Tuple[int, MoGeModelReconstructOutput]] = []
        for i, moge_result in enumerate(moge_results):
            if moge_result.err:
                continue
            valid_moge_results.append((top_n_indices[i], moge_result.result))

        if len(valid_moge_results) < 1:
            return moge_results[0]

        # Step 4: Calculate metric scale
        ratios, selected_index = [], []
        for img_idx, moge_result in valid_moge_results:
            depth_metric = moge_result.depth[0].to(torch.float32)  # (H_m, W_m)
            mask_metric = moge_result.mask[0].to(torch.bool)
            selected_index.append(img_idx)

            depth_relative = reconstruction._depth[img_idx].squeeze().to(torch.float32)  # (H_r, W_r)
            conf_map_relative = reconstruction._depth_conf[img_idx].squeeze()
            if conf_map_relative.numel() == 0:
                mask_relative = torch.zeros_like(depth_relative, dtype=torch.bool)
            else:
                threshold_relative = torch.quantile(conf_map_relative.float(), self.RELATIVE_CONF_PERCENTILE)
                mask_relative = conf_map_relative > threshold_relative
            
            tasks = [
                self.reconstructor._tensor_transform.remote(
                    tensor=depth_metric.float(),
                    reconstruction=reconstruction,
                    selected_index=img_idx,
                    interpolation='bilinear'
                ),
                self.reconstructor._tensor_transform.remote(
                    tensor=mask_metric.float(),
                    reconstruction=reconstruction,
                    selected_index=img_idx,
                    interpolation='nearest',
                )
            ]
            results = await asyncio.gather(*tasks)
            for result in results:
                if result.err:
                    return result
            
            depth_metric_aligned = results[0].result.transformed_tensor
            mask_metric_aligned = results[1].result.transformed_tensor.bool()
            valid_mask = (
                mask_relative & 
                mask_metric_aligned & 
                (depth_relative > 1e-4) & 
                (depth_metric_aligned > 1e-4)
            )
            if valid_mask.sum() < 100:
                continue 

            ratios.append(
                depth_metric_aligned[valid_mask] / depth_relative[valid_mask]
            )
        # Step 5: Gather all results
        if not ratios:
            return self.error(msg=f'Failed to find any valid overlapping depth pixels')
        final_ratios = torch.cat(ratios)
        if final_ratios.numel() == 0:
            return self.error(msg=f'Failed to find any valid overlapping depth pixels')

        scale_factor = torch.median(final_ratios).item()
        output = MetricScaleOutput(
            scale_factor=scale_factor,
            _selected_index=selected_index
        )
        return self.success(result=output)
