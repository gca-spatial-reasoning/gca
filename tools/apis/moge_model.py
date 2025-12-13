import asyncio
from dataclasses import dataclass
import inspect
from numbers import Number
import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ray import serve
import utils3d

third_party_dir = os.path.join(os.path.dirname(__file__), '..', 'third_party')
from tools.utils.misc import add_sys_path
with add_sys_path(os.path.join(third_party_dir, 'MoGe')):
    from moge.model.v2 import MoGeModel as MoGe
    from moge.utils.geometry_torch import recover_focal_shift
    import utils3d

from tools.apis.base import AgentTool, AgentContext, AgentContext, AgentToolOutput
from tools.apis.io import ImageLoader
from tools.apis.sam2_model import SAM2Model, SAM2ModelOutput


__ALL__ = [
    'MoGeModel', 
    'MoGeModelReconstructOutput', 
    'MoGeModelProjectionOutput',
]


@torch.inference_mode()
def infer(
    model, 
    image: torch.Tensor, 
    num_tokens: int = None,
    resolution_level: int = 9,
    force_projection: bool = True,
    apply_mask: bool = True,
    fov_x: Optional[Union[Number, torch.Tensor]] = None,
    use_fp16: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    User-friendly inference function

    ### Parameters
    - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
    - `num_tokens`: the number of base ViT tokens to use for inference, `'least'` or `'most'` or an integer. Suggested range: 1200 ~ 2500. 
        More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: `'most'`. 
    - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
    - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
    - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
    - `use_fp16`: if True, use mixed precision to speed up inference. Default: True
        
    ### Returns

    A dictionary containing the following keys:
    - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
    - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
    - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
    """
    if image.dim() == 3:
        omit_batch_dim = True
        image = image.unsqueeze(0)
    else:
        omit_batch_dim = False
    image = image.to(dtype=model.dtype, device=model.device)

    original_height, original_width = image.shape[-2:]
    area = original_height * original_width
    aspect_ratio = original_width / original_height
    
    # Determine the number of base tokens to use
    if num_tokens is None:
        min_tokens, max_tokens = model.num_tokens_range
        num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

    # Forward pass
    with torch.autocast(device_type=model.device.type, dtype=torch.float16, enabled=use_fp16 and model.dtype != torch.float16):
        output = model.forward(image, num_tokens=num_tokens)
    points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])

    # Always process the output in fp32 precision
    points, normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, normal, mask, metric_scale, fov_x])
    with torch.autocast(device_type=model.device.type, dtype=torch.float32):
        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None
            
        if points is not None:
            # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
            # NOTE: Focal here is the focal length relative to half the image diagonal
            if fov_x is None:
                # Recover focal and shift from predicted point map
                focal, shift = recover_focal_shift(points, mask_binary)
            else:
                # Focal is known, recover shift only
                focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _, shift = recover_focal_shift(points, mask_binary, focal=focal)
            fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
            intrinsics = utils3d.pt.intrinsics_from_focal_center(fx, fy, torch.tensor(0.5, device=points.device, dtype=points.dtype), torch.tensor(0.5, device=points.device, dtype=points.dtype))
            points[..., 2] += shift[..., None, None]
            if mask_binary is not None:
                mask_binary &= points[..., 2] > 0        # in case depth is contains negative values (which should never happen in practice)
            depth = points[..., 2].clone()
        else:
            depth, intrinsics = None, None

        # If projection constraint is forced, recompute the point map using the actual depth map & intrinsics
        if force_projection and depth is not None:
            points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)

        # Apply metric scale
        if metric_scale is not None:
            if points is not None:
                points *= metric_scale[:, None, None, None]
            if depth is not None:
                depth *= metric_scale[:, None, None]

        # Apply mask
        if apply_mask and mask_binary is not None:
            points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
            depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None
            normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None
                
    return_dict = {
        'points': points,
        'points_conf': mask,
        'intrinsics': intrinsics,
        'depth': depth,
        'mask': mask_binary,
        'normal': normal
    }
    return_dict = {k: v for k, v in return_dict.items() if v is not None}

    if omit_batch_dim:
        return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

    return return_dict


@dataclass
class MoGeModelReconstructOutput(AgentContext):
    """
    depth (torch.Tensor): Shape `(H, W)`. The scene's metric depth map, with units in meters. Each value corresponds to the distance along the Z-axis from a point in the scene to the camera.
    mask (torch.Tensor): Shape `(H, W)`. Identifies the valid foreground pixel regions where the model considers its geometric estimates to be reliable.
    """
    points: torch.Tensor
    points_conf: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    mask: torch.Tensor
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor

    # not exposed to Planner
    _image: List[Image.Image]
    _image_tensor: torch.Tensor
    _depth: torch.Tensor
    _transform_info: Dict

    def to_message_content(self) -> str:
        return f"Successfully performed metric reconstruction."


@dataclass
class MoGeModelTensorTransformOutput(AgentContext):
    """
    transformed_tensor (torch.Tensor): The transformed tensor. Its spatial dimensions (H, W) now align with the VGGT model's preprocessed input space, making it suitable for direct use with outputs like `world_points`. The other dimensions are preserved.
    """
    transformed_tensor: torch.Tensor

    def to_message_content(self) -> str:
        shape_str = 'x'.join(map(str, self.transformed_tensor.shape))
        return f'Successfully transformed tensor to shape: {shape_str}.'


@dataclass
class MoGeModelProjectionOutput(AgentContext):
    """
    points_3d (torch.Tensor): Shape `(N, 3)`. A tensor of 3D world coordinates corresponding to the input 2D box.
    points_confidence (torch.Tensor): Shape `(N,)`. Confidence scores for each projected 3D point.
    """
    points_3d: torch.Tensor
    points_confidence: torch.Tensor

    # not expose to planner
    _points_rgb: torch.Tensor

    def to_message_content(self) -> str:
        num_points = self.points_3d.shape[0]

        if num_points == 0:
            return 'Failed: Projection resulted in 0 points. The input mask may be empty or not overlap with the reconstructed scene.'

        avg_conf = torch.mean(self.points_confidence).item()
        return f"Projected {num_points} points into 3D with an average confidence of {avg_conf:.2f}."


def moge_preprocess_image(image: Image.Image, max_size: int = 800):
    """
    Preprocesses a single image for MoGe model inference, resizing based on the longest dimension,
    which is consistent with the official MoGe implementation.
    Args:
        image (Image.Image): Input PIL image.
        max_size (int): The maximum size of the longer side of the image.

    Returns:
        torch.Tensor: A preprocessed image tensor (C, H, W) normalized to [0, 1].
    """
    image_np = np.array(image.convert('RGB'))
    
    h, w, _ = image_np.shape
    larger_size = max(h, w)
    transform_info = {
        'original_shape': (w, h),
        'preprocessed_shape': None,
        'crop_box': None,
        'pad_box': None,
        'multi_image_padding': None,
    }
    
    if larger_size > max_size:
        scale = max_size / larger_size
        target_h, target_w = int(h * scale), int(w * scale)
        # Use INTER_AREA for downscaling to avoid artifacts, as in the official demo
        image_np = cv2.resize(image_np, (target_w, target_h), interpolation=cv2.INTER_AREA)
        transform_info['preprocessed_shape'] = (target_w, target_h)

    # Convert to tensor, permute to (C, H, W), and normalize to [0, 1]
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image_tensor, transform_info


@serve.deployment
class MoGeModel(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = 20.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 4

    MODEL_ID = 'Ruicheng/moge-2-vitl-normal'
    DEVICE = 'cuda'
    DTYPE = torch.float16

    def __init__(self, image_loader: ImageLoader, sam2: SAM2Model):
        super().__init__()

        self.model = MoGe.from_pretrained(self.MODEL_ID).to(self.DEVICE).eval()
        if self.DTYPE == torch.float16:
            self.model.half()

        self.image_loader = image_loader
        self.sam2 = sam2

    @torch.no_grad()
    def _reconstruct(self, image: Image.Image) -> MoGeModelReconstructOutput:
        image_tensor, transform_info = moge_preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.DEVICE)

        if self.DTYPE == torch.float16:
            image_tensor = image_tensor.half()
        
        with torch.amp.autocast(self.DEVICE, dtype=self.DTYPE):
            predictions = infer(self.model, image_tensor, apply_mask=False)

        camera_extrinsics = torch.tensor([[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]], dtype=torch.float)
        return MoGeModelReconstructOutput(
            points=predictions['points'].cpu().detach(),
            points_conf = predictions['points_conf'].cpu().detach(),
            depth=predictions['depth'].cpu().detach(),
            normal=predictions['normal'].cpu().detach(),
            mask=predictions['mask'].cpu().detach(),
            intrinsics=predictions['intrinsics'].cpu().detach(),
            extrinsics=camera_extrinsics,
            _image=image,
            _image_tensor=image_tensor.cpu().detach(),
            _depth=predictions['depth'].cpu().detach(),
            _transform_info = transform_info,
        )

    @torch.no_grad()
    async def _tensor_transform(
        self,
        tensor: torch.Tensor,
        transform_info: Dict[str, Any],
        interpolation: str = 'auto',
    ) -> AgentToolOutput:
        if interpolation == 'auto':
            interp_mode = 'nearest' if tensor.dtype == torch.bool else 'bilinear'
        else:
            interp_mode = interpolation

        if tensor.dim() == 2: # (H, W)
            tensor_4d = tensor.unsqueeze(0).unsqueeze(0).float()
        elif tensor.dim() == 3: # (H, W, C)
            tensor_4d = tensor.permute(2, 0, 1).unsqueeze(0).float()
        else:
            err_msg = f'Unsupported tensor dimension: {tensor.dim()}'
            return self.error(msg=err_msg)

        preprocessed_shape_hw = transform_info['preprocessed_shape'][::-1] # (H, W)
        resized_tensor = F.interpolate(
            tensor_4d, 
            size=preprocessed_shape_hw, 
            mode=interp_mode, 
            align_corners=(False if interp_mode != 'nearest' else None)
        )

        crop_box = transform_info.get('crop_box', None)
        if crop_box:
            x1, y1, x2, y2 = crop_box
            cropped_tensor = resized_tensor[..., y1:y2, :]
        else:
            cropped_tensor = resized_tensor

        pad_box = transform_info.get('pad_box', None)
        if pad_box:
            padded_tensor = torch.nn.functional.pad(
                cropped_tensor, pad_box, mode="constant", value=0
            )
        else:
            padded_tensor = cropped_tensor

        padding = transform_info.get('multi_image_padding', None)
        if padding:
            final_tensor_4d = F.pad(padded_tensor, padding, mode='constant', value=0)
        else:
            final_tensor_4d = padded_tensor

        final_tensor = final_tensor_4d.squeeze(0)
        if tensor.dim() == 2: # for 2d input, (1, H', W') -> (H', W')
            final_tensor = final_tensor.squeeze(0)
        elif tensor.dim() == 3: # for 3d input (C, H', W') -> (H', W', C)
            final_tensor = final_tensor.permute(1, 2, 0)

        if tensor.dtype == torch.bool:
            final_tensor = (final_tensor > 0.5).bool()
        else:
            final_tensor = final_tensor.to(tensor.dtype)
        output = MoGeModelTensorTransformOutput(transformed_tensor=final_tensor)
        return self.success(result=output)
    
    @torch.no_grad()
    async def _project_2d_mask_to_3d(
        self, 
        reconstruction: MoGeModelReconstructOutput, 
        mask_2d: torch.Tensor,
        selected_index: int = 0,
    ) -> AgentToolOutput:
        points = reconstruction.points[selected_index] # Shape: (H, W, 3)
        points_conf = reconstruction.points_conf[selected_index] # Shape: (H, W)
        image_tensor = reconstruction._image_tensor[selected_index].permute(1, 2, 0) # Shape: (C, H, W) -> (H, W, C)
        transform_info = reconstruction._transform_info
        if mask_2d.shape == points.shape[:2]:
            output = MoGeModelProjectionOutput(
                points_3d=points[mask_2d],
                points_confidence=points_conf[mask_2d],
                _points_rgb=image_tensor[mask_2d],
            )
            return self.success(result=output)

        transform_mask_output = await self._tensor_transform(
            tensor=mask_2d,
            transform_info=transform_info,
            interpolation='nearest'
        )
        if transform_mask_output.err:
            return transform_mask_output
        transform_mask_output = transform_mask_output.result
        
        transformed_mask = transform_mask_output.transformed_tensor.to(points.device)
        if transformed_mask.shape != points.shape[:2]:
            err_msg = (
                f'Shape mismatch after transform! Mask: {transformed_mask.shape}, '
                f'Points: {points.shape[:2]}'
            )
            return self.error(msg=err_msg)

        output = MoGeModelProjectionOutput(
            points_3d=points[transformed_mask],
            points_confidence=points_conf[transformed_mask],
            _points_rgb=image_tensor[transformed_mask],
        )
        return self.success(result=output)

    @AgentTool.document_output_class(MoGeModelReconstructOutput)
    async def reconstruct(
        self,
        image_source: str | Image.Image,
    ) -> AgentToolOutput:
        """
        Reconstructs a 3D scene from a single image with true metric scale (meters). This tool is specialized for tasks requiring real-world measurements and serves as the foundation for the MetricScaleEstimator.
        Args:
            image_source (Image.Image): A single image (PIL object).
        """
        if isinstance(image_source, str):
            image_result = await self.image_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result
        else:
            image = image_source

        moge_output = await asyncio.to_thread(self._reconstruct, image)
        return self.success(result=moge_output)
            
    @AgentTool.document_output_class(MoGeModelProjectionOutput)
    async def project_box_to_3d_points(
        self, 
        reconstruction: MoGeModelReconstructOutput,
        box: torch.Tensor,
        selected_index: int = 0,
    ) -> AgentToolOutput:
        # 1. Call SAM2Model.segment to get the pixel mask
        sam2_output = await self.sam2.segment.remote(
            image_source=reconstruction._image,
            box=box
        )
        if sam2_output.err:
            return sam2_output
        
        sam2_output: SAM2ModelOutput = sam2_output.result
        mask_2d = sam2_output.mask

        # 2. Project the mask to 3d points
        projection_output = await self._project_2d_mask_to_3d(
            reconstruction=reconstruction,
            mask_2d=mask_2d,
            selected_index=selected_index,
        )
        if projection_output.err:
            return projection_output

        return projection_output
