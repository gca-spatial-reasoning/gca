import asyncio
from dataclasses import dataclass
import inspect
from typing import Dict

import cv2
import numpy as np
from PIL import Image
from ray import serve
import torch

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.io import ImageLoader

__ALL__ = ['OpticalFlowTool', 'OpticalFlowOutput']


@dataclass
class OpticalFlowOutput(AgentContext):
    """
    mean_flow (torch.Tensor): Shape `(2,)`. The average motion vector `(dx, dy)` across the entire image. `dx` represents the average horizontal movement and `dy` represents the average vertical movement of pixels from the first image to the second.
    avg_magnitude (float): The average length of the motion vectors across all pixels, indicating the overall intensity of the movement in the scene.
    """
    mean_flow: torch.Tensor
    avg_magnitude: float

    # not exposed to planner
    _flow: torch.Tensor

    def to_message_content(self) -> str:
        if self.avg_magnitude < 0.5:  # Threshold for negligible motion
            return (
                f'Optical flow analysis detected negligible scene movement (average magnitude: {self.avg_magnitude:.2f} pixels). '
                'The camera was likely stationary.'
            )

        dx, dy = self.mean_flow.tolist()
        
        # Determine dominant directions
        if dy < 0:
            pixel_move_v = 'upward'
            camera_move_v = 'downward'
        else:
            pixel_move_v = 'downward'
            camera_move_v = 'upward'
        
        if dx > 0:
            pixel_move_h = 'rightward'
            camera_move_h = 'leftward'
        else:
            pixel_move_h = 'leftward'
            camera_move_h = 'rightward'

        abs_dx, abs_dy = abs(dx), abs(dy)

        if abs_dy > abs_dx * 2:
            pixel_dominant_direction = pixel_move_v
            camera_dominant_direction = camera_move_v
        elif abs_dx > abs_dy * 2:
            pixel_dominant_direction = pixel_move_h
            camera_dominant_direction = camera_move_h
        else:
            pixel_dominant_direction = f'Both {pixel_move_v} + {pixel_move_h}'
            camera_dominant_direction = f'Both {camera_move_v} + {camera_move_h}'

        return (
            f'Optical flow analysis:\n- Pixel movement: {pixel_dominant_direction}\n'
            f'- Camera movement: {camera_dominant_direction}\nAverage pixel displacement: '
            f'{self.avg_magnitude:.2f} pixels.'
        )


@serve.deployment
class OpticalFlowTool(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 0
    AUTOSCALING_MAX_REPLICAS = 2

    def __init__(self, image_loader: ImageLoader):
        super().__init__()
        self.image_loader = image_loader

    def _calculate(self, image1: Image.Image, image2: Image.Image) -> OpticalFlowOutput:
        # Convert images to grayscale NumPy arrays
        prev_gray = np.array(image1.convert('L'))
        current_gray = np.array(image2.convert('L'))

        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=current_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Calculate mean flow vector (dx, dy)
        mean_flow_np = np.mean(flow, axis=(0, 1))
        # Calculate magnitude of flow vectors and their average
        magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        avg_magnitude = np.mean(magnitudes)

        return OpticalFlowOutput(
            mean_flow=torch.from_numpy(mean_flow_np).float(),
            avg_magnitude=float(avg_magnitude),
            _flow=torch.from_numpy(flow).float(),
        )

    @AgentTool.document_output_class(OpticalFlowOutput)
    async def analyze_motion(
        self,
        image_source_1: str | Image.Image,
        image_source_2: str | Image.Image,
    ) -> AgentToolOutput:
        """
        Analyzes the motion between two sequential images to infer camera movement, especially for subtle or minor camera movements that 3D reconstruction might miss.
        Args:
            image_source_1 (Image.Image): The first image (earlier in time) of the sequence.
            image_source_2 (Image.Image): The second image (later in time) of the sequence.
        """
        images, load_tasks = [None, None], []
        for i, source in enumerate([image_source_1, image_source_2]):
            if isinstance(source, str):
                load_tasks.append(self.image_loader.load_image.remote(source))
            else:
                images[i] = source
        if load_tasks:
            loaded_results = await asyncio.gather(*load_tasks)
            img_idx = 0
            for i in range(len(images)):
                if images[i] is None:
                    result = loaded_results[img_idx]
                    if result.err:
                        return result
                    images[i] = result.result
                    img_idx += 1

        flow_output = await asyncio.to_thread(self._calculate, images[0], images[1])
        return self.success(result=flow_output)
