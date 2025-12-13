import asyncio
from dataclasses import dataclass
import os
import threading
from typing import Dict

from PIL import Image
from ray import serve
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.io import ImageLoader

__ALL__ = ['SAM2Model', 'SAM2ModelOutput']


@dataclass
class SAM2ModelOutput(AgentContext):
    """
    mask (torch.Tensor): Shape `(H, W)`. A tensor of binary segmentation mask, where `(H, W)` are the dimensions of the original image. The mask is a boolean tensor where `True` indicates a pixel belonging to the segmented object.
    score (torch.Tensor): Shape `(1,)`. Confidence score for generated mask. Higher score indicate a higher quality segmentation.
    """
    mask: torch.Tensor
    score: torch.Tensor
    
    # not exposed to planner
    _prompt_box: torch.Tensor
    
    def _get_mask_bounding_box(self, mask: torch.Tensor) -> torch.Tensor:
        # Find the indices of all non-zero elements (pixels in the mask)
        rows, cols = torch.where(mask)
        if rows.numel() == 0:
            return torch.zeros(4) # Return an empty box if mask is empty
            
        x1, y1 = cols.min(), rows.min()
        x2, y2 = cols.max(), rows.max()
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float)

    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        # Determine the coordinates of the intersection rectangle
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        # Compute the area of intersection
        intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute the area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute the area of union
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return (intersection_area / union_area).item()

    def to_message_content(self) -> str:
        mask_area = torch.sum(self.mask).item()
        confidence_score = self.score.item()

        if mask_area < 10:
            return (
                f'Failed: The generated mask is too small (area: {mask_area} pixels). '
                f'The prompt box may not contain a valid object.'
            )
        if confidence_score < 0.2:
            return (
                f'Failed: Segmentation confidence is too low (score: {confidence_score:.2f}). '
                f'Consider using a more precise bounding box.'
            )
        
        box_list = [round(coord, 2) for coord in self._prompt_box.cpu().tolist()]
        total_area = self.mask.numel()
        percentage = (mask_area / total_area) * 100
        
        # Calculate the IoU between the prompt box and the mask's bounding box
        mask_bbox = self._get_mask_bounding_box(self.mask)
        iou = self._calculate_iou(self._prompt_box.cpu(), mask_bbox)
        
        return (
            f'For input box {box_list}, SAM2 generates mask with a score of {self.score.item():.2f}. '
            f'The mask covers {percentage:.1f}% of the image and has an IoU of {iou:.2f} with '
            f'the prompt box.'
        )


@serve.deployment
class SAM2Model(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = 10.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 8

    DEVICE = 'cuda'

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()
        
        checkpoint = os.path.join(
            os.path.dirname(__file__), '..', 'third_party', 'sam2', 'checkpoints', 'sam2.1_hiera_large.pt'
        )
        model_cfg = os.path.join('configs', 'sam2.1', 'sam2.1_hiera_l.yaml')

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'Please first download SAM2 checkpoints following README')

        self.model = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint, device=self.DEVICE)
        )
        self.image_loader = image_loader
        
        self.segment_lock = threading.Lock()

    @torch.no_grad()
    def _segment(self, image: Image.Image, box: torch.Tensor) -> SAM2ModelOutput:
        # Ensure that `set_image` and `predict` are atomic for each call to prevent 
        # race conditions from parallel invocations.
        with self.segment_lock:
            if box.ndim == 1:
                box = box[None, :]

            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                self.model.set_image(image)
                masks, scores, _ = self.model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box.to(self.DEVICE),
                    multimask_output=False
                )

            return SAM2ModelOutput(
                _prompt_box=box[0].cpu(),
                mask=torch.from_numpy(masks).bool()[0],
                score=torch.from_numpy(scores)
            )

    @AgentTool.document_output_class(SAM2ModelOutput)
    async def segment(
        self, 
        image_source: str | Image.Image, 
        box: torch.Tensor
    ) -> Dict:
        """
        Segments an object in an image based on a bounding box.
        Args:
            image_source (Image.Image): The `PIL.Image.Image` object.
            box (torch.Tensor): A bounding box specifying the object to segment.
        """
        if isinstance(image_source, Image.Image):
            image = image_source
        else:
            image_result = await self.image_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result

        sam2_output = await asyncio.to_thread(self._segment, image, box)
        return self.success(result=sam2_output)
