import asyncio
from dataclasses import dataclass
import threading
from typing import List, Set

from PIL import Image
from ray import serve
import torch
import torchvision
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.io import ImageLoader
from workflow.config import get_config

__ALL__ = ['GroundingDINOModel', 'GroundingDINOModelOutput']


@dataclass
class GroundingDINOModelOutput(AgentContext):
    """
    boxes (torch.Tensor): Shape `(N, 4)`. `N` detected bounding boxes.
    scores (torch.Tensor): Shape `(N,)`. Confidence score for each detected box.
    labels (List[str]): A list of `N` text label corresponds to each detected box.
    """
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: List[str]

    def to_message_content(self) -> str:
        num_detections = len(self.labels)
        if num_detections == 0:
            return 'Failed: cannot detect any objects. Maybe Consider a different prompt.'
        
        # Summarize the most confident detection
        top_score, top_index = torch.max(self.scores, 0)
        top_label = self.labels[top_index]
        top_box = [round(coord, 2) for coord in self.boxes[top_index].cpu().tolist()]
        summary = (
            f'The most confident detection is a "{top_label}" with a score of {float(top_score):.2f} '
            f'at {top_box}'
        )
        
        if num_detections > 1:
            return (
                f'[AMBIGUITY] Detected {num_detections} objects. '
                'You MUST analyze the visualization to select the correct one for subsequent steps.\n'
                f'{summary}'
            )
        
        return f'Detected 1 object. {summary}'
    
    def get_computation_doc(self) -> Set[str]:
        return set(['boxes'])


@serve.deployment
class GroundingDINOModel(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = 10.0
    AUTOSCALING_MIN_REPLICAS = 2
    AUTOSCALING_MAX_REPLICAS = 8

    MODEL_ID = 'IDEA-Research/grounding-dino-base'
    DEVICE = 'cuda'
    THRESHOLD = 0.2
    TEXT_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.5

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.MODEL_ID, cache_dir=get_config().cache_dir
        ).to(self.DEVICE)
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.image_loader = image_loader

        self.detect_lock = threading.Lock()

    @torch.no_grad()
    def _detect(
        self, 
        image: Image.Image, 
        text_labels: List[str]
    ) -> GroundingDINOModelOutput:
        with self.detect_lock:
            inputs = self.processor(
                images=image, text=text_labels, return_tensors='pt'
            ).to(self.DEVICE)
            
            outputs = self.model(**inputs)
            raw_results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]]
            )[0]

            boxes = raw_results['boxes']
            scores = raw_results['scores']
            labels = [text_labels[0]] * len(scores) 

            if boxes.numel() > 0:
                indices_to_keep = torchvision.ops.nms(
                    boxes, scores, self.NMS_THRESHOLD,
                )
                boxes = boxes[indices_to_keep]
                scores = scores[indices_to_keep]
                labels = [labels[i.cpu().item()] for i in indices_to_keep]

            return GroundingDINOModelOutput(
                boxes=boxes.cpu().detach(),
                scores=scores.cpu().detach(),
                labels=labels,
            )

    @AgentTool.document_output_class(GroundingDINOModelOutput)
    async def detect(
        self, 
        image_source: str | Image.Image, 
        prompt: str | List[str],
    ) -> AgentToolOutput:
        """
        Detects objects in an image using text prompts.
        Args:
            image_source (Image.Image): The `PIL.Image.Image` object.
            prompt (str): The text description of the object to detect.
        """
        if not isinstance(prompt, List):
            prompt = [prompt]

        if isinstance(image_source, Image.Image):
            image = image_source
        else:
            image_result = await self.image_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result

        grounding_dino_output = await asyncio.to_thread(self._detect, image, prompt)
        return self.success(result=grounding_dino_output)
