import functools
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch
from PIL import Image
from ray import serve

from tools.apis.base import AgentContext, AgentTool, AgentToolOutput
from tools.apis.cot_reasoner import CoTReasoner
from tools.apis.grounding_dino_model import GroundingDINOModel, GroundingDINOModelOutput
from tools.apis.io import ImageLoader
from tools.utils.llm_invoke import invoke_with_retry
from tools.utils.vlm_as_detector import VLM_AS_DETECTOR
from workflow.config import get_config


__ALL__ = ['SemanticDetector', 'SemanticDetectorOutput']


@dataclass
class SemanticDetectorOutput(AgentContext):
    """
    boxes (torch.Tensor): Shape `(N, 4)`. `N` detected bounding boxes.
    labels (List[str]): A list of `N` text label corresponds to each detected box.
    """
    boxes: torch.Tensor
    labels: List[str]

    # not exposed to Planner
    _detector_type: str

    def to_message_content(self) -> str:
        num_detections = len(self.labels)
        if num_detections == 0:
            return 'Failed: cannot detect any objects. Maybe Consider a different prompt.'
        
        if num_detections > 1:
            return (
                f'[AMBIGUITY] Detected {num_detections} objects for category "{self.labels[0]}". '
                'You MUST analyze the visualization to select the correct one for subsequent steps.'
            )
        
        return f'Detected 1 object for category "{self.labels[0]}".'

    def get_computation_doc(self) -> Set[str]:
        return set(['boxes'])
    

@serve.deployment
class SemanticDetector(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 4
    AUTOSCALING_MAX_REPLICAS = 8

    MAX_RETRIES = 3

    @staticmethod
    def CHECK_OPTIONAL_DINO() -> bool:
        config = get_config()
        vlm_as_detector = get_config().cot_reasoner_model.lower() in VLM_AS_DETECTOR.keys()
        return not (vlm_as_detector and config.use_reasoner_for_detection)
    
    def __init__(
        self,
        image_loader: ImageLoader,
        reasoner: CoTReasoner, 
        dino: 'Optional[GroundingDINOModel]' = None
    ):
        super().__init__()
        self.image_loader = image_loader

        self.use_reasoner_for_detection = not self.CHECK_OPTIONAL_DINO()
        if self.use_reasoner_for_detection:
            self.detector = reasoner
        else:
            if dino is None:
                raise ValueError(
                    'SemanticDetector requires GroundingDINOModel, but it was not '
                    'deployed or injected.'
                )
            self.detector = dino

    async def _detect_with_reasoner(
        self, 
        image_source: Image.Image, 
        prompt: str
    ) -> Dict:
        if isinstance(prompt, list):
            category = ', '.join(prompt)
        else:
            category = prompt

        invoker = functools.partial(
            self.detector.cot_reason.remote,
            input_images=image_source,
            add_label=False 
        )

        model = get_config().cot_reasoner_model.lower()
        prompter, parser = VLM_AS_DETECTOR[model]
        prompter = functools.partial(
            prompter, category=category
        )
        parser = functools.partial(
            parser, image=image_source
        )

        try:
            boxes, labels = await invoke_with_retry(
                invoker=invoker,
                prompter=prompter,
                parser=parser,
                max_retries=self.MAX_RETRIES
            )

            if labels is None:
                labels = [category] * len(boxes)
            detection_output = SemanticDetectorOutput(
                boxes=boxes, 
                labels=labels, 
                _detector_type='cot_reasoner'
            )
            return self.success(result=detection_output)

        except Exception as e:
            err_msg = f'Failed to detect the "{category}". Last error: {e}'
            return self.error(msg=err_msg)

    async def _detect_with_dino(
        self, 
        image_source: Image.Image, 
        prompt: str
    ) -> AgentToolOutput:
        dino_output = await self.detector.detect.remote(
            image_source=image_source,
            prompt=prompt
        )
        if dino_output.err:
            return dino_output

        dino_result: GroundingDINOModelOutput = dino_output.result
        output = SemanticDetectorOutput(
            boxes=dino_result.boxes,
            labels=dino_result.labels,
            _detector_type='dino',
        )
        return self.success(result=output)
    
    @AgentTool.document_output_class(SemanticDetectorOutput)
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
            
        **Note**: If the detection target is **not the entire object but a specific part of it**, you **MUST** ensure that the `prompt` clearly describes that part.  
        For example, instead of `"the car"`, use `"the car’s front wheel"` or `"the door handle of the car"` when only a subcomponent is needed.
        """
        if isinstance(image_source, Image.Image):
            image = image_source
        else:
            image_result = await self.image_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result

        if self.use_reasoner_for_detection:
            return await self._detect_with_reasoner(image, prompt)
        else:
            return await self._detect_with_dino(image, prompt)
