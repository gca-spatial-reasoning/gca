from dataclasses import dataclass
import functools
import os
from typing import List, Set

from PIL import Image
from ray import serve

from tools.apis.base import AgentContext, AgentTool, AgentToolOutput
from tools.apis.cot_reasoner import CoTReasoner, CoTReasonerOutput
from tools.utils.llm_invoke import invoke_with_retry
from workflow.config import get_config
from workflow.prompts.ref_frame import build_ref_frame_prompt
from workflow.utils.parse_utils import parse_json_str

__ALL__ = ['ReferenceFrameAnalyst', 'ReferenceFrameConstraint']


@dataclass
class ReferenceFrameConstraint(AgentContext):
    """
    formalization (str): The precise mathematical mapping of a semantic direction to a solvable geometric vector.
    """
    formalization: str
    
    _reasoning: str
    _anchor: str
    _type: str

    def to_message_content(self) -> str:
        msg = f'Reference Frame is defined by a(n) {self._type}: {self.formalization}.'
        if self._type == 'object_axes':
            msg = f'{msg}\nYou are required to call `ObjPoseEstimator.predict_obj_pose` to solve this formalization.'
        elif self._type == 'camera_axes':
            msg = f'{msg}\nYou are required to acquire camera\'s extrinsic from reconstruction context to solve this formalization.'
        else:
            msg = f'{msg}\nYou are required to provide the 3D position/point cloud of both objects (first detect then project_box_to_3d_points) to solve this formalization.'

        return msg

    def get_computation_doc(self) -> Set[str]:
        docs = set(['rotation', 'homo_coord'])
        if self._type == 'object_axes':
            docs.update(['obj_coord_sys', 'obj_pose', 'ref_obj'])
        if self._type == 'camera_axes':
            docs.update(['extrinsic', 'ref_cam'])
        if self._type == 'inter_object_vec':
            docs.update(['ref_dir'])
        return docs


@serve.deployment
class ReferenceFrameAnalyst(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 8

    MAX_RETRIES = 3

    def __init__(self, reasoner: CoTReasoner):
        super().__init__()
        self.reasoner = reasoner

    async def _parse_response(self, output: CoTReasonerOutput) -> ReferenceFrameConstraint:
        data, _ = parse_json_str(output.content)

        required_keys = [
            'anchor',
            'reasoning',
            'formalization',
            'primitive_type'
        ]
        if not all(key in data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in data]
            raise ValueError(f'JSON is missing required keys: {missing_keys}')

        return ReferenceFrameConstraint(
            formalization=data['formalization'],
            _reasoning=data['reasoning'],
            _anchor=data['anchor'],
            _type=data['primitive_type'],
        )

    @AgentTool.document_output_class(ReferenceFrameConstraint)
    async def analyze(
        self, 
        instruction: str,
        input_images: Image.Image | List[Image.Image],
    ) -> AgentToolOutput:
        """
        Analyzes the user's question to define the reference frame for subsequent calculation.
        """
        other_images = None
        if get_config().use_visual_in_context_examples:
            assets_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'assets', 'visual_in_ctx_examples'
            )
            total_examples = [
                file for file in os.listdir(assets_dir) if file.startswith('ref_example')
            ]

            other_images = {}
            for i in range(len(total_examples)):
                label = f'[Image for Example {i + 1}]'
                other_images[label] = Image.open(os.path.join(assets_dir, f'ref_example{i+1}.png'))
        
        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=input_images,
            other_images=other_images
        )

        prompter = functools.partial(
            build_ref_frame_prompt,
            user_request=instruction
        )

        try:
            parsed_output = await invoke_with_retry(
                invoker=invoker, 
                prompter=prompter, 
                parser=self._parse_response,
                max_retries=self.MAX_RETRIES
            )
            return self.success(result=parsed_output)

        except Exception as e:
            err_msg = f'Failed to get a valid reference frame. Last error: {e}'
            return self.error(msg=err_msg)
