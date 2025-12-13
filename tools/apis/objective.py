from dataclasses import dataclass
import functools
from typing import List

from PIL import Image
from ray import serve

from tools.apis.base import AgentContext, AgentTool, AgentToolOutput
from tools.apis.cot_reasoner import CoTReasoner, CoTReasonerOutput
from tools.utils.llm_invoke import invoke_with_retry
from workflow.prompts.objective import build_objective_prompt
from workflow.utils.parse_utils import parse_json_str

__ALL__ = ['ObjectiveAnalyst', 'ObjectiveConstraint']


@dataclass
class ObjectiveConstraint(AgentContext):
    """
    objective (str): The objective of input question.
    """
    objective: str
    _reasoning: str

    def to_message_content(self) -> str:
        return (
            f'Objective: {self.objective}.\nYou are required to acquire target data and perform '
            f'calculation in reference frame to solve this objective.'
        )


@serve.deployment
class ObjectiveAnalyst(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 8

    MAX_RETRIES = 3

    def __init__(self, reasoner: CoTReasoner):
        super().__init__()
        self.reasoner = reasoner

    async def _parse_response(self, output: CoTReasonerOutput) -> ObjectiveConstraint:
        data, _ = parse_json_str(output.content)

        required_keys = [
            'objective',
            'reasoning',
        ]
        if not all(key in data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in data]
            raise ValueError(f'JSON is missing required keys: {missing_keys}')

        return ObjectiveConstraint(
            objective=data['objective'],
            _reasoning=data['reasoning'],
        )

    @AgentTool.document_output_class(ObjectiveConstraint)
    async def analyze(
        self, 
        instruction: str,
        input_images: Image.Image | List[Image.Image],
    ) -> AgentToolOutput:
        """
        Analyzes the user's question to define the objective for subsequent calculation.
        """
        
        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=input_images,
        )

        prompter = functools.partial(
            build_objective_prompt,
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
            err_msg = f'Failed to get a valid objective. Last error: {e}'
            return self.error(msg=err_msg)
