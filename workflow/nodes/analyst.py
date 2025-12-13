import asyncio
from typing import Any, Dict, Tuple

from langchain_core.messages import AIMessage

from tools.apis.base import InputImages, Instruction
from tools.apis.objective import ObjectiveAnalyst, ObjectiveConstraint
from tools.apis.ref_frame import ReferenceFrameAnalyst, ReferenceFrameConstraint
from workflow.config import get_config
from workflow.logging import AgentLogger
from workflow.state import AgentState


class SemanticAnalyst:
    MAX_RETRIES = 3
    
    def __init__(
        self, 
        ref_frame_analyst: ReferenceFrameAnalyst,
        objective_analyst: ObjectiveAnalyst,
        logger: AgentLogger,
    ) -> None:
        self.ref_frame_analyst = ref_frame_analyst
        self.objective_analyst = objective_analyst
        self.logger = logger

    async def _analyze_ref_frame(
        self, 
        instruction: Instruction, 
        input_images: InputImages,
    ) -> Tuple[AIMessage, ReferenceFrameConstraint]:
        for _ in range(self.MAX_RETRIES + 1):
            ref_frame_constraint = await self.ref_frame_analyst.analyze.remote(
                instruction=instruction, input_images=input_images
            )
            if ref_frame_constraint.err is None:
                break
        if ref_frame_constraint.err:
            raise RuntimeError(ref_frame_constraint.err['msg'])

        ref_frame_constraint: ReferenceFrameConstraint = ref_frame_constraint.result
        msg = AIMessage(
            content=ref_frame_constraint.formalization,
            additional_kwargs={
                '_type': 'ref_frame_constraint',
                'reasoning_content': ref_frame_constraint._reasoning,
            }
        )
        return msg, ref_frame_constraint
    
    async def _analyze_objective(
        self, 
        instruction: Instruction, 
        input_images: InputImages,
    ) -> Tuple[AIMessage, ObjectiveConstraint]:
        for _ in range(self.MAX_RETRIES + 1):
            objective_constraint = await self.objective_analyst.analyze.remote(
                instruction=instruction, input_images=input_images
            )
            if objective_constraint.err is None:
                break
        if objective_constraint.err:
            raise RuntimeError(objective_constraint.err['msg'])

        objective_constraint: ObjectiveConstraint = objective_constraint.result
        msg = AIMessage(
            content=objective_constraint.objective,
            additional_kwargs={
                '_type': 'objective_constraint',
                'reasoning_content': objective_constraint._reasoning,
            }
        )
        return msg, objective_constraint

    async def analyze(self, state: AgentState) -> Dict[str, Any]:
        workspace = state.get('workspace', {})

        instruction = workspace['instruction'].text
        input_images = workspace['input_images'].images

        tasks = [
            self._analyze_ref_frame(instruction, input_images),
            self._analyze_objective(instruction, input_images)
        ]
        results = await asyncio.gather(*tasks)

        ref_frame_msg, ref_frame_constraint = results[0]
        objective_msg, objective_constraint = results[1]

        messages = [ref_frame_msg, objective_msg]
        if get_config().enable_logging:
            session_id = state['session_id']
            self.logger.log_messages(session_id, messages)

        final_workspace = workspace.copy()
        final_workspace['ref_frame_constraint'] = ref_frame_constraint
        final_workspace['objective_constraint'] = objective_constraint
        return {
            'messages': messages,
            'workspace': final_workspace,
        }
