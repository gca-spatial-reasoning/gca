from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.messages import AnyMessage
from ray import serve

from tools.apis.base import AgentContext, AgentTool, AgentToolOutput, InputImages
from tools.apis.cot_reasoner import CoTReasoner
from workflow.prompts.final_answer import build_final_answer_prompt

__ALL__ = ['FinalAnswer', 'FinalAnswerGenerator']


@dataclass
class FinalAnswer(AgentContext):
    result: AgentContext | Any = None
    natural_language_summary: Optional[str] = None

    def to_message_content(self):
        if self.natural_language_summary:
            return self.natural_language_summary
        
        if isinstance(self.result, AgentContext) and hasattr(self.result, 'to_message_content'):
            summary = self.result.to_message_content()
            return f'Final answer computed. Result summary: {summary}'
        return f'Final answer computed. Result ({type(self.result).__name__}): {self.result}.'


@serve.deployment
class FinalAnswerGenerator(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 8

    def __init__(self, reasoner: CoTReasoner):
        super().__init__()
        self.reasoner = reasoner

    async def generate(
        self,
        answer_variable: AgentContext | Any,
        messages: List[AnyMessage] = None,
        input_images: Optional[InputImages] = None,
    ) -> AgentToolOutput:
        """
        Wraps the final answer variable and generates a user-friendly summary.
        Args: 
            answer_variable (AgentContext | Any): A reference to the workspace variable or its attribute, e.g., `$movement_result.execution_result`.
        """
        summary_prompt = build_final_answer_prompt(answer_variable, messages)
        # images = input_images.images if input_images is not None else None
        cot_output = await self.reasoner.cot_reason.remote(
            prompt=summary_prompt,
            # input_images=images,
        )
        if cot_output.err:
            # If summarization fails, still return the structured result
            final_answer = FinalAnswer(result=answer_variable)
            return self.success(result=final_answer)
        
        summary_text = cot_output.result.content
        final_answer = FinalAnswer(
            result=answer_variable,
            natural_language_summary=summary_text
        )
        return self.success(result=final_answer)
