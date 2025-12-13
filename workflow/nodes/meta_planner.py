from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.messages import AIMessage
from ray.serve.handle import DeploymentHandle

from tools.apis import AgentContext
from workflow.logging import AgentLogger
from workflow.prompts.meta_planner import build_router_prompt
from workflow.state import AgentState


@dataclass
class TaskRouterOutput(AgentContext):
    decision: str

    def to_message_content(self) -> str:
        return f"Task type determined: {self.decision}"


class MetaPlanner:
    
    def __init__(self, reasoner: DeploymentHandle, logger: AgentLogger):
        self.reasoner = reasoner
        self.logger = logger

    async def decide(self, state: AgentState) -> Dict[str, Any]:
        workspace = state.get('workspace', {})
        instruction = workspace['instruction'].text

        try:
            prompt = build_router_prompt(instruction)
            cot_output = await self.reasoner.cot_reason.remote(prompt=prompt)
            
            if cot_output.err:
                raise RuntimeError(cot_output.err['msg'])

            response = cot_output.result
            if not hasattr(response, 'content'):
                raise TypeError('CoTReasoner result missing `content`')

            raw_content = response.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            decision_raw = raw_content.strip().lower()
            decision_token = decision_raw.split()[0].strip("`'\".,;:!?()[]{}") if decision_raw else ''

            if decision_token == 'yes':
                decision = 'text-driven'
            else:
                decision = 'image-driven'

            router_output = TaskRouterOutput(decision=decision)
            self.logger.log_router_decision(
                state['session_id'],
                decision=decision,
                raw_content=raw_content,
            )
        except Exception as e:
            decision = 'image-driven'
            router_output = TaskRouterOutput(decision=decision)
            self.logger.log_router_decision(
                state['session_id'],
                decision=decision,
                err=str(e),
            )
            self.logger.log_messages(
                state['session_id'],
                AIMessage(f'[MetaPlanner] fallback to image-driven, err: {str(e)}')
            )

        msg = AIMessage(f'[MetaPlanner] decision: {decision}')
        self.logger.log_messages(state['session_id'], msg)

        updated_workspace = {**workspace, 'task_router_output': router_output}

        return {
            'workspace': updated_workspace,
            'messages': [msg],
        }
