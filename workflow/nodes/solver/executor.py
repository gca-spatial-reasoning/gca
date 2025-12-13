import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AnyMessage, ToolMessage

from tools import ParameterResolver, ToolCall
from tools.apis import AgentContext
from tools.apis.python_tool import build_context_documentation
from workflow.config import get_config
from workflow.logging import AgentLogger
from workflow.state import AgentState


@dataclass
class ExecutionResult:
    call_success: bool
    call: ToolCall
    result: Optional[AgentContext] = None
    err_msg: Optional[str] = None


class SolverExecutor:

    def __init__(
        self, 
        configured_tools: Dict[str, Tuple[Callable, str]],
        logger: AgentLogger,
    ) -> None:
        """
        Args:
            configured_tools (Dict[str, Tuple[Callable, str]]): A dictionary 
                mapping a tool's call name (e.g., 'VGGTModel.reconstruct') to 
                a tuple containing its callable handle and its pre-fetched 
                docstring, i.e., (callable_method, docstring).
        """
        self.configured_tools = configured_tools
        self.logger = logger
        self.resolver = ParameterResolver()        

    async def execute_block(self, state: AgentState) -> Dict[str, Any]:
        """
        Args:
            state (AgentState): The current state of the LangGraph agent. It expects 
                'current_plan' and 'messages' keys.

        Returns:
            Dict[str, Any]: A dictionary containing updates for 'workspace' and 
                'messages'.
        """
        plan: List[ToolCall] = state.get('current_plan', [])
        if not plan:
            return {}

        grouped_steps = defaultdict(list)
        for call in plan:
            grouped_steps[call.step_id].append(call)
        
        initial_workspace = state.get('workspace', {})
        messages = state.get('messages', [])
        session_id = state.get('session_id')
        variable_to_call_id_map = state.get('variable_to_call_id_map', {}).copy()
        call_id_to_input_map = state.get('call_id_to_input_map', {}).copy()

        block_outputs = {}
        tool_messages = []

        for step_id in sorted(grouped_steps.keys()):
            calls_in_step = grouped_steps[step_id]
            workspace_view = {**initial_workspace, **block_outputs}

            tasks = [
                self.execute_single_call(call, workspace_view, messages)
                for call in calls_in_step
            ]
            step_results: List[ExecutionResult] = await asyncio.gather(*tasks)

            for result in step_results:
                call = result.call
                call_id = call.get_call_id()

                variable_to_call_id_map[call.output_variable] = call_id
                args_str = json.dumps(call.args)
                dependencies = re.findall(r'"\$([a-zA-Z0-9_]+)', args_str)
                call_id_to_input_map[call_id] = list(set(dependencies))

                success, output, tool_message = self.process_result(result)
                tool_messages.append(tool_message)

                viz_path = None
                if success:
                    block_outputs[call.output_variable] = output
                    if get_config().enable_visual_feedback:
                        viz_path = self.logger.log_visualization(
                            session_id, result, workspace_view
                        )

                if get_config().enable_logging:
                    self.logger.log_execution(session_id, result, success, viz_path)
                    self.logger.log_messages(session_id, tool_message)

        final_workspace = initial_workspace.copy()
        final_workspace.update(block_outputs)

        final_state = state.copy()
        final_state.update({
            'messages': tool_messages,
            'workspace': final_workspace,
            'variable_to_call_id_map': variable_to_call_id_map,
            'call_id_to_input_map': call_id_to_input_map,
        })
        return final_state

    async def execute_single_call(
        self, 
        call: ToolCall, 
        workspace: Dict[str, AgentContext],
        messages: List[AnyMessage],
    ) -> ExecutionResult:
        try:
            if call.tool_name not in self.configured_tools:
                err_msg = f'Tool "{call.tool_name}" is not in the configured_tools registry.'
                raise KeyError(err_msg)

            resolved_args = self.resolver.resolve(call.args, workspace)

            if call.tool_name == 'FinalAnswerGenerator.generate':
                call.output_variable = 'final_answer'
                resolved_args['messages'] = messages
                if 'input_images' in workspace:
                    resolved_args['input_images'] = workspace['input_images']

            if call.tool_name == 'PythonTool.code':
                resolved_args['user_request'] = messages[0].content
                resolved_args['context_vars_documentation'] = build_context_documentation(
                    call.args.get('context_vars', {}), 
                    workspace,
                )
                resolved_args['ref_frame_constraint'] = workspace['ref_frame_constraint']
                resolved_args['objective_constraint'] = workspace['objective_constraint']

            if call.tool_name == 'ObjPoseEstimator.predict_obj_pose':
                box_ref = call.args.get('box').strip()
                matches = re.search(r'\$([a-zA-Z0-9_]+)\.boxes\[(\d+)\]', box_ref)
                if matches:
                    detection_ref, box_idx = matches.group(1), matches.group(2)
                    detection = self.resolver.resolve_reference_string(
                        f'${detection_ref}', workspace
                    )
                    obj_label = detection.labels[int(box_idx)]
                else:
                    obj_label = None

                resolved_args['obj_label'] = obj_label

            callable_method, _ = self.configured_tools[call.tool_name]
            output = await callable_method(**resolved_args)

            if output.err:
                error_details = output.err
                err_msg = (
                    f'Tool "{call.tool_name}" failed during execution.\n'
                    f'Source: {error_details.get("src", "Unknown")}\n'
                    f'Msg: {error_details.get("msg", "No message")}'
                )
                return ExecutionResult(call_success=False, call=call, err_msg=err_msg)
            
            return ExecutionResult(call_success=True, call=call, result=output.result)

        except Exception as e:
            err_msg = (
                f'Failed to execute tool "{call.tool_name}".\n'
                f'Error Type: {type(e).__name__}\n'
                f'Message: {str(e)}'
            )
            return ExecutionResult(call_success=False, call=call, err_msg=err_msg)

    def process_result(
        self,
        result: ExecutionResult,
    ) -> Tuple[bool, Optional[AgentContext], ToolMessage]:
        call = result.call
        call_id = call.get_call_id()

        if not result.call_success:
            content = result.err_msg
            success = False
        else:
            content = result.result.to_message_content()
            semantic_success = 'failed' not in content.lower() \
                and 'error' not in content.lower()
            success = semantic_success

        if success:
            output = result.result
            tool_message = ToolMessage(
                content=content,
                status='success',
                tool_call_id=call_id,
                additional_kwargs={'args': call.args},
            )
        else:
            output = None
            tool_message = ToolMessage(
                content=content,
                status='error',
                tool_call_id=call_id,
                additional_kwargs={'args': call.args},
            )
                
        return success, output, tool_message
