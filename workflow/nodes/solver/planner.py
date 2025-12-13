import functools
import os
import re
from typing import Any, Callable, Dict, List, Set, Tuple

from langchain_core.messages import (
    AIMessage, 
    AnyMessage, 
    ToolMessage
)
from PIL import Image

from tools import ToolCall
from tools.apis import CoTReasoner, CoTReasonerOutput
from tools.utils.llm_invoke import invoke_with_retry
from tools.utils.mm_utils import visualize_detection
from workflow.config import get_config
from workflow.logging import AgentLogger
from workflow.prompts.solver_planner import build_solver_planner_prompt
from workflow.state import AgentState
from workflow.utils.msg_utils import format_history_messages
from workflow.utils.parse_utils import parse_json_str


class SolverPlanner:

    def __init__(
        self, 
        reasoner: CoTReasoner,
        configured_tools: Dict[str, Tuple[Callable, str]],
        logger: AgentLogger,
    ) -> None:
        self.reasoner = reasoner
        self.logger = logger
        self.configured_tools = configured_tools

    def get_calls_in_last_step(
        self, 
        messages: List[AnyMessage]
    ) -> Tuple[List[str], List[str]]:
        success_call_ids, failed_call_ids = [], []
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                break
            if 'failed' in msg.content.lower() or 'error' in msg.content.lower():
                failed_call_ids.append(msg.tool_call_id)
            else:
                success_call_ids.append(msg.tool_call_id)

        return success_call_ids, failed_call_ids

    def load_visual_feedback(
        self, 
        state: AgentState,
        max_visuals_to_load: int = 8,
    ) -> Dict[str, Image.Image]:
        visualization_dir = self.logger.get_visualization_dir(state['session_id'])
        if not visualization_dir:
            return {}

        messages = state['messages']
        success_call_ids, failed_call_ids = self.get_calls_in_last_step(messages)

        visual_contexts_with_labels = {}
        loaded_images = set()

        def _load_img_with_labels(file_path, call_id):
            label = f'[SUCCESS] {call_id}'
            visual_contexts_with_labels[label] = Image.open(file_path)
            loaded_images.add(call_id)

        # load scene visualization if available
        methods_viz = {
            'GeometricReconstructor.reconstruct': [], 
            'VGGTModel.reconstruct': [], 
            'SceneAligner.align': []
        }
        for file in os.listdir(visualization_dir):
            for method in methods_viz.keys():
                pattern = r'(\d+)_' + method + r'_(.+)\.png'
                matches = re.match(pattern, file)
                if not matches:
                    continue

                step_id, output = matches.group(1), matches.group(2)
                call_id = f'{step_id}/{method}/{output}'
                methods_viz[method].append((call_id, file))
        
        if len(methods_viz['GeometricReconstructor.reconstruct']) > 0:
            viz = methods_viz['GeometricReconstructor.reconstruct']
            for (call_id, file) in viz:
                viz_path = os.path.join(visualization_dir, file)
                _load_img_with_labels(viz_path, call_id)
        elif len(methods_viz['SceneAligner.align']) > 0:
            viz = methods_viz['SceneAligner.align']
            for (call_id, file) in viz:
                viz_path = os.path.join(visualization_dir, file)
                _load_img_with_labels(viz_path, call_id)
        else:
            viz = methods_viz['VGGTModel.reconstruct']
            for (call_id, file) in viz:
                viz_path = os.path.join(visualization_dir, file)
                _load_img_with_labels(viz_path, call_id)

        for call_id in success_call_ids:
            viz_path = os.path.join(visualization_dir, f'{call_id.replace("/", "_")}.png')
            if os.path.exists(viz_path) and call_id not in loaded_images:
                _load_img_with_labels(viz_path, call_id)

        if failed_call_ids:
            variable_to_call_id_map = state.get('variable_to_call_id_map', {})
            call_id_to_input_map = state.get('call_id_to_input_map', {})

            dependency_map: Dict[str, Set[str]] = {}
            for failed_call_id in failed_call_ids:
                queue = [failed_call_id]
                traced_ids = {failed_call_id}

                while queue:
                    current_id = queue.pop(0)
                    input_vars = call_id_to_input_map.get(current_id, [])
                    for var in input_vars:
                        producer_id = variable_to_call_id_map.get(var)
                        if producer_id and producer_id not in traced_ids:
                            dependency_map.setdefault(producer_id, set()).add(failed_call_id)
                            traced_ids.add(producer_id)
                            queue.append(producer_id)

            for dep_id, root_ids in dependency_map.items():
                if dep_id in loaded_images:
                    continue

                viz_path = os.path.join(visualization_dir, f'{dep_id.replace("/", "_")}.png')
                if os.path.exists(viz_path):
                    root_ids_str = ', '.join([f'`{root_id}`' for root_id in list(root_ids)])
                    label = f'[Dependency of FAILED {root_ids_str}] {dep_id}'
                    visual_contexts_with_labels[label] = Image.open(viz_path)
                    loaded_images.add(dep_id)

        if len(visual_contexts_with_labels) > max_visuals_to_load:
            sorted_labels = sorted(
                visual_contexts_with_labels.keys(),
                key=lambda x: int(x.split(' ')[-1].split('/')[0]),
                reverse=True
            )
            final_labels = sorted_labels[:max_visuals_to_load]

            return {label: visual_contexts_with_labels[label] for label in final_labels}
        return visual_contexts_with_labels

    def _verify_response_json(self, response_json: Dict) -> Dict[str, Any]:
        response_keys = ['tool_calls']
        for key in response_keys:
            if key not in response_json:
                err_msg = f'Generated response has not key `{key}`, got {response_json.keys()}'
                raise ValueError(err_msg)
        
        tool_calls = response_json['tool_calls']
        tool_call_keys = ['tool_name', 'args', 'output_variable', 'step_id']
        for tool_call in tool_calls:
            for key in tool_call_keys:
                if key not in tool_call:
                    err_msg = f'Generated `tool_calls` json has not key `{key}`, got {tool_call.keys()}'
                    raise ValueError(err_msg)
                
        if 'analysis' in response_json:
            analysis = response_json['analysis']
            analysis_keys = ['current_situation', 'next_plan']
            for key in analysis_keys:
                if key not in analysis:
                    err_msg = f'Generated `analysis` json has not key `{key}`, got {analysis.keys()}'
                    raise ValueError(err_msg)

        return response_json

    def _parse_llm_response(self, response: CoTReasonerOutput) -> Dict[str, Any]:
        content = response.content.strip()
        reasoning_content = response.reasoning_content

        try:
            response_json, json_str = parse_json_str(content)
            response_json = self._verify_response_json(response_json)
        except Exception as e:
            return {
                'analysis': None,
                'tool_calls': None,
                'content': content,
                'reasoning_content': reasoning_content,
                'err_msg': str(e),
            }

        if 'analysis' in response_json:
            analysis = response_json['analysis']
        else:
            analysis = content.replace(json_str, '')

        return {
            'analysis': analysis,
            'tool_calls': response_json['tool_calls'],
            'content': content,
            'reasoning_content': reasoning_content,
            'err_msg': None
        }

    def _extract_plans(
        self, 
        parsed_response: Dict[str, Any],
    ) -> Tuple[List[AnyMessage], List[ToolCall]]:
        if parsed_response['err_msg'] is not None:
            err_msg = (
                'Your previous response did not follow the required format. Error: '
                f'{parsed_response["err_msg"]}. Please follow the format in instruction and ensure all '
                'required keys in json and the output is enclosed in ```json ... ```.'
            )
            raise RuntimeError(err_msg)

        parsed_tool_calls, standard_tool_calls = parsed_response['tool_calls'], []
        for call in parsed_tool_calls:
            args_with_metadata = call.get('args', {}).copy()
            args_with_metadata['_output_variable'] = call.get('output_variable')
            args_with_metadata['_step_id'] = call.get('step_id')
            call_id = f'{call.get("step_id")}/{call.get("tool_name")}/{call.get("output_variable")}'
            standard_tool_calls.append({
                'name': call.get('tool_name'),
                'args': args_with_metadata,
                'id': call_id
            })
        ai_message = AIMessage(
            content=[parsed_response['analysis']],
            tool_calls=standard_tool_calls,
            additional_kwargs={
                '_type': 'solver_planning',
                'reasoning_content': parsed_response['reasoning_content']
            },
        )

        current_plan = [
            ToolCall(
                step_id=call.get('step_id'),
                tool_name=call.get('tool_name'),
                output_variable=call.get('output_variable'),
                args=call.get('args', {}),
            ) for call in parsed_tool_calls
        ]

        return [ai_message], current_plan
    
    async def parse_llm_plan(
        self, 
        output: CoTReasonerOutput,
    ) -> Tuple[List[AnyMessage], List[ToolCall]]:
        parsed_response = self._parse_llm_response(output)
        return self._extract_plans(parsed_response)

    async def plan(self, state: AgentState) -> Dict[str, Any]:
        workspace = state.get('workspace', {})
        messages = state.get('messages', [])

        input_images = None
        if 'input_images' in workspace:
            input_images = workspace['input_images'].images
        
        other_images = None
        if get_config().enable_visual_feedback:
            other_images = self.load_visual_feedback(
                state, get_config().max_visuals_to_load
            )

        bbox2d = None
        if get_config().benchmark == 'cvbench' and 'input_bbox2d' in workspace:
            bbox2d = workspace['input_bbox2d'].bbox2d
            input_bbox2d_image = visualize_detection(
                input_images[0], bbox2d, [""] * len(bbox2d)
            )
            other_images = {
                'Input Bounding Box 2D': input_bbox2d_image
            }

        history_prompt = format_history_messages(messages)

        router_decision = None
        task_router_output = workspace.get('task_router_output')
        if task_router_output:
            router_decision = task_router_output.decision

        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=input_images,
            other_images=other_images,
        )

        prompter = functools.partial(
            build_solver_planner_prompt,
            configured_tools=self.configured_tools,
            history_prompt=history_prompt,
            bbox2d=bbox2d,
            router_decision=router_decision
        )

        try:
            messages, current_plan = await invoke_with_retry(
                invoker=invoker,
                prompter=prompter,
                parser=self.parse_llm_plan,
                max_retries=3
            )

        except Exception as e:
            raise RuntimeError(str(e))

        if get_config().enable_logging:
            session_id = state['session_id']
            self.logger.log_planning(session_id, current_plan, router_decision=router_decision)
            self.logger.log_messages(session_id, messages, history_prompt)
    
        return {
            'messages': messages,
            'current_plan': current_plan
        }
