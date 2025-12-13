from datetime import datetime
import html
import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AnyMessage, 
    AIMessage, 
    HumanMessage, 
    ToolMessage,
)
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from tools import ToolCall, ParameterResolver
from tools.apis import (
    AgentContext,
    EasyOCROutput,
    GeometricProjectionOutput,
    GeometricReconstructionOutput,
    GroundingDINOModelOutput,
    ObjPoseEstimatorOutput,
    OpticalFlowOutput,
    SAM2ModelOutput,
    VGGTModelProjectionOutput,
    VGGTModelReconstructOutput,
    SceneAlignerOutput,
    SemanticDetectorOutput,
)
from tools.utils.mm_utils import (
    visualize_detection,
    visualize_ocr,
    visualize_segmentation,
    visualize_3d_scene,
    visualize_3d_object,
    visualize_obj_pose,
    visualize_optical_flow,
    visualize_aligned_3d_scene,
)
from workflow.config import get_config


def safe_default(o):
    return f'<non-serializable type: {type(o).__name__}>'


class AgentLogger:

    def __init__(self):
        self.resolver = ParameterResolver()
    
    def get_session_dir(self, session_id: str) -> str:
        session_dir = os.path.join(get_config().work_dir, f'session-{session_id}')
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def get_visualization_dir(self, session_id: str) -> Optional[str]:
        if not get_config().enable_visual_feedback:
            return None

        session_dir = self.get_session_dir(session_id)
        visualization_dir = os.path.join(session_dir, 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        return visualization_dir
    
    def _trace_node(self, session_id: str, log_entry: Dict[str, Any]):
        session_dir = self.get_session_dir(session_id)
        log_entry['timestamp'] = datetime.utcnow().isoformat()
        with open(os.path.join(session_dir, 'trace.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, default=safe_default) + '\n')
    
    def log_execution(
        self,
        session_id: str,
        result,
        success: bool,
        viz_path: Optional[str] = None,
    ) -> None:
        call: ToolCall = result.call
        log_entry = {
            'event_type': 'executor',
            'tool_name': call.tool_name,
            'call_id': call.get_call_id(),
            'status': 'success' if success else 'failure',
            'input_args': call.args,
        }
        if success:
            log_entry['result_summary'] = result.result.to_message_content()
        elif result.err_msg is not None:
            log_entry['err_msg'] = result.err_msg
        else:
            log_entry['err_msg'] = result.result.to_message_content()

        if viz_path:
            log_entry['viz_path'] = viz_path

        self._trace_node(session_id, log_entry)

    def log_planning(
        self,
        session_id: str,
        plans: List[ToolCall],
        router_decision: str | None = None,
    ) -> None:
        plan_list = []
        for call in plans:
            plan_list.append({
                'step_id': call.step_id,
                'tool_name': call.tool_name,
                'output_variable': call.output_variable,
                'args': call.args
            })

        log_entry = {'event_type': 'planner', 'plans': plan_list}
        if router_decision:
            log_entry['router_decision'] = router_decision
        self._trace_node(session_id, log_entry)

    def log_router_decision(
        self,
        session_id: str,
        decision: str,
        raw_content: str | None = None,
        err: str | None = None,
    ) -> None:
        log_entry = {
            'event_type': 'meta_planner',
            'decision': decision,
        }
        if raw_content is not None:
            log_entry['raw_content'] = raw_content
        if err is not None:
            log_entry['err'] = err
        self._trace_node(session_id, log_entry)

    def log_visualization(
        self, 
        session_id: str,
        result, 
        workspace_view: Dict[str, AgentContext],
    ) -> Optional[str]:
        call: ToolCall = result.call
        call_id = call.get_call_id()
        viz_dir = self.get_visualization_dir(session_id)
        viz_path = os.path.join(viz_dir, f'{call_id.replace("/", "_")}.png')
        if os.path.exists(viz_path):
            os.remove(viz_path)
        
        try:
            viz_image = None
            match result.result:
                case EasyOCROutput() as ocr_output:
                    image = self.resolver.resolve_reference_string(
                        call.args.get('image_source'), 
                        workspace_view
                    )
                    if isinstance(image, Image.Image):
                        viz_image = visualize_ocr(
                            image=image,
                            boxes=ocr_output.boxes,
                            texts=ocr_output.texts,
                            scores=ocr_output.scores
                        )

                case GeometricProjectionOutput() as projection_output:
                    viz_image = visualize_3d_object(
                        points=projection_output.points_3d,
                        colors=projection_output._points_rgb,
                    )
                
                case GeometricReconstructionOutput() as reconstruction_output:
                    viz_image = visualize_3d_scene(
                        points=reconstruction_output.world_points,
                        points_conf=reconstruction_output.world_points_conf,
                        image_tensor=reconstruction_output._image_tensor,
                        camera_extrinsics=reconstruction_output.extrinsic,
                    )

                case GroundingDINOModelOutput() as detection_output:
                    image = self.resolver.resolve_reference_string(
                        call.args.get('image_source'), 
                        workspace_view
                    )
                    if isinstance(image, Image.Image):
                        viz_image = visualize_detection(
                            image=image, 
                            boxes=detection_output.boxes,
                            labels=detection_output.labels,
                            scores=detection_output.scores,
                        )

                case ObjPoseEstimatorOutput() as pose_output:
                    reconstruction = self.resolver.resolve_reference_string(
                        call.args.get('reconstruction'), 
                        workspace_view
                    )
                    selected_index = call.args.get('selected_index')

                    box_ref = call.args.get('box').strip()
                    matches = re.search(r'\$([a-zA-Z0-9_]+)\.boxes\[(\d+)\]', box_ref)
                    if matches:
                        detection_ref, box_idx = matches.group(1), matches.group(2)
                        detection = self.resolver.resolve_reference_string(
                            f'${detection_ref}', workspace_view
                        )
                        obj_label = detection.labels[int(box_idx)]
                    else:
                        obj_label = None

                    viz_image = visualize_obj_pose(
                        image=to_pil_image(reconstruction._image_tensor[selected_index]),
                        points_3d=reconstruction.world_points[selected_index],
                        image_tensor=reconstruction._image_tensor[selected_index].unsqueeze(0),
                        obb=pose_output._obj_obb,
                        axes=pose_output._obj_axes,
                        extrinsic=reconstruction.extrinsic[selected_index],
                        intrinsic=reconstruction.intrinsic[selected_index],
                        text_label=obj_label,
                    )

                case OpticalFlowOutput() as flow_output:
                    image = self.resolver.resolve_reference_string(
                        call.args.get('image_source_1'), 
                        workspace_view
                    )
                    if isinstance(image, Image.Image):
                        viz_image = visualize_optical_flow(
                            image=image,
                            flow=flow_output._flow,
                        )

                case SAM2ModelOutput() as segmentation_output:
                    image = self.resolver.resolve_reference_string(
                        call.args.get('image_source'), 
                        workspace_view
                    )
                    if isinstance(image, Image.Image):
                        viz_image = visualize_segmentation(
                            image=image, 
                            mask=segmentation_output.mask, 
                            prompt_box=segmentation_output._prompt_box
                        )
                
                case SemanticDetectorOutput() as detection_output:
                    image = self.resolver.resolve_reference_string(
                        call.args.get('image_source'), 
                        workspace_view
                    )
                    if isinstance(image, Image.Image):
                        viz_image = visualize_detection(
                            image=image, 
                            boxes=detection_output.boxes,
                            labels=detection_output.labels,
                        )

                case VGGTModelReconstructOutput() as reconstruction_output:
                    viz_image = visualize_3d_scene(
                        points=reconstruction_output.world_points,
                        points_conf=reconstruction_output.world_points_conf,
                        image_tensor=reconstruction_output._image_tensor,
                        camera_extrinsics=reconstruction_output.extrinsic,
                    )

                case VGGTModelProjectionOutput() as projection_output:
                    viz_image = visualize_3d_object(
                        points=projection_output.points_3d,
                        colors=projection_output._points_rgb,
                    )

                case SceneAlignerOutput() as align_output:
                    ref_scene = self.resolver.resolve_reference_string(
                        call.args.get('reference_scene'),
                        workspace_view
                    )
                    src_scene = self.resolver.resolve_reference_string(
                        call.args.get('source_scene'),
                        workspace_view
                    )
                    viz_image = visualize_aligned_3d_scene(
                        ref_points=ref_scene.world_points,
                        ref_conf=ref_scene.world_points_conf,
                        ref_extrinsic=ref_scene.extrinsic,
                        ref_colors=ref_scene._image_tensor,
                        ref_static_pts=align_output._ref_static_object_points,
                        src_points=src_scene.world_points,
                        src_conf=src_scene.world_points_conf,
                        src_extrinsic=src_scene.extrinsic,
                        src_colors=src_scene._image_tensor,
                        src_static_pts=align_output._src_static_object_points,
                        transform_matrix=align_output.align_transform,
                    )

                case _:
                    pass
            
            if viz_image:
                viz_image.save(viz_path)
                return viz_path
        
        except Exception:
            return

    def _log_human_msg(self, msg: HumanMessage) -> Dict:
        log_entry = {
            'msg_type': 'human',
            'content': msg.content,
        }
        if msg.additional_kwargs.get('retry', -1) > 0:
            log_entry.update({
                'err_output': msg.additional_kwargs['content'],
                'err_reasoning': msg.additional_kwargs['reasoning_content'],
                'retry': msg.additional_kwargs['retry']
            })
        return log_entry

    def _log_ai_msg(self, msg: AIMessage) -> Dict:
        content = msg.content[0] if isinstance(msg.content, List) \
            and len(msg.content) > 0 else msg.content

        log_entry = {'msg_type': 'ai', 'content': content}
        if msg.tool_calls:
            log_entry.update({'tool_call_ids': [call['id'] for call in msg.tool_calls]})
        if msg.additional_kwargs.get('reasoning_content') is not None:
            log_entry['reasoning_content'] = msg.additional_kwargs['reasoning_content']

        return log_entry
    
    def _log_tool_msg(self, msg: ToolMessage) -> Dict:
        return {
            'msg_type': 'tool',
            'tool_call_id': msg.tool_call_id,
            'content': msg.content,
            'status': msg.status,
        }

    def log_messages(
        self, 
        session_id: str, 
        messages: AnyMessage | List[AnyMessage],
        history_prompt: str = None,
    ) -> None:
        if not isinstance(messages, List):
            messages = [messages]
        
        log_entries = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                log_entries.append(self._log_human_msg(msg))
            elif isinstance(msg, AIMessage):
                log_entry = self._log_ai_msg(msg)
                if history_prompt:
                    log_entry['history_prompt'] = history_prompt
                log_entries.append(log_entry)
            elif isinstance(msg, ToolMessage):
                log_entries.append(self._log_tool_msg(msg))
        
        session_dir = self.get_session_dir(session_id)
        with open(os.path.join(session_dir, 'msg.jsonl'), 'a', encoding='utf-8') as f:
            for log_entry in log_entries:
                log_entry['timestamp'] = datetime.utcnow().isoformat()
                f.write(json.dumps(log_entry, ensure_ascii=False, default=safe_default) + '\n')

    def generate_session_report(
        self, 
        session_id: str,
        instruction: str,
        input_images: Optional[str | List[str]] = None,
        ground_truth: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates a rich HTML report for a session, including inline visualizations,
        reasoning traces, and tool outputs. Replaces the abstract call graph.
        """
        session_dir = self.get_session_dir(session_id)
        viz_dir = self.get_visualization_dir(session_id)
        trace_path = os.path.join(session_dir, 'trace.jsonl')
        msg_path = os.path.join(session_dir, 'msg.jsonl')

        if not os.path.exists(trace_path) or not os.path.exists(msg_path):
            return None

        # 1. Load Logs
        with open(trace_path, 'r', encoding='utf-8') as f:
            traces = [json.loads(line) for line in f]
        with open(msg_path, 'r', encoding='utf-8') as f:
            msgs = [json.loads(line) for line in f]

        # 2. Map Visualizations
        viz_map = {}
        if viz_dir and os.path.exists(viz_dir):
            for f in os.listdir(viz_dir):
                if f.endswith('.png') or f.endswith('.jpg'):
                    viz_map[f] = os.path.join('visualizations', f)

        # 3. Helpers
        def _get_ai_msg_for_plan(plan_items):
            """Find the AI message that generated these tool calls."""
            if not plan_items: 
                return None

            # Reconstruct call_id format: step_id/tool_name/output_variable
            target_ids = {f'{p["step_id"]}/{p["tool_name"]}/{p["output_variable"]}' for p in plan_items}
            for msg in msgs:
                if msg.get('msg_type') == 'ai':
                    msg_tool_ids = set(msg.get('tool_call_ids', []))
                    if target_ids & msg_tool_ids: # Intersection exists
                        return msg
            return None

        def _format_python_output(raw_output):
            """Separate code and result for PythonTool."""
            # Regex to split "Execution result: ... Generated Code: ..."
            # Matches pattern in PythonToolOutput.to_message_content
            pattern = r'Execution result: ([\s\S]*?)\nGenerated Code: ([\s\S]*)'
            match = re.search(pattern, raw_output)
            if match:
                result, code = match.groups()
                return True, result.strip(), code.strip()
            return False, raw_output, None

        # 4. Build HTML Components
        html_content = []
        
        # --- Header: User Instruction & Input Images & GT ---
        img_html = ''
        if input_images:
            if isinstance(input_images, str): input_images = [input_images]
            img_divs = []
            for idx, img_path in enumerate(input_images):
                # Copy logic or relative path logic. For report, we assume relative path if inside work_dir, 
                # else we might need absolute or copying. Here simpler: try relative.
                try:
                    rel_path = os.path.relpath(img_path, session_dir)
                except ValueError:
                    rel_path = img_path # Fallback
                
                img_divs.append(f"""
                    <div class="input-img-container">
                        <img src="{rel_path}" class="lightbox-trigger" alt="Input {idx}">
                        <div class="caption">Figure {idx+1}</div>
                    </div>
                """)
            img_html = f'<div class="input-images-row">{"".join(img_divs)}</div>'

        gt_html = ""
        if ground_truth is not None:
            gt_html = f'<div class="gt-box"><strong>🎯 Ground Truth:</strong> {ground_truth}</div>'

        html_content.append(f"""
        <div class="card input-card">
            <div class="card-header">👤 User Instruction</div>
            <div class="card-body">
                <div class="instruction-text">{html.escape(instruction)}</div>
                {img_html}
                {gt_html}
            </div>
        </div>
        """)

        # --- Execution Steps ---
        final_answer_extracted = None
        current_step_html = []
        
        for trace in traces:
            event_type = trace.get('event_type')
            
            if event_type == 'planner':
                # Commit previous step block
                if current_step_html:
                    html_content.append(f'<div class="step-group">{"".join(current_step_html)}</div>')
                    current_step_html = []
                
                plans = trace.get('plans', [])
                ai_msg = _get_ai_msg_for_plan(plans)
                
                # Content (Analysis) - Default Show
                analysis_text = 'No analysis found.'
                reasoning_html = ''
                
                if ai_msg:
                    # Try parsing content if it's JSON-like or just text
                    raw_content = ai_msg.get('content', '')
                    if isinstance(raw_content, Dict):
                        situation_analysis = raw_content.get('current_situation', 'No situation analysis found.')
                        next_plan_analysis = raw_content.get('next_plan', 'No next plan analysis found.')
                        analysis_text = f"""
                        <div class="plan-list">
                            <strong>Analysis</strong>
                            <ul>
                                <li><strong>Current Situation</strong>: {situation_analysis}</li>
                                <li><strong>Next Plan</strong>: {next_plan_analysis}</li>
                            </ul>
                        </div>
                        """.strip()

                    else:
                        analysis_text = html.escape(raw_content)
                    
                    # Reasoning (Thought) - Default Hidden
                    if 'reasoning_content' in ai_msg and ai_msg['reasoning_content']:
                        reasoning_text = html.escape(ai_msg['reasoning_content'])
                        reasoning_html = f"""
                        <details class="reasoning-details">
                            <summary>💭 Show Thinking Process</summary>
                            <div class="reasoning-content">{reasoning_text}</div>
                        </details>
                        """

                # Plan List
                plan_items = []
                for p in plans:
                    plan_items.append(f'<li>Call <b>{p["tool_name"]}</b> &rarr; <code>{p["output_variable"]}</code></li>')
                plan_html = f'<div class="plan-list"><strong>Plan</strong><ul>{"".join(plan_items)}</ul></div>'
                
                current_step_html.append(f"""
                <div class="card planner-card">
                    <div class="card-header">🧠 Agent Planner</div>
                    <div class="card-body">
                        {analysis_text}
                        {plan_html}
                        {reasoning_html}
                    </div>
                </div>
                """)

            elif event_type == 'executor':
                tool_name = trace.get('tool_name')
                status = trace.get('status')
                raw_result = str(trace.get('result_summary', trace.get('err_msg', ''))).strip()
                viz_path = trace.get('viz_path')
                
                # Capture final answer for the verdict card
                if tool_name == 'FinalAnswerGenerator.generate' and status == 'success':
                    # Extract the actual text from the summary format
                    final_answer_extracted = raw_result

                # Visualization
                viz_html = ''
                if viz_path and os.path.exists(viz_path):
                    rel_viz_path = os.path.relpath(viz_path, session_dir)
                    viz_html = f"""
                    <div class="viz-container">
                        <img src="{rel_viz_path}" class="lightbox-trigger" alt="Visual Output">
                    </div>
                    """

                # Special Formatting for PythonTool
                if tool_name == 'PythonTool.code' and status == 'success':
                    is_py, res_txt, code_txt = _format_python_output(raw_result)
                    if is_py:
                        body_html = f"""
                        <div class="code-section">
                            <div class="code-label">Generated Code:</div>
                            <pre class="code-block language-python"><code>{html.escape(code_txt)}</code></pre>
                        </div>
                        <div class="result-section">
                            <div class="result-label">Execution Result:</div>
                            <div class="result-text plain-text">{html.escape(res_txt)}</div>
                        </div>
                        """
                    else:
                        body_html = f'<div class="result-text plain-text">{html.escape(raw_result)}</div>'
                else:
                    # Standard Tool Output (No gray box, just text)
                    body_html = f'<div class="result-text plain-text">{html.escape(raw_result)}</div>'

                status_class = 'success' if status == 'success' else 'error'
                status_icon = '✅' if status == 'success' else '❌'
                
                current_step_html.append(f"""
                <div class="card tool-card {status_class}">
                    <div class="card-header">
                        <span>🔧 {tool_name}</span>
                        <span class="badge {status_class}">{status_icon} {status}</span>
                    </div>
                    <div class="card-body">
                        {body_html}
                        {viz_html}
                    </div>
                </div>
                """)

        if current_step_html:
            html_content.append(f'<div class="step-group">{"".join(current_step_html)}</div>')

        # --- Footer: Final Verdict ---
        verdict_html = ''
        if final_answer_extracted:
            # Simple heuristic check if GT is provided
            verdict_class, verdict_icon, match_text = 'neutral', '🏁', ''
            
            if ground_truth is not None:
                # Basic string inclusion check, can be improved based on benchmark type
                gt_str = str(ground_truth).strip().lower()
                pred_str = str(final_answer_extracted).strip().lower()
                if gt_str in pred_str:
                    verdict_class = 'correct'
                    verdict_icon = '✅'
                    match_text = '<div class="match-success">Match Found</div>'
                else:
                    verdict_class = 'incorrect'
                    verdict_icon = '❌'
                    match_text = '<div class="match-fail">Mismatch</div>'

            verdict_html = f"""
            <div class="card verdict-card {verdict_class}">
                <div class="card-header">{verdict_icon} Final Result</div>
                <div class="card-body">
                    <div class="final-answer-box">
                        <strong>Agent Answer:</strong><br>
                        {html.escape(final_answer_extracted)}
                    </div>
                    {match_text}
                </div>
            </div>
            """
        
        html_content.append(verdict_html)

        # 5. Final Layout & CSS
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Session Report: {session_id}</title>
            <meta charset="utf-8">
            <style>
                :root {{
                    --bg-color: #f4f6f8;
                    --card-bg: #ffffff;
                    --border-color: #e1e4e8;
                    --text-primary: #24292e;
                    --text-secondary: #586069;
                    --accent-blue: #0366d6;
                    --success-bg: #f0fff4;
                    --success-border: #28a745;
                    --error-bg: #fff5f5;
                    --error-border: #d73a49;
                }}
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: var(--bg-color); color: var(--text-primary); padding: 30px; margin: 0; line-height: 1.5; }}
                .container {{ max-width: 1000px; margin: 0 auto; }}
                h1 {{ font-size: 24px; margin-bottom: 20px; color: #1b1f23; }}
                
                /* Cards */
                .card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); overflow: hidden; }}
                .card-header {{ background: #f6f8fa; padding: 12px 16px; font-weight: 600; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; font-size: 14px; }}
                .card-body {{ padding: 16px; }}
                
                /* Input Section */
                .input-card {{ border-left: 4px solid var(--accent-blue); }}
                .instruction-text {{ font-size: 16px; font-weight: 500; margin-bottom: 15px; }}
                .input-images-row {{ display: flex; gap: 15px; overflow-x: auto; padding-bottom: 10px; }}
                .input-img-container {{ text-align: center; }}
                .input-img-container img {{ height: 150px; border-radius: 4px; border: 1px solid #ddd; cursor: zoom-in; transition: transform 0.2s; }}
                .input-img-container img:hover {{ transform: scale(1.02); }}
                .input-img-container .caption {{ font-size: 12px; color: var(--text-secondary); margin-top: 4px; }}
                .gt-box {{ margin-top: 15px; padding: 10px; background: #eef7ff; border-radius: 6px; color: #0366d6; border: 1px solid #c8e1ff; }}

                /* Step Groups */
                .step-group {{ position: relative; margin-bottom: 30px; padding-left: 20px; border-left: 2px solid #e1e4e8; }}
                .step-group::before {{ content: ""; position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #e1e4e8; }}

                /* Planner */
                .planner-card {{ border-left: 4px solid #6f42c1; }}
                .analysis-text {{ margin-bottom: 10px; }}
                .plan-list {{ background: #fafbfc; padding: 10px; border-radius: 6px; border: 1px solid #eee; margin-bottom: 10px; }}
                .plan-list ul {{ margin: 0; padding-left: 20px; }}
                .reasoning-details summary {{ cursor: pointer; color: var(--text-secondary); font-size: 13px; font-weight: 600; outline: none; }}
                .reasoning-content {{ margin-top: 8px; padding: 10px; background: #fffbdd; border-radius: 6px; font-size: 13px; color: #735c0f; white-space: pre-wrap; }}

                /* Tools */
                .tool-card.success {{ border-left: 4px solid var(--success-border); }}
                .tool-card.error {{ border-left: 4px solid var(--error-border); }}
                .badge {{ padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
                .badge.success {{ background: var(--success-bg); color: var(--success-border); border: 1px solid var(--success-border); }}
                .badge.error {{ background: var(--error-bg); color: var(--error-border); border: 1px solid var(--error-border); }}
                
                /* Tool Content Styles */
                .plain-text {{ white-space: pre-wrap; color: var(--text-primary); font-family: inherit; }}
                
                /* Python Code Specific */
                .code-section {{ margin-bottom: 10px; }}
                .code-label {{ font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; }}
                .code-block {{ background: #282c34; color: #abb2bf; padding: 12px; border-radius: 6px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 13px; overflow-x: auto; margin: 0; }}
                .result-section {{ margin-top: 10px; padding-top: 10px; border-top: 1px dashed #eee; }}
                .result-label {{ font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; }}

                /* Visualization */
                .viz-container {{ margin-top: 15px; background: #fafafa; padding: 10px; border-radius: 6px; border: 1px solid #eee; text-align: center; }}
                .viz-container img {{ max-width: 100%; max-height: 500px; border-radius: 4px; cursor: zoom-in; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}

                /* Final Verdict */
                .verdict-card.correct {{ border-left: 4px solid #28a745; background: #f0fff4; }}
                .verdict-card.incorrect {{ border-left: 4px solid #d73a49; background: #fff5f5; }}
                .final-answer-box {{ font-size: 16px; margin-bottom: 10px; }}
                .match-success {{ color: #28a745; font-weight: bold; font-size: 14px; }}
                .match-fail {{ color: #d73a49; font-weight: bold; font-size: 14px; }}

                /* Lightbox */
                .lightbox {{ display: none; position: fixed; z-index: 1000; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); justify-content: center; align-items: center; backdrop-filter: blur(4px); }}
                .lightbox img {{ max-width: 90%; max-height: 90%; border-radius: 4px; box-shadow: 0 5px 30px rgba(0,0,0,0.5); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🕵️ Agent Execution Report</h1>
                <h2><small style="color:#777; font-weight:400; font-size:0.7em">{get_config().benchmark} #{session_id}</small></h2>
                {"".join(html_content)}
            </div>
            
            <div id="lightbox" class="lightbox" onclick="this.style.display='none'">
                <img id="lightbox-img">
            </div>
            <script>
                document.querySelectorAll('.lightbox-trigger').forEach(img => {{
                    img.onclick = (e) => {{
                        e.stopPropagation();
                        document.getElementById('lightbox-img').src = img.src;
                        document.getElementById('lightbox').style.display = 'flex';
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        output_path = os.path.join(session_dir, 'session_report.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        return output_path
