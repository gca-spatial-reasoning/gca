import torch
from typing import Callable, Dict, Optional, Tuple

from tools.apis import InputImages, Instruction, InputBBoxes2D
from workflow.config import get_config


SOLVER_PLANNER_PROMPT_V1 = """
You are an expert spatial intelligence agent. Your **ONLY** valid output format is a single, valid JSON code block (wrapped in ```json ... ```). Any other format, including plain text or conversational replies, will result in a system failure.

**[CORE MISSION & THE WORKSPACE]**
Your mission is to generate a sequence of tool calls that rigorously computes the answer. It follows an iterative Plan -> Update cycle, using a workspace as your computational memory.
- **Plan**: Decide the next tool calls based on the goal and current workspace.
- **Update**: Tool results are saved as new variables in the workspace.
- **Repeat**: Continue until the workspace contains enough information to conclude the final answer.

{formatted_variable_docs}
{bbox_value_docs}

**[AVAILABLE TOOLS]**
{formatted_tool_docs}

**[A TYPICAL WORKFLOW]**:
{workflow_prompt}

**[RESPONSE GUIDELINES]**
Your response is parsed programmatically and MUST be a single, valid JSON code block wrapped in ```json ... ```. It contains two parts.

**Part 1: Analysis**, contains following keys:
- `current_situation` (string): Briefly summarize what you know from history. After `ReferenceFrameAnalyst` has run, this **MUST** include your detailed analysis of how you will implement the formalization (e.g., "To implement +Z_ref = -Z_toaster, I need to get the toaster's pose using ObjPoseEstimator...") and what target data is need.
- `next_plan` (string): State the immediate next tool(s) you will call to acquire all necessary data for both world-to-reference transformation and the final calculation, as determined by your analysis.

**Part 2: Tool Calls**, it is a list of tool call objects, each of tool call containing:
- `tool_name` (string): the name of called tool, selected from **[AVAILABLE TOOLS]**.
- `args` (dict): A dictionary of arguments for the tool. Values can be:
    - A primitive type (string, number, boolean).
    - A workspace variable. 
- `output_variable` (string): A unique name for the output variable. It will stored in the workspace with this name.
- `step_id` (int, required): Defines the execution step. All tools with the same `step_id` are executed in parallel. 

**Critical Rules**
- **Mandatory Tool Use**: You **MUST NOT** skip the workflow and jump to a conclusion. Your primary function is to demonstrate the solution path by calling the appropriate tools. Always assume that spatial reasoning questions require computational verification via reference frame analysis, 3D reconstruction (multi-image reconstruction or single-image one + alignment) and `PythonTool.code`.
- Your plan **MUST** only be for the immediate next step.
- Reference workspace variables in tool calls using the `$variable_name`, e.g., `$cat_detection.boxes[0]`.
- User-facing references like "Figure 1" map to the zero-indexed `$input_images.images[0]`.
- Use parallel `step_id' for independent calls to maximize efficiency.
{visual_feedback_prompt}
---- Example Response ----
```json
{{
  "analysis": {{
    "current_situation": "The workspace contains the multi-image reconstruction and detection for toaster. The toaster detection returns 2 ambiguous boxes (likely the sink and a smaller region). I think index 0 should be target toaster. The next goal is to get toaster's pose to construct reference frame formalized by `+Z_ref = -Z_toaster`.",
    "next_plan": "To proceed, I will call `ObjPoseEstimator.predict_obj_pose` to get toaster's pose."
  }},
  "tool_calls": [
    {{
      "step_id": 1,
      "tool_name": "ObjPoseEstimator.predict_obj_pose",
      "args": {{
        "reconstruction": "$reconstruction",
        "box": "$toaster_detection.boxes[0]",
        "selected_index": 0,
      }},
      "output_variable": "toaster_pose"
    }}
  ]
}}
```
---- Example Response (End) ----

**[HISTORY]**
Each `TURN` consists of your `PLAN` and the `TOOL CALL RESULTS` from executing that plan. Here is the history so far:

{history_prompt}
{feedback_prompt}
Please analyze current situation and history messages, and then generate your response. Your plan **MUST** only includes the immediate next one step.
""".strip()


def build_text_driven_workflow_prompt() -> str:
    return """
1. **Calculate Viewpoint**: Call `LanguageToCamera.visualize_camera_layout`. This tool analyzes the user's question to determine the final viewpoint and identifies which input image is most relevant for answering the question.
2. **Generate Final Answer**: Using the results from the previous steps, call `FinalAnswerGenerator.generate` to produce the final, user-facing natural language answer.
""".strip()


def build_image_driven_workflow_prompt() -> str:
    workflow = """
1. **Strictly Follow the Task Formalization**: The input question is pre-formalized and consists of two parts: **Reference Frame Constraint** and **Objective Constraint**. You **MUST** strictly follow this formalization to solve the question.
    - **Reference Frame**: Reference frame is the only one coordinate system that matters for interpreting the final answer (left/right, north/south, etc.). The `formalization` is the equation you must solve with the specified geometric tools. E.g., formalization `+Z_ref = -Z_toaster = North` indicates reference frame is defined by object toaster's local frame, so you **MUST** perform all calculations in toaster's frame, and it's semantic back (`-Z_toaster`) points North.
    - **Objective**: Objective is the ultimate goal of the question. You must calculate this objective within the reference frame.
2. **Geometric Reconstruction in World Frame**: **ALWAYS** call `GeometricReconstructor.reconstruct`. This tool provides the 3D reconstruction context in a unified world frame, bridging the gap between input 2D images and geometric perception.
    - The world frame's coordinate system is defined by camera 0 (camera of `$input_images.images[0]`) and follows OpenCV conventions: +Z_world points camera 0's forward, +Y points camera 0's down, and +X points camera 0's right. 
""".strip()

    if 'MetricScaleEstimator' in get_config().tools_to_use:
        workflow += '\n    - `GeometricReconstructor.reconstruct` produces **scale-ambiguous, unitless** 3D data — its coordinates are not expressed in real-world units such as meters, feet, or inches. If the task or downstream module involves **real-world measurements, physical units** (e.g. meters, m/s, foot, inch), or requires **absolute metric** accuracy, you **MUST** use `MetricScaleEstimator.estimate_scale` to Estimates the metric scale factor for a relative 3D reconstruction.'

    workflow += f"""
3. **Acquire Geometric Data**: Based on user's question and task formalization, plan the necessary tool calls to gather all data required for the final calculation. This involves two parallel goals:
    - **A. Solve for the Reference Frame**: **ALL CALCULATION MUST BE PERFORMED IN REFERENCE FRAME**. The `formalization` mathematically defines the **World-to-Reference** Transformation. To solve this `formalization`, your plan **MUST** gather all the geometric "ingredients" in the world frame (from geometric data in Step 2).
        - If `formalization` involves an object's axes (e.g., `+Z_ref = -Z_toaster = North`), you **MUST** call `ObjPoseEstimator.predict_obj_pose` to solve that object's local frame. The resulting `T_obj2world` is required to establish the reference frame.
        - If `formalization` involves a camera's axes (e.g., `-Z_cam0 = North`), you **MUST** acquire reconstruction context (include extrinsic matrix) to implement the formalization.
        - If `formalization` involves a vector between two objects (e.g., `Centroid(B) - Centroid(A) = North`), your plan **MUST** include calls to `GeometricReconstructor.project_box_to_3d_points` for both object A and B.
    - **B. Solve for the Objective**: Follow the objective to identify the target data that need to be analyzed within the reference frame. Your plan **MUST** include the necessary tool calls (usually `GeometricReconstructor.project_box_to_3d_points`) to acquire this target data in world frame.
4. **Perform Final Calculation in Reference Frame**: Once all required variables are available in the workspace, call `PythonTool.code`. Your task is to construct a clear `context_desc` for `PythonTool.code`. `context_desc` **MUST** detail two things:
    - Explain explicitly how to use the provided context variables to build the World-to-Reference transformation matrix following `formalization`. 
    - State what the target data represents and confirm that it **MUST** be transformed into the Reference Frame before performing any final calculations.
5. Conclude the final answer using `FinalAnswerGenerator.generate`.
""".rstrip()

    return workflow


def build_workflow_prompt(router_decision: Optional[str] = None) -> str:
    if get_config().use_meta_planner:
        if router_decision == 'text-driven':
            prompt = build_text_driven_workflow_prompt()
            prompt = f'''
- **Selected Workflow (`text-driven`)**
{prompt}
'''.strip()
        else:
            prompt = build_image_driven_workflow_prompt()
            prompt = f'''
- **Selected Workflow (`image-driven`)**
{prompt}
'''.strip()

        return prompt

    return build_image_driven_workflow_prompt()


def build_tool_docs(configured_tools: Dict[str, Tuple[Callable, str]]) -> str:
    tool_docs = ['Here are the tools you have access to:\n']
    for tool_name, (_, docstring) in configured_tools.items():
        clean_doc = '\n'.join(line.strip() for line in docstring.strip().split('\n'))
        tool_docs.append(f'--- Tool: {tool_name} ---\n{clean_doc}\n')
    tool_docs = '\n'.join(tool_docs)
    return tool_docs


def build_variable_docs(bbox2d: Optional[torch.Tensor] = None) -> str:
    variable_docs = ['\nThe following variables are pre-loaded into the workspace:\n']
    initial_variables = {
        InputImages: 'input_images', 
        Instruction: 'instruction'
    }
    if bbox2d is not None:
        initial_variables.update(
            {InputBBoxes2D: 'input_bbox2d'}
        )

    for variable_class, variable_name in initial_variables.items():
        clean_doc = '\n'.join(
            line.strip() for line in variable_class.__doc__.strip().split('\n')
        )
        variable_docs.append(
            f'--- Variable: `{variable_name}` (Type: {variable_class.__name__}) ---\n{clean_doc}\n'
        )
    variable_docs = '\n'.join(variable_docs)
    return variable_docs


def build_feedback(err_msg: str = None) -> str:
    if err_msg is None:
        feedback_prompt = ''
    else:
        feedback_prompt = f'''
**[Feedback on the Last Response]**
Please revise your response based on the following error message.
error: {err_msg}
'''
    return feedback_prompt


def build_visual_feedback_prompt() -> str:
    if get_config().enable_visual_feedback:
        prompt = '- If previous results are marked `[AMBIGUITY]` (e.g., multiple objects detected), you **MUST** use the visual feedback and question context to choose the correct item/index for next step. If a tool call fails (`[FAILED]`), you **MUST** analyze the error message and visualization to diagnose the root cause, then formulate a new plan to correct it (e.g., change tool parameters, use a different tool).\n'
    else:
        prompt = ''
    return prompt


def build_solver_planner_prompt(
    configured_tools: Dict[str, Tuple[Callable, str]],
    history_prompt: str,
    bbox2d: Optional[torch.Tensor] = None,
    err_msg: str = None,
    router_decision: Optional[str] = None,
    **kwargs
) -> str:
    bbox_value_docs = ''
    if bbox2d is not None:
        bbox_value_docs = f'\n**[BOUNDING BOX VALUE]**\n{bbox2d.tolist()}'

    if get_config().use_meta_planner:  # mindcube
        if router_decision == 'image-driven':
            tools_to_filter = ['LanguageToCamera']
        else:
            tools_to_filter = ['GeometricReconstructor', 'ObjPoseEstimator', 'PythonTool', 'SemanticDetector']
        filtered_tools = {}
        for tool_name, val in configured_tools.items():
            if not any(f_tool in tool_name for f_tool in tools_to_filter):
                filtered_tools[tool_name] = val
        configured_tools = filtered_tools

    return SOLVER_PLANNER_PROMPT_V1.format(
        formatted_variable_docs=build_variable_docs(bbox2d),
        formatted_tool_docs=build_tool_docs(configured_tools),
        bbox_value_docs=bbox_value_docs,
        workflow_prompt=build_workflow_prompt(router_decision=router_decision),
        visual_feedback_prompt=build_visual_feedback_prompt(),
        history_prompt=history_prompt,
        feedback_prompt=build_feedback(err_msg),
    )
