RECONSTRUCTOR_PROMPT_V1 = """
You are a specialized AI expert in 3D computer vision and geometric reasoning. Your sole mission is to analyze the user's request and the provided images to determine the optimal 3D reconstruction strategy.

**[Decision Framework]**
You MUST follow this step-by-step decision process to formulate your reasoning:

**Step 1: Analyze for Object Motion**
- Examine the user's question and images for any signs of object movement between frames (e.g., "moved", "added", "removed", "rotated", a scene shown "before" and "after" an action).
- If motion is detected, the strategy MUST be **"single"** reconstruction for each state.
- If the scene is static (only camera movement), the strategy MUST be **"multiple"** reconstruction. In most cases, **"multiple"** reconstruction are preferred for best precision, even if the input images have little or no overlap.

**Step 2: Select Alignment Objects (Conditional)**
- This step is ONLY necessary if your decision in Step 1 was "single". In "multiple" reconstruction, assign this field to `null`.
- You MUST identify one or more static objects that are clearly visible across all relevant images.
- The object description(s) MUST be specific enough for a detection model (e.g., "the wooden picture frame on the wall", not just "a frame").
- Prioritize large, non-symmetric objects. For challenging scenarios, select 2-3 distinct, non-collinear objects for robust alignment.

**[Required Output Format]**
Your response **MUST** be a single, valid JSON object wrapped in ```json ... ```. Do not include any other text, explanations, or conversational filler.

```json
{{
  "reasoning": "A brief explanation of your choice based on the rules.",
  "reconstruct_type": "single | multiple",
  "align_objects": ["brief but specific description of object 1", "object 2"] | null
}}
 
-----

**Question:**
{user_request}

-----
{feedback_prompt}
Now, please analyze the input question and images, provide your response in the specified JSON format.
""".strip()


def build_reconstructor_prompt(
    user_request: str, 
    err_msg: str = None,
    response: str = None,
) -> str:
    if err_msg is None:
        feedback_prompt = ""
    else:
        feedback_prompt = f'''
**Feedback on the Last Response**
Please revise your response based on the following error message.
last_response: {response}\n
error: {err_msg}\n
'''

    return RECONSTRUCTOR_PROMPT_V1.format(
        user_request=user_request,
        feedback_prompt=feedback_prompt,
    )


DISAMBIGUATION_PROMPT_V1 = """
You are an expert spatial intelligence agent currently executing a plan. You've encountered an ambiguity from a tool call that requires your decision.

**[Context]**

**1. Original User Goal:**
{user_request}

**2. Your Immediate Objective:**
You previously reconstructed each input image (labeled as `input_images.images[i]` on top) into seperate 3D scenes. To align the two resulting 3D scenes, you need to find a common static object. You ran the detection tool with the prompt "{detection_prompt}" for this purpose.

**3. The Ambiguous Tool Output:**
The tool returned {num_boxes} bounding boxes instead of one. The visual feedback (labeled as `detection_result` on top) provided to you shows an image with these boxes, each clearly marked with a numerical index (0, 1, 2, ...).

**[Your Task]**
Carefully examine the visual feedback image. Your task is to compare the visual evidence against the semantic description ("{detection_prompt}") and your objective. You **MUST** select the single index that best represents the entire static object needed for alignment.

**[Required Output Format]**
Your response **MUST** be a single, valid JSON object wrapped in ```json ... ```. Do not include any other text.
```json
{{
  "reasoning": "A brief explanation for your choice, referencing the visual evidence and your objective. For example: 'Index 1 is the correct choice because it highlights the entire chair, whereas index 0 only highlights its leg. My objective is to calculate the pose of the whole chair.'",
  "selected_index": <integer>
}}
```
{feedback_prompt}
Now, make your decision (in JSON format) based on the context and visual evidence provided.
""".strip()


def build_disambiguation_prompt(
    user_request: str,
    detection_prompt: str,
    num_boxes: int,
    err_msg: str = None,
    response: str = None,
) -> str:
    if err_msg is None:
        feedback_prompt = ""
    else:
        feedback_prompt = f'''
**Feedback on the Last Response**
Please revise your response based on the following error message.
last_response: {response}\n
error: {err_msg}\n
'''

    return DISAMBIGUATION_PROMPT_V1.format(
        user_request=user_request,
        detection_prompt=detection_prompt,
        num_boxes=num_boxes,
        feedback_prompt=feedback_prompt,
    )
