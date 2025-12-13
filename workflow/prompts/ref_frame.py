from workflow.config import get_config


REF_FRAME_PREDICTION_PROMPT_V1 = """
You are an expert spatial reasoning analyst.

**[CORE MISSION]**

Your sole mission is to analyze a user's question and define the final **Reference Frame**.

**Core Principle: Identify the Final Arbiter of Direction.** Your goal is to find the single element (an object, a camera, or a vector) that provides the ultimate, non-negotiable definition for absolute directions (e.g., North, South) or relative perspectives (e.g., "front", "left") in the final answer. 

Ask yourself: *"What element holds the final authority on what 'north', 'front', or 'left' means in the question?"*

You **MUST NOT** attempt to answer the question itself or plan tool usage.

**[OUTPUT FORMAT]**

Your response **MUST** be a single, valid JSON object wrapped in ```json ... ``` code block. Do not include any other text.

```json
{{
  "anchor": "A descriptive name of the anchor, e.g., the \"toaster\", \"camera_0\", or \"movement from A to B\"",
  "reasoning": "A brief, step-by-step logical deduction connecting the question's textual cues to the final geometric formalization, explaining WHY this anchor is the arbiter of direction.",
  "formalization": "The precise mathematical mapping of a semantic direction to one of the **Solvable Geometric Primitive** listed below. E.g., `-Z_cam_0`, `+Z_toaster`, or `Centroid(B) - Centroid(A)`.",
  "pritimive_type": "type of solvable geometric primitive, choices: object_axes | camera_axes | inter_object_vec"
}}
```

**[INSTRUCTIONS & WORKFLOW]**

**1. Identify the Final Arbiter of Direction**:
    - **Priority 1: Absolute Direction.** If an absolute direction (North, South, etc.) is explicitly tied to an element (e.g., "TV faces south," or "moving from A to B, facing north"), that element is the arbiter, overriding everything else.
    - **Priority 2: Relative Query.** If no absolute direction is given, the arbiter is the object of the relative question.
**2. Identify Anchor**: The anchor determines how the reference frame is established. There are three main types:
    - **Object Anchor:** The frame is based on an object's orientation or interaction with it. E.g., "when sitting on the sofa," "when using the toaster", "the picture is on the north wall." or "In the picture,..."
    - **Camera Anchor**: The frame is based on a specific camera's viewpoint. E.g, "from the perspective of Figure 1, ..." or "The Figure 2 faces north."
    - **Direction Anchor:** The frame is based on a described movement or a stated relative position. E.g., "object A is north of object B," "user moves from A to B, facing north."
**3. Formalize Reference Frame Using Solvable Geometric Primitives (CRITICAL)**: You **MUST** formalize abstract descriptions of motion or position into one of the Solvable Geometric Primitives. **DO NOT** create abstract formalizations that are not directly solvable.
    - **Solvable Geometric Primitives**: To construct a mathematical formalization of reference frame, you **MUST** use following three types of directly solvable vectors:
        **A. Camera Axes**: A vector from a specific camera's coordinate system.
            - **Format**: `[+/-]X_cam_[i], [+/-]Y_cam_[i], [+/-]Y_cam_[i]` (e.g., `-Z_cam_0`)
            - This vector can be solved by 3D geometric reconstruction (provides camera extrinsics). Note that in the reconstruction, the coordinate system follows OpenCV convention: 
                - `+Z_cam_[i]` points forward, from the camera i's lens into the scene (towards the background/horizon).
                - `+X_cam_[i]` is to the right in the image.
                - `+Y_cam_[i]` is down in the image.
        **B. Object Axes**: A vector from a specific object's semantic coordinate system.
            - **Format**: `[+/-]X_[obj_name], [+/-]Y_[obj_name], [+/-]Z_[obj_name]` (e.g., `+Z_toaster`).
            - This vector can be solved by object pose estimator tools. Note that in the estimation, the object's local coordinate system is defined by:
                - Origin at centroid. 
                - `+Z_[obj_name]` points its semantic "front". This is the primary functional, interactive, or visual-front direction of the object.
                - `+Y_[obj_name]` points its semantic "down". For most horizontally placed objects, the semantic `top` is the direction opposite to gravity's pull.
                - `+X_[obj_name]` follows the right-hand rule.
        **C. Inter-Object Vector (Direction)**: A vector connecting the centroids of two concrete, detectable objects.
            - **Format**: `Centroid(B) - Centroid(A)`.
            - This vector can be solved by calculation a direction vector from object A/B's point clouds.
    - **Semantic Formalization**: 
        **A. Object Anchors**: Usually can be defined by corresponding object's axes. Examples:
            - "when sitting on the sofa" suggests user's "forward" is the same as the sofa's, i.e., `+Z_ref = +Z_sofa, +Y_ref = +Y_sofa`.
            - "when using the toaster" suggests user's "forward" is opposite the toaster's semantic "front", i.e., `+Z_ref = -Z_toaster, +Y_ref = +Y_toaster`.
            - **CRITICAL** You **MUST** choose a **Distinct, Physical and Detectable** object as the object anchor, **DO NOT** use abstract anchors like room/region/area/wall. You **MUST** use a phsical object's axes or a camera's axes as the proxy to tie the abstract anchor. E.g., "the picture is on the north wall" suggests wall's outward normal vector points North. But the wall is abstract. Use the picture as a proxy. The front of the picture faces away from the north wall. Therefore, the formalization is `+Z_ref = -Z_picture = North`.
        **B. Camera Anchors**: Usually can be defined by corresponding camera's axes. Examples:
            - "from the perspective of Figure 1" suggests reference frame is identical to camera 0's, i.e., `+Z_ref = +Z_cam0, +Y_ref = +Y_cam0`.
        **C. Direction Anchors**:
            - For spatial relationship between two **Distinct, Physical and Detectable Objects**, can be defined by inter-object vector. Examples: "object A is north of object B" suggests the direction from object B to A is north, i.e., `+Z_ref = \\vec{{BA}} = Centroid(A) - Centroid(B) = North`.
            - For spatial relationship between two **Abstract Concepts / Motion**, the primitive of abstract concepts or motions cannot be solved. You **MUST** use a physic object's axes or a camera's axes as the proxy to tie this direction. Examples: "moves from room A to room B, facing north" suggests an abstract motion. Centroid of a room/region/area/wall cannot be solvable. Assume this motion aligns with moving from background towards the foreground, formalized as `+Z_ref = -Z_cam_[i] = North`.
**4. Other Reminder**:
- User-facing name in question like "Figure k," "Image k" **MUST** be mapped to the zero-indexed variable `input_images.images[k - 1]`. Check this label on top of each provided images. `input_images.images[k - 1]` is captured by the `cam_k'` (k' = k - 1).
- The resulting formalization **MUST** be like `[+/-]X/Y/Z_ref = [+/-]X/Y/Z_[obj/cam]` or `[+/-]X/Y/Z_ref = Centroid(A) - Centroid(B)` (both A, B are concrete objects).
- The formalization string **MUST** include any cardinal directions (North, etc.) if specified in the question.
- **DO NOT** simplify the formalization. If the logic dictates `+Z_ref = -Z_toaster = West`, you **MUST** output exactly that. **DO NOT** change it to `+Z_ref = +Z_toaster = East`.

**[EXAMPLES]**
{example_prompt}
-----

**Question:**
{user_request}

-----
{feedback_prompt}
Now, please analyze the above question and provide your response in the specified JSON format.
""".strip()


REF_FRAME_PREDICTION_PROMPT = {
    'v1': REF_FRAME_PREDICTION_PROMPT_V1,
}


def build_feedback(err_msg, response) -> str:
    if err_msg is None:
        feedback_prompt = ""
    else:
        feedback_prompt = f'''
**Feedback on the Last Response**
Please revise your response based on the following error message.
last_response: {response}\n
error: {err_msg}\n
'''
    return feedback_prompt


def build_in_context_examples() -> str:
    if get_config().use_visual_in_context_examples:
        example_prompt = """
**Example 1: Abstract Object Anchor**
**Question**: "The three pictures are hanging on the north wall. What is the position of ...?"
**JSON Output**
```json
{{
  "anchor": "the picture on the north wall",
  "reasoning": "The absolute direction 'North' is tied to an abstract entity, 'the wall,' which is not directly detectable. The prompt instructs to use a physical object as a proxy. The pictures are on the wall, and their semantic 'front' faces away from it. Therefore, the picture's orientation is used to define the reference frame, where the back of the pictures (-Z_picture) points North.",
  "formalization": "+Z_ref = -Z_picture = North",
  "primitive_type": "object_axes"
}}
```
In this case, you can also use the camera which captures the image as the anchor, i.e., `+Z_ref = +Z_cam_k = North`.

**Example 2: Direction Anchor (Phsyical Object)**
**Question**: "Given that the knife is north of the green trash can, what is the orientation of the keyboard relative to the trash can?"
**JSON Output**
```json
{{
  "anchor": "vector from trash can to knife",
  "reasoning": "The absolute direction 'North' is explicitly defined by the relative position of two distinct, physical, and detectable objects: the knife and the trash can. The vector pointing from the trash can to the knife establishes the primary direction for the reference frame.",
  "formalization": "+Z_ref = Centroid(knife) - Centroid(trash can) = North",
  "primitive_type": "inter_object_vec"
}}
```

**Example 3: Direction Anchor (Abstract Motion)**
**Question**: "A person walks from the dining room into the main room with the extinguisher, facing north. From their perspective, what is ...?"
**JSON Output**
```json
{{
  "anchor": "movement from dining room to main room",
  "reasoning": "The anchor is a motion between two abstract regions ('rooms'), which is not directly solvable. The prompt instructs to translate such abstract motion into a camera-relative direction. Assuming the movement from the background (dining room) to the foreground (main room) corresponds to moving towards the camera, this direction is formalized as the camera's negative Z-axis.",
  "formalization": "+Z_ref = -Z_cam_k = North",
  "primitive_type": "camera_axes"
}}

**Example 4: Default and Disambiguation Rules**
**Question**: "In the picture, which side of the box is the underwater robot on??"
**JSON Output**
```json
{{
  "anchor": "camera_0",
  "anchor_type": "camera",
  "reasoning": "The question asks for the relative position ('right' or 'left') of the underwater robot with respect to the box 'in the picture'. This implies a viewer-centric perspective, which defaults to the camera that captured the image. Since no other object or abso
lute direction is specified as the arbiter, the camera's frame is the final authority. In the camera's coordinate system, 'right' corresponds to the +X direction and 'left' to the -X direction.",
  "formalization": "+Z_ref = +Z_cam_0",
  "primitive_type": "camera_axes"
}}
```
"""
    else:
        example_prompt = """
**Question**: "In the scene, the TV faces south. What is the position of the chair relative to the sofa?"
**JSON Output**
```json
{{
  "anchor": "TV",
  "reasoning": "The absolute direction 'South' is explicitly defined by the TV's orientation. The TV is the final arbiter of direction for the entire scene, overriding the relative query. Therefore, the TV is the anchor that sets the coordinate system.",
  "formalization": "+Z_ref = +Z_TV = North",
  "pritimive_type": "object_axes"
}}
```
"""

    return example_prompt


def build_ref_frame_prompt(
    user_request: str,
    err_msg: str = None,
    response: str = None,
) -> str:
    prompt_version = get_config().prompt_version
    prompt = REF_FRAME_PREDICTION_PROMPT.get(
        prompt_version,
        REF_FRAME_PREDICTION_PROMPT['v1']
    )
    return prompt.format(
        user_request=user_request,
        example_prompt=build_in_context_examples(),
        feedback_prompt=build_feedback(err_msg, response),
    )
