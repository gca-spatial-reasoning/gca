# ==============================================================================
# 1. Basic Transformation Formulas
# ==============================================================================


EXTRINSIC_TRANSFORM = """
**Extrinsic Transformation (World <-> Camera)**:
    - **World -> Camera**: To transform a world point `P_world` into camera `s`'s frame, use its extrinsic matrix `E_s = extrinsic[s]`. The formula is: `P_cam_homo = P_world_homo @ E_s.T`.
    - **Camera -> World**: The pose of camera `s` in the world, `Pose_s`, is the inverse of its extrinsic matrix: `Pose_s = np.linalg.inv(extrinsic[s])`. To transform a point from camera `s`'s local frame to the world frame, use the formula: `P_world_homo = P_cam_homo @ Pose_s.T`.
    - **Relative Rotation Analysis (Camera -> Camera)**: To describe the rotation of **camera `j`'s pose relative to camera `i`'s pose**, use the camera poses in the world frame (`Pose_i`, `Pose_j`). The relative rotation from `i` to `j` is: `R_rel = R_j_pose @ R_i_pose.T` which simplifies to `R_rel = (R_j.T) @ (R_i.T).T = R_j.T @ R_i`.
""".strip()


OBJ_POSE_TRANSFORM = """
**Object Pose Transformation (World <-> Object)**:
    - **Object -> World**: The `T_obj2world` matrix (aliased as `Pose_obj`) transforms points from the object's local frame to the world frame using the formula: `P_world_homo = P_local_homo @ Pose_obj.T`.
    - **World -> Object**: To transform a world point `P_world` into the object's local frame, use the inverse matrix `T_world_to_obj = np.linalg.inv(Pose_obj)`. The formula is: `P_local_homo = P_world_homo @ T_world_to_obj.T`.
""".strip()


ALIGN_TRANSFORM = """
**Alignment Tranformation Matrix**: The homogeneous form of the `[R | T]` matrix composed of a 3x3 rotation matrix (R) and a 3x1 translation vector (T).
""".strip()


# ==============================================================================
# 2. Reference Frame Translation
# ==============================================================================


REF_FRAME_OBJECT_FORMALIZATION_TRANSLATION = """
**Reference Frames Defined by Object Axes**: For reference frame defined by an object's axes, `[+/-]X/Z_ref = [+/-]X/Z_[obj]`, you **MUST** define a set of orthogonal axes (a basis) in the world frame and use vector projection (dot product). **DO NOT** directly perform world-to-reference transformation.
    - **Step 1: Define Reference Frame Axes**. Follow these steps:
        - **A. Extract the Object's Local Axes from its Pose**. The object's pose `T_obj2world` contains a 3x3 rotation matrix. Its columns are the object's local axes expressed in the world frame: `X_obj_axis = normalize(T_obj2world[:3, 0])`, `Y_obj_axis = normalize(T_obj2world[:3, 1])` (Down), `Z_obj_axis = normalize(T_obj2world[:3, 2])` (Front)
        - **B. Define the Reference Frame Axes from `formalization`**.
            - Use the `formalization` to define the primary unit vector. E.g., `+Z_ref = -Z_obj` denotes `Z_ref_axis = -Z_obj_axis`, `+Z_ref = +X_obj` denotes `Z_ref_axis = X_obj_axis`.
            - Assuming `Y_ref_axis` remains down: `Y_ref_axis = Y_obj_axis`.
            - Calculate the third axis using the cross product to ensure a right-handed system:
                - If `Z_ref_axis` is the primary vector, `X_ref_axis = np.cross(Y_ref_axis, Z_ref_axis)`.
                - If `X_ref_axis` is the primary vector, `Z_ref_axis = np.cross(X_ref_axis, Y_ref_axis)`.
    - **Step 2: Calculate Relative Position using Dot Product**. 
        - **A. Calculate Displacement Vector**. The origin of the reference frame is the object's centroid. Calculate the displacement vector in world frame: `disp_vec_world = target_point_centroid - origin_point = target_point_centroid - T_obj2world[:3, 3]`.
        - **B. Dot Projection**. Project the world displacement vector onto your reference axes: `dx = np.dot(disp_vec_world, X_ref_axis), dy = np.dot(disp_vec_world, Y_ref_axis), dz = np.dot(disp_vec_world, Z_ref_axis)`.
        - **C. Interpretation**. The resulting `disp_vec_world` and `disp_vec_ref` are used for further interpretation. The component (`dx`, `dy`, and `dz`) with the largest absolute value is always considered the primary component of direction. When evaluting compound direction, only consider single direction if the absolute value of secondary component is less than 10% (0.1) of the primary component.
""".strip()


REF_FRAME_CAMERA_FORMALIZATION_TRANSLATION = """
**Reference Frames Defined by Camera Axes**: For a reference frame defined by a camera's axes, such as `[+/-]X/Z_ref = [+/-]X/Z_cam_[i]`, you **MUST** define a set of orthogonal axes (a basis) in the world frame and then use vector projection (dot product). **DO NOT** directly perform world-to-reference transformation.
    - **Step 1: Define Reference Frame Axes**.
        - **A. Extract the Camera's Local Axes from its Pose**. The camera's pose `Pose_cam_i` is the inverse of its extrinsic matrix `extrinsic[i]`. The columns of the pose's rotation matrix are the camera's local axes expressed in the world frame:
            - `Pose_cam_i = np.linalg.inv(extrinsic[i])`
            - `X_cam_axis = normalize(Pose_cam_i[:3, 0])` (Right), `Y_cam_axis = normalize(Pose_cam_i[:3, 1])` (Down), `Z_cam_axis = normalize(Pose_cam_i[:3, 2])` (Forward)
        - **B. Define the Reference Frame Axes from `formalization`**.
            - Use the `formalization` to define the primary unit vector. E.g., `+Z_ref = -Z_cam_0` implies `Z_ref_axis = -Z_cam_axis`.
            - Assume the vertical axis remains unchanged: `Y_ref_axis = Y_cam_axis`.
            - Calculate the third axis using the cross product to ensure a right-handed system:
                - If `Z_ref_axis` is the primary vector, `X_ref_axis = np.cross(Y_ref_axis, Z_ref_axis)`.
                - If `X_ref_axis` is the primary vector, `Z_ref_axis = np.cross(X_ref_axis, Y_ref_axis)`.
    - **Step 2: Calculate Relative Position using Dot Product**.
        - **A. Calculate Displacement Vector**. The origin of the reference frame is the camera's position. Calculate the displacement vector in the world frame: `disp_vec_world = target_point_centroid - origin_point = target_point_centroid - Pose_cam_i[:3, 3]`.
        - **B. Dot Projection**. Project the world displacement vector onto your reference axes: `dx = np.dot(disp_vec_world, X_ref_axis), dy = np.dot(disp_vec_world, Y_ref_axis), dz = np.dot(disp_vec_world, Z_ref_axis)`.
        - **C. Interpretation**. The resulting `disp_vec_world` and `disp_vec_ref` are used for further interpretation. The component (`dx`, `dy`, and `dz`) with the largest absolute value is always considered the primary component of direction. When evaluting compound direction, only consider single direction if the absolute value of secondary component is less than 10% (0.1) of the primary component.
""".strip()


REF_FRAME_DIRECTION_FORMALIZATION_TRANSLATION = """
**Reference Frames Defined by Direction Vector**: For reference frame defined by a direction vector `+Z_ref = Centroid(B) - Centroid(A)`, you **MUST** define a set of orthogonal axes (a basis) in the world frame and use vector projection (dot product). **DO NOT** directly perform world-to-reference transformation.
    - **Step 1: Define Reference Frame Axes**.
        - Use the `formalization` to define the primary unit vector. For example, if `formalization` is `+Z_ref = Centroid(B) - Centroid(A) = North`, calculate `Z_ref_axis = normalize(Centroid(B) - Centroid(A))`.
        - Define the vertical axis as the world's down vector: `Y_ref_axis = np.array([0, 1, 0])`.
        - Calculate the third axis using the cross product to ensure a right-handed system:
            - If `Z_ref_axis` is the primary vector, `X_ref_axis = np.cross(Y_ref_axis, Z_ref_axis)`.
            - If `X_ref_axis` is the primary vector, `Z_ref_axis = np.cross(X_ref_axis, Y_ref_axis)`.
    - **Step 2: Determine the Origin and Target for Calculation from the User's Question**.
        - The **`origin_point`** is the centroid of the object that the question is asking the relation *to*. For a question like "Where is C relative to A?", the `origin_point` is `Centroid(A)`.
        - The **`target_point`** is the centroid of the object whose position is being asked about.
    - **Step 3: Calculate Relative Position using Dot Product**.
        - **A. Calculate Displacement Vector**. Calculate the displacement vector in world frame: `disp_vec_world = target_point - origin_point`.
        - **B. Dot Projection**. Project the world displacement vector onto your reference axes: `dx = np.dot(disp_vec_world, X_ref_axis), dy = np.dot(disp_vec_world, Y_ref_axis), dz = np.dot(disp_vec_world, Z_ref_axis)`.
        - **C. Interpretation**. The resulting `disp_vec_world` and `disp_vec_ref` are used for further interpretation. The component (`dx`, `dy`, and `dz`) with the largest absolute value is always considered the primary component of direction. When evaluting compound direction, only consider single direction if the absolute value of secondary component is less than 10% (0.1) of the primary component.
""".strip()


# ==============================================================================
# 3. Advanced Interpretation Guides
# ==============================================================================


ROTATION_INTERPRETATION = """
**Interpreting Rotation `R`**:
    - **General Rotation Direction**: For most questions about rotation direction, convert the relative rotation matrix `R` to a rotation vector `[rx, ry, rz]` via `scipy.spatial.transform.Rotation.from_matrix(R).as_rotvec()`.
        - The component with the largest absolute value (`rx`, `ry`, or `rz`) indicates the primary axis of rotation.
        - Based on the **OpenCV coordinate system** (+X right, +Y down, +Z forward) and the right-hand rule: `ry > 0` corresponds to a pan to the **right**, `rx > 0` corresponds to a tilt **Upward**, `rz > 0` corresponds to a **clockwise** roll.
    - **Sequential Rotations**: **When** the options explicitly describe a sequence of rotations (e.g., "Rotate Y then X"), you **MUST use the Primary Axis First strategy**:
        - Identify the primary axis using the relative rotation `R`'s rotation vector as described above. Then discard any option that does not start with a rotation around this primary axis.
        - For the remaining options, verify the signs of the angles using `scipy.spatial.transform.Rotation.from_matrix(R).as_euler(order)`. The `order` parameter MUST be a 3-character string. For a sequence like "Rotate Y then X", map it to an order string "yx*" where "*" is the remaining axis, i.e., "yxz". The signs of the resulting angles must match the option's description.
""".strip()


CARDINAL_DIRECTION_INTERPRETATION = """
**Interpreting Coordinates as Cardinal Directions (N, E, S, W)**: After calculating the displacement vector in the world frame (`disp_vec_world`) and the reference frame's orthogonal axes (`X_ref_axis`, `Y_ref_axis`, `Z_ref_axis`), You **MUST** follow these steps to determine the final compass direction.
    - **Step 1: Identify the Cardinal Anchor Axis from the `formalization`**.
        - The `formalization` string (e.g., `+Z_ref = -Z_obj = South`) links one of your reference axes to a cardinal direction.
        - Parse this string to find the anchor. In the example, the anchor is `South`, and it corresponds to the `Z_ref_axis`. This defines your first cardinal vector: `South_axis = Z_ref_axis`.
    - **Step 2: Derive the Complete Set of Cardinal Axes**. You **MUST** use the corresponding formula set below to calculate cardinal axes. They are **ALWAYS** true, **DO NOT** change the order. **DO NOT** relying on logical deducing. Find the remaining cardinal axes via applying cross product:
        - If `South_axis` is known, starting with `West_axis = np.cross(Y_ref_axis, South_axis)`.
        - If `West_axis` is known, starting with `North_axis = np.cross(Y_ref_axis, West_axis)`.
        - If `North_axis` is known, starting with `East_axis = np.cross(Y_ref_axis, North_axis)`.
        - If `East_axis` is known, starting with `South_axis = np.cross(Y_ref_axis, East_axis)`.
    - **Step 3: Project and Determine the Final Quadrant**.
        - Project the `disp_vec_world` onto the primary horizontal cardinal axes (North and East).
            - `projection_north = np.dot(disp_vec_world, cardinal_map["N"])`
            - `projection_east = np.dot(disp_vec_world, cardinal_map["E"])`
        - Use the signs of these projections to determine the final answer.
""".strip()


# ==============================================================================
# 4. Other Miscellaneous Documentation
# ==============================================================================


BBOX_FORMAT = """
**Bounding Box Format**: All bounding box in input variable is provided in the `[x1, y1, x2, y2]` format.
""".strip()


HOMO_COORDINATE = """
**Handling Homogeneous Coordinates**: Our system uses a strict **row vector** convention.
    - To transform points, you **MUST** first convert them from (N, 3) to (N, 4) by appending a column of ones.
    - When applying any 4x4 transformation matrix `T`, you **MUST** use post-multiplication with the **transposed matrix (`T.T`)**.
    - The correct formula is: `P_transformed_homo = P_homogeneous @ T.T`
    - After transformation, convert back to 3D by dividing the first three columns by the fourth column.
""".strip()


OBJ_COORD_SYS_DEFINE = """
**Object Canonical Coordinate System**: The object's local frame is defined with its origin at the centroid, `+Z_[obj_name]` pointing towards the semantic "front", `+Y_[obj_name]` pointing towards the semantic "down", and `+X_[obj_name]` derived from the right-hand rule.
""".strip()


COMPUTATION_DOCS_REGISTRY = {
    'basic_transform': {
        'align_transform': ALIGN_TRANSFORM,
        'extrinsic': EXTRINSIC_TRANSFORM,
        'obj_pose': OBJ_POSE_TRANSFORM,
    },
    'ref_frame_transform': {
        'ref_cam': REF_FRAME_CAMERA_FORMALIZATION_TRANSLATION,
        'ref_dir': REF_FRAME_DIRECTION_FORMALIZATION_TRANSLATION,
        'ref_obj': REF_FRAME_OBJECT_FORMALIZATION_TRANSLATION,
    },
    'interpretation': {
        'cardinal_dir': CARDINAL_DIRECTION_INTERPRETATION,
        'rotation': ROTATION_INTERPRETATION,
    },
    'other': {
        'boxes': BBOX_FORMAT,
        'homo_coord': HOMO_COORDINATE,
    },
    'obj_coord_sys': OBJ_COORD_SYS_DEFINE,
}