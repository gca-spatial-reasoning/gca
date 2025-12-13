from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from ray import serve

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext


__ALL__ = ['LanguageToCamera', 'LanguageToCameraOutput']


def _angle_to_direction(angle: float) -> str:
    """Convert angle (in degrees) to human-readable direction."""
    angle %= 360
    if 337.5 < angle or angle <= 22.5:
        return "Right"
    if 22.5 < angle <= 67.5:
        return "Front-Right"
    if 67.5 < angle <= 112.5:
        return "Front"
    if 112.5 < angle <= 157.5:
        return "Front-Left"
    if 157.5 < angle <= 202.5:
        return "Left"
    if 202.5 < angle <= 247.5:
        return "Back-Left"
    if 247.5 < angle <= 292.5:
        return "Back"
    if 292.5 < angle <= 337.5:
        return "Back-Right"
    return "Unknown"


def _angular_distance(a: float, b: float) -> float:
    """Compute the shortest angular distance between two angles."""
    return abs((a - b + 180) % 360 - 180)


@dataclass
class LanguageToCameraOutput(AgentContext):
    """
    Output containing camera spatial relationships and visualization.
    
    Attributes:
        angles_deg (List[float]): Viewing angles in degrees (90°=Front, 0°=Right, 180°=Left, 270°=Back).
        center_view_index (Optional[int]): Reference camera index for the diagram (defaults to 0).
        labels (List[str]): Camera labels (e.g., ["Image 1", "Image 2"]).
        relationship_summary (str): A natural language summary describing the spatial relationships.
        final_view_angle_deg (Optional[float]): The final viewing angle after rotation.
        relative_rotation_deg (Optional[float]): The rotation applied. Convention: right=-ve, left=+ve.
        target_view_index (Optional[int]): Index of the camera aligning with the question's perspective.
        target_view_label (Optional[str]): Label of the camera aligning with the question's perspective.
    """
    angles_deg: List[float]
    center_view_index: Optional[int]
    labels: List[str]
    relationship_summary: str
    final_view_angle_deg: Optional[float] = None
    relative_rotation_deg: Optional[float] = None
    target_view_index: Optional[int] = None
    target_view_label: Optional[str] = None

    def to_message_content(self) -> str:
        """Returns a summary of spatial relationships from the final reference perspective."""
        center_idx = self.center_view_index if self.center_view_index is not None else 0
        center_label = self.labels[center_idx]

        # Determine the reference angle and the title of the summary
        if self.final_view_angle_deg is not None:
            ref_angle = self.final_view_angle_deg
            rot_deg = self.relative_rotation_deg or 0
            if rot_deg > 0:
                rot_desc = f"{abs(rot_deg)}° left"
            elif rot_deg < 0:
                rot_desc = f"{abs(rot_deg)}° right"
            else:
                rot_desc = "no"
            
            if rot_desc != "no":
                summary_title = f"From the final perspective (based on **{center_label}** with a {rot_desc} turn):"
            else:
                summary_title = f"From the final perspective (aligned with **{center_label}**):"
        else:
            ref_angle = self.angles_deg[center_idx]
            summary_title = f"From **{center_label}**'s perspective:"

        parts = []
        aligned_views = []
        rotation_offset = 90 - ref_angle

        for i, other_label in enumerate(self.labels):
            other_angle = self.angles_deg[i]
            if _angular_distance(other_angle, ref_angle) < 1.0:
                aligned_views.append(f"**{other_label}**")
                continue
            relative_angle = (other_angle + rotation_offset) % 360
            direction = _angle_to_direction(relative_angle)
            parts.append(f"**{other_label}** is at the **{direction}**")
        
        summary = summary_title
        if aligned_views:
            summary += f" This perspective is aligned with {', '.join(aligned_views)}."
        if parts and not self.target_view_label:
            summary += f" {'; '.join(parts)}."
        if self.target_view_label:
            summary += f" Focus on **{self.target_view_label}** to answer the question."
        elif not aligned_views:
            summary += " No other views to compare."
            
        return summary

    def get_computation_doc(self) -> Set[str]:
        """Returns a set of field names that are computationally expensive."""
        return set()


@serve.deployment(num_replicas=4, ray_actor_options={'num_cpus': 0.25})
class LanguageToCamera(AgentTool):
    """
    Calculates and summarizes camera spatial relationships from natural-language constraints.
    """
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 4
    AUTOSCALING_MAX_REPLICAS = 8

    def __init__(self):
        super().__init__()

    def _compute_relationship_summary(
        self, 
        angles_deg: List[float], 
        labels: List[str]
    ) -> str:
        """Compute relationship summary from all perspectives."""
        summary_lines = []
        num_cameras = len(labels)
        
        for ref_idx in range(num_cameras):
            ref_label = labels[ref_idx]
            ref_angle = angles_deg[ref_idx]
            rotation_offset = 90 - ref_angle
            
            parts = []
            for other_idx in range(num_cameras):
                if ref_idx == other_idx:
                    continue
                other_label = labels[other_idx]
                other_angle = angles_deg[other_idx]
                relative_angle = (other_angle + rotation_offset) % 360
                direction = _angle_to_direction(relative_angle)
                parts.append(f"**{other_label}** is at the **{direction}**")
            
            if parts:
                summary_lines.append(
                    f"From **{ref_label}**'s perspective: {'; '.join(parts)}."
                )
        
        return "\n".join(summary_lines)

    def _find_target_view(
        self,
        angles_deg: List[float],
        labels: List[str],
        target_angle: float,
        tolerance: float = 1.0,
    ) -> tuple:
        """Find the camera view that best matches the target angle."""
        min_dist = float('inf')
        best_match_idx = -1
        
        for i, angle in enumerate(angles_deg):
            dist = _angular_distance(angle, target_angle)
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i
        
        if min_dist < tolerance:
            return best_match_idx, labels[best_match_idx]
        return None, None

    @AgentTool.document_output_class(LanguageToCameraOutput)
    async def visualize_camera_layout(
        self,
        angles_deg: List[float],
        labels: List[str],
        center_view_index: Optional[int] = None,
        relative_rotation_deg: Optional[float] = None,
        question_view_rotation_deg: Optional[float] = None,
    ) -> AgentToolOutput:
        """
        Calculates and summarizes camera spatial relationships from all perspectives.
        This tool computes the relative direction (e.g., "Front", "Left", "Back-Right")
        of every camera from the viewpoint of every other camera.
        
        If `relative_rotation_deg` is provided, it also calculates a new, final
        viewing angle by rotating the view of the `center_view_index` camera.
        
        Args:
            angles_deg (List[float]): Camera viewing angles in degrees. 
                - **Convention**: 90°=Front, 0°=Right, 180°=Left, 270°=Back.
                - **Default Mapping**: If a problem only provides directional names (e.g., "front view", "right view"), you MUST use this convention to map them to the corresponding degree values.
            labels (List[str]): Human-readable labels for each camera view.
                - **Convention**: It is strongly recommended to use 1-based naming that matches user questions (e.g., `["Image 1", "Image 2", "Image 3"]`). Avoid using `Image 0`.
            center_view_index (Optional[int]): The index of the camera to use as the
                reference point for any relative calculations. Defaults to 0.
            relative_rotation_deg (Optional[float]): An optional rotation to apply to the `center_view_index`'s angle to define a final viewing angle.
                - **Convention**: A **right turn** corresponds to a **negative** angle (e.g., a 90-degree right turn is `-90`), and a **left turn** corresponds to a **positive** angle (e.g., a 90-degree left turn is `+90`).
                - **Example**: If `center_view_index` points to a camera facing Front (90°) and you want the perspective after a 90-degree right turn, pass `relative_rotation_deg=-90`. The tool will compute a `final_view_angle_deg` of (90 - 90) % 360 = 0°.
            question_view_rotation_deg (Optional[float]): An optional rotation to apply to the *final viewing angle* to determine the perspective required by the question (e.g., 'what is to my left?').
                - **Convention**: Follows the same convention as `relative_rotation_deg`. A **left** direction is **positive** (`+90`), a **right** direction is **negative** (`-90`), **behind** is `+180`.
                - **Example**: If the final viewpoint is facing Front (90°) and the question asks "what is to my left?", you should pass `question_view_rotation_deg=90`. The tool will compute a target angle of (90 + 90) % 360 = 180° (Left).
        """
        # Validate inputs
        if not angles_deg:
            return self.error(msg='angles_deg cannot be empty')
        if len(labels) != len(angles_deg):
            return self.error(
                msg=f'Number of labels ({len(labels)}) must match angles ({len(angles_deg)})'
            )

        diagram_center_index = center_view_index if center_view_index is not None else 0

        # Compute relationship summary
        relationship_summary = self._compute_relationship_summary(angles_deg, labels)
        
        # Compute final viewing angle from relative rotation
        final_view_angle_deg = None
        if relative_rotation_deg is not None:
            center_angle = angles_deg[diagram_center_index]
            final_view_angle_deg = (center_angle + relative_rotation_deg) % 360

        # Compute target view based on question's perspective
        target_view_index = None
        target_view_label = None
        if question_view_rotation_deg is not None:
            ref_angle = (
                final_view_angle_deg 
                if final_view_angle_deg is not None 
                else angles_deg[diagram_center_index]
            )
            target_angle = (ref_angle + question_view_rotation_deg) % 360
            target_view_index, target_view_label = self._find_target_view(
                angles_deg, labels, target_angle
            )

        output = LanguageToCameraOutput(
            angles_deg=angles_deg,
            center_view_index=diagram_center_index,
            labels=labels,
            relationship_summary=relationship_summary,
            final_view_angle_deg=final_view_angle_deg,
            relative_rotation_deg=relative_rotation_deg,
            target_view_index=target_view_index,
            target_view_label=target_view_label,
        )
        
        return self.success(result=output)
