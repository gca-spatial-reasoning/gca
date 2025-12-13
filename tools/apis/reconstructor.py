import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch
from ray import serve
from PIL import Image

from tools.apis import AgentTool, AgentToolOutput, AgentContext
from tools.apis.cot_reasoner import CoTReasoner, CoTReasonerOutput
from tools.apis.scene_aligner import SceneAligner
from tools.apis.semantic_detector import (
    SemanticDetector,
    SemanticDetectorOutput
)
from tools.apis.vggt_model import (
    VGGTModel,
    VGGTModelReconstructOutput,
    VGGTModelTensorTransformOutput,
    VGGTModelProjectionOutput,
)
from tools.utils.llm_invoke import invoke_with_retry
from tools.utils.mm_utils import visualize_detection
from workflow.prompts.reconstructor import (
    build_disambiguation_prompt,
    build_reconstructor_prompt
)
from workflow.utils.parse_utils import parse_json_str


__ALL__ = [
    'GeometricReconstructor',
    'GeometricReconstructionOutput',
    'GeometricProjectionOutput',
]


@dataclass
class GeometricReconstructionOutput(AgentContext):
    """
    Note that all `torch.Tensor` attributes have a sequence dimension `S` corresponding to the number of input images (S=1 if single image reconstruction else S>1). Attributes:
    world_points (torch.Tensor): Shape `(S, H, W, 3)`. 3D geometry output. For each pixel `(h, w)` in each image `s`, this tensor provides the corresponding `(X, Y, Z)` coordinates in a unified world coordinate system. This world system is defined by the first camera's pose (camera 0) and follows the OpenCV convention: `+X_world` points to camera 0's right, `+Y_world` points to camera 0's down, `+Z_world` points forward into the camera 0's scene.
    world_points_conf (torch.Tensor): Shape `(S, H, W)`. Confidence scores for `world_points`. Higher values indicate more reliable points.
    extrinsic (torch.Tensor): Shape `(S, 4, 4)`. The **world-to-camera** SE(3) homogeneous transformation matrix for each image. Each `extrinsic[s]` is the homogeneous form of a `[R_s | t_s]` matrix that transforms a point from the world frame to camera `s`'s frame: `P_camera = R_s @ P_world + t_s`. The world system is defined by camera 0's pose, so `extrinsic[0]` is an identity matrix.
    intrinsic (torch.Tensor): Shape `(S, 3, 3)`. Camera intrinsic matrices. Transforms a 3D point in a camera's coordinate system to the 2D image plane. The principal point `(cx, cy)` is assumed to be at the image center.
    """
    world_points: torch.Tensor
    world_points_conf: torch.Tensor
    extrinsic: torch.Tensor
    intrinsic: torch.Tensor
    
    # not exposed to planner
    _image: List[Image.Image]
    _image_tensor: torch.Tensor
    _depth: torch.Tensor
    _depth_conf: torch.Tensor
    _transform_info: List[Dict[str, Any]]

    _reconstruction_type: str
    _reconstruction_model: str
    _align_objects: List[str]
    _align_transform: torch.Tensor

    def _summarize_geometry(self) -> str:
        num_images = len(self._image)
        total_points = self.world_points.reshape(-1, 3)

        if total_points.shape[0] < 1000: # Check if there are enough points
            return (
                f'Failed: Reconstruction from {num_images} image(s) resulted in a low-quality point cloud.'
            )

        min_coords, _ = torch.min(total_points, dim=0)
        max_coords, _ = torch.max(total_points, dim=0)
        scene_dims = max_coords - min_coords

        if self._reconstruction_type == 'multiple':
            reconstruction_type = f'{self._reconstruction_type} ({self._reconstruction_model})'
        else:
            reconstruction_type = f'single ({self._reconstruction_model}) + align ({self._align_objects})'

        return (
            f'Reconstruction from {num_images} image(s) successful.\n'
            f'Total_points: {len(total_points)}\n'
            f'Scene bounding box size: (W: {scene_dims[0]:.2f}, H: {scene_dims[1]:.2f}, D: {scene_dims[2]:.2f})\n'
            f'Reconstruction type: {reconstruction_type}'
        )

    def _summarize_poses(self) -> str:
        num_cameras = self.extrinsic.shape[0]
        return f'{num_cameras} camera pose(s) are estimated.'

    def to_message_content(self) -> str:
        geo_summary = self._summarize_geometry()
        pose_summary = self._summarize_poses()

        return (
            f'3D Reconstruction Summary:\n- Geometry: {geo_summary}\n- Camera Poses: {pose_summary}'
        )

    def get_computation_doc(self) -> Set[str]:
        return set(['extrinsic', 'rotation', 'homo_coord'])
    
    def to_vggt(self) -> VGGTModelReconstructOutput:
        return VGGTModelReconstructOutput(
            world_points=self.world_points,
            world_points_conf=self.world_points_conf,
            extrinsic=self.extrinsic,
            intrinsic=self.intrinsic,
            _image=self._image,
            _image_tensor=self._image_tensor,
            _depth=self._depth,
            _depth_conf=self._depth_conf,
            _transform_info=self._transform_info,
        )

    @classmethod    
    def from_vggt(
        cls, 
        vggt: VGGTModelReconstructOutput, 
        reconstruct_type: str,
        _align_objects: Optional[List[str]] = None,
        _align_transform: Optional[torch.Tensor] = None,
    ):
        return cls(
            world_points=vggt.world_points,
            world_points_conf=vggt.world_points_conf,
            extrinsic=vggt.extrinsic,
            intrinsic=vggt.intrinsic,
            _image=vggt._image,
            _image_tensor=vggt._image_tensor,
            _depth=vggt._depth,
            _depth_conf=vggt._depth_conf,
            _transform_info=vggt._transform_info,
            _reconstruction_type=reconstruct_type,
            _reconstruction_model='vggt',
            _align_objects=_align_objects,
            _align_transform=_align_transform,
        )


@dataclass
class GeometricTensorTransformOutput(AgentContext):
    """
    transformed_tensor (torch.Tensor): The transformed tensor. Its spatial dimensions (H, W) now align with the VGGT model's preprocessed input space, making it suitable for direct use with outputs like `world_points`. The other dimensions are preserved.
    """
    transformed_tensor: torch.Tensor

    def to_vggt(self) -> VGGTModelTensorTransformOutput:
        return VGGTModelTensorTransformOutput(self.transformed_tensor)
    
    @classmethod
    def from_vggt(cls, vggt: VGGTModelTensorTransformOutput):
        return cls(transformed_tensor=vggt.transformed_tensor)


@dataclass
class GeometricProjectionOutput(AgentContext):
    """
    points_3d (torch.Tensor): Shape `(N, 3)`. A tensor of 3D world coordinates corresponding to the input 2D box.
    points_confidence (torch.Tensor): Shape `(N,)`. Confidence scores for each projected 3D point.
    """
    points_3d: torch.Tensor
    points_confidence: torch.Tensor

    # not expose to planner
    _points_rgb: torch.Tensor

    def to_message_content(self) -> str:
        num_points = self.points_3d.shape[0]

        if num_points == 0:
            return 'Failed: Projection resulted in 0 points. The input mask may be empty or not overlap with the reconstructed scene.'

        avg_conf = torch.mean(self.points_confidence).item()
        return f"Projected {num_points} points into 3D with an average confidence of {avg_conf:.2f}."

    def to_vggt(self) -> VGGTModelProjectionOutput:
        return VGGTModelProjectionOutput(
            points_3d=self.points_3d,
            points_confidence=self.points_confidence,
            _points_rgb=self._points_rgb
        )
    
    @classmethod
    def from_vggt(cls, vggt: VGGTModelProjectionOutput):
        return cls(
            points_3d=vggt.points_3d,
            points_confidence=vggt.points_confidence,
            _points_rgb=vggt._points_rgb
        )


@serve.deployment
class GeometricReconstructor(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 4

    MAX_RETRIES = 3

    def __init__(
        self,
        reasoner: CoTReasoner,
        vggt: VGGTModel,
        aligner: SceneAligner,
        detector: SemanticDetector,
    ) -> None:
        super().__init__()

        self.reasoner = reasoner
        self.vggt = vggt
        self.aligner = aligner
        self.detector = detector
    
    async def _tensor_transform(
        self,
        tensor: torch.Tensor,
        reconstruction: GeometricReconstructionOutput,
        selected_index: int = 0,
        interpolation: str = 'auto'
    ) -> AgentToolOutput:
        transform_info = reconstruction._transform_info[selected_index]
        if reconstruction._reconstruction_model == 'vggt':
            output = await self.vggt._tensor_transform.remote(
                tensor=tensor,
                transform_info=transform_info,
                interpolation=interpolation
            )
            if output.err:
                return output
            
            output = GeometricTensorTransformOutput.from_vggt(output.result)
            return self.success(output)

        else:
            raise NotImplementedError
    
    async def _parse_strategy_json(self, output: CoTReasonerOutput) -> Dict[str, str]:
        strategy_json, _ = parse_json_str(output.content)

        json_key = ['reconstruct_type', 'align_objects', 'reasoning']
        for key in json_key:
            if key not in strategy_json:
                raise ValueError(f'JSON is missing required keys: {key}')
        
        reconstruct_type = strategy_json['reconstruct_type']
        if reconstruct_type not in ['single', 'multiple']:
            raise ValueError(
                f'Invalid reconstruction_type, got "{reconstruct_type}", '
                f'expected ["single", "multiple"]'
            )
        align_objects = strategy_json['align_objects']
        if reconstruct_type == 'single':
            if align_objects is None or not isinstance(align_objects, list) \
                  or len(align_objects) < 1 or not align_objects[0]:
                raise ValueError(
                    f'MUST provide align_objects when reconstruct_type == "single", got {align_objects}'
                )
        
        return strategy_json
            
    async def _parse_disambiguation_json(self, output: CoTReasonerOutput) -> Dict[str, str]:
        disambiguation_json, _ = parse_json_str(output.content)

        json_key = ['selected_index', 'reasoning']
        for key in json_key:
            if key not in disambiguation_json:
                raise ValueError(f'JSON is missing required keys: {key}')
        
        return disambiguation_json
            
    async def _get_reconstruct_strategy(
        self, 
        image_source: List[Image.Image], 
        user_question: str,
    ) -> Dict:
        if len(image_source) == 1:
            strategy = {
                'reconstruct_type': 'single',
                'align_objects': None,
                'reasoning': 'default reconstruction strategy for single image input'
            }
            return strategy

        if len(image_source) > 2:
            strategy = {
                'reconstruct_type': 'multiple',
                'align_objects': None,
                'reasoning': 'default reconstruction strategy for >2 image inputs'
            }
            return strategy
        
        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=image_source,
            add_label=True
        )

        prompter = functools.partial(
            build_reconstructor_prompt,
            user_request=user_question
        )

        try:
            strategy = await invoke_with_retry(
                invoker=invoker,
                prompter=prompter,
                parser=self._parse_strategy_json,
                max_retries=self.MAX_RETRIES
            )

        except:
            strategy = {
                'reconstruct_type': 'multiple',
                'align_objects': None,
                'reasoning': 'default strategy for last retry'
            }
        
        return strategy
            
    async def _disambiguate_detection(
        self, 
        detection: SemanticDetectorOutput,
        user_question: str, 
        image: Image.Image,
        all_images: List[Image.Image],
    ) -> AgentToolOutput:
        viz_image = visualize_detection(
            image=image,
            boxes=detection.boxes,
            labels=detection.labels
        )

        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=all_images,
            other_images={'detection_result': viz_image},
            add_label=True
        )

        prompter = functools.partial(
            build_disambiguation_prompt,
            user_request=user_question,
            detection_prompt=detection.labels[0],
            num_boxes=len(detection.boxes),
        )

        try:
            disambiguation_json = await invoke_with_retry(
                invoker=invoker,
                prompter=prompter,
                parser=self._parse_disambiguation_json,
                max_retries=self.MAX_RETRIES
            )
            return self.success(int(disambiguation_json['selected_index']))
        
        except Exception as e:
            err_msg = f'Failed to disambiguate the "{detection.labels[0]}". Last error: {e}'
            return self.error(msg=err_msg)
    
    async def _get_align_box(
        self,
        image: Image.Image,
        prompt: str,
        user_question: str,
        all_images: List[Image.Image],
    ) -> AgentToolOutput:
        output = await self.detector.detect.remote(image_source=image, prompt=prompt)
        if output.err:
            return output
        
        detection: SemanticDetectorOutput = output.result
        boxes = detection.boxes
        if len(boxes) == 0:
            return self.error(msg=f'Could not detect "{prompt}" for alignment.')
        if len(boxes) == 1:
            return self.success(result=boxes[0])

        # disambiguate
        box_idx = await self._disambiguate_detection(detection, user_question, image, all_images)
        if box_idx.err:
            return box_idx
        box_idx = box_idx.result
        if 0 <= box_idx < len(boxes):
            return self.success(boxes[box_idx])
        
        err_msg = f'index out of range, got {box_idx}, expected 0 <= index < {len(boxes)}'
        return self.error(msg=err_msg)

    async def _single_image_reconstruction(
        self,
        image_source: List[Image.Image],
        strategy: Dict[str, str],
        user_question: str,
    ) -> AgentToolOutput:
        reconstruction_model = self.vggt
        align_objects = strategy.get("align_objects")
    
        reconstruction_tasks = [
            reconstruction_model.reconstruct.remote(image_source=img) 
            for img in image_source
        ]

        detection_tasks = []
        for image in image_source:
            for prompt in align_objects:
                detection_tasks.append(self._get_align_box(image, prompt, user_question, image_source))
         
        all_results = await asyncio.gather(*reconstruction_tasks, *detection_tasks)
        reconstruction_results = all_results[:len(image_source)]
        detection_results = all_results[len(image_source):]

        for res in reconstruction_results:
            if res.err:
                return res
        scene1, scene2 = [res.result for res in reconstruction_results]

        # match valid detection results by prompts
        detections_by_prompt = {prompt: [None, None] for prompt in align_objects}
        detection_idx = 0
        for i in range(len(image_source)):
            for prompt in align_objects:
                res = detection_results[detection_idx]
                if not res.err:
                    detections_by_prompt[prompt][i] = res.result
                detection_idx += 1

        static_boxes1, static_boxes2, prompts = [], [], []
        for prompt, boxes in detections_by_prompt.items():
            if boxes[0] is not None and boxes[1] is not None:
                static_boxes1.append(boxes[0])
                static_boxes2.append(boxes[1])
                prompts.append(prompt)
         
        if not static_boxes1:
            err_msg = f'Alignment failed: Could not find any common static objects matching {align_objects} in both images.'
            return self.error(msg=err_msg)

        for i in range(self.MAX_RETRIES + 1):
            try:
                # align to camera 0 / world frame
                align_output = await self.aligner.align.remote(
                    reference_scene=scene1,
                    source_scene=scene2,
                    reference_static_boxes=static_boxes1,
                    source_static_boxes=static_boxes2,
                    reference_selected_index=[0 for _ in range(len(static_boxes1))],
                    source_selected_index=[0 for _ in range(len(static_boxes2))],
                )
                if align_output.err:
                    raise RuntimeError(align_output.err['msg'])

                align_transform = align_output.result.align_transform
                points_s2 = scene2.world_points.squeeze(0)  # Shape: (H, W, 3)
                points_s2_h = torch.cat([
                    points_s2.view(-1, 3),
                    torch.ones(points_s2.shape[0] * points_s2.shape[1], 1)
                ], dim=-1)  # Shape: (N, 4)

                transform_s2_to_world = align_transform

                # transform scene2 into scene1 (world)
                points_s2_aligned_h = (transform_s2_to_world @ points_s2_h.T).T
                points_s2_aligned = (
                    points_s2_aligned_h[:, :3] / points_s2_aligned_h[:, 3, None]
                ).view(points_s2.shape)
            except Exception as e:
                if i < self.MAX_RETRIES:
                    await asyncio.sleep(0.5)
                else:
                    return self.error(msg=str(e))

        world_points_combined = torch.stack([scene1.world_points.squeeze(0), points_s2_aligned], dim=0)
        world_points_conf_combined = torch.cat([scene1.world_points_conf, scene2.world_points_conf], dim=0)
        extrinsic_combined = torch.stack([
            scene1.extrinsic.squeeze(0),         # Identity matrix
            torch.inverse(align_transform)       # Inverse of T_s2_to_s1
        ], dim=0)
        intrinsic_combined = torch.cat([scene1.intrinsic, scene2.intrinsic], dim=0)
        depth_combined = torch.cat([scene1._depth, scene2._depth], dim=0)
        depth_conf_combined = torch.cat([scene1._depth_conf, scene2._depth_conf], dim=0)
        transform_info_combined = scene1._transform_info + scene2._transform_info
        image_tensor_combined = torch.cat([scene1._image_tensor, scene2._image_tensor], dim=0)

        output = GeometricReconstructionOutput(
            world_points=world_points_combined,
            world_points_conf=world_points_conf_combined,
            extrinsic=extrinsic_combined,
            intrinsic=intrinsic_combined,
            _image=image_source,
            _image_tensor=image_tensor_combined,
            _depth=depth_combined,
            _depth_conf=depth_conf_combined,
            _transform_info=transform_info_combined,
            _reconstruction_type='single',
            _reconstruction_model='vggt',
            _align_objects=align_objects,
            _align_transform=align_transform,
        )
        return self.success(result=output)

    @AgentTool.document_output_class(GeometricReconstructionOutput)
    async def reconstruct(
        self,
        image_source: List[Image.Image],
        user_question: str
    ) -> AgentToolOutput:
        """
        Perform 3D geometric reconstruction on multiple images, automatically selecting the optimal strategy.
        Args:
            image_source (List[Image.Image]): Input image list.
            user_question (str): user's original request, used for contextual decision-making.
        """
        if not isinstance(image_source, List):
            image_source = [image_source]
        strategy = await self._get_reconstruct_strategy(image_source, user_question)
        reconstruct_type = strategy.get('reconstruct_type')
        reconstruction_model = self.vggt

        if reconstruct_type == 'multiple' or len(image_source) == 1:
            reconstruction_output = await reconstruction_model.reconstruct.remote(
                image_source=image_source
            )
            if reconstruction_output.err:
                return reconstruction_output

            reconstruction_output = reconstruction_output.result
            output = GeometricReconstructionOutput.from_vggt(
                reconstruction_output,
                reconstruct_type=reconstruct_type,
            )
            del reconstruction_output

        else:
            output = await self._single_image_reconstruction(
                image_source=image_source,
                strategy=strategy,
                user_question=user_question
            )
            if output.err:
                return output

            output = output.result
        return self.success(result=output)

    @AgentTool.document_output_class(GeometricProjectionOutput)
    async def project_box_to_3d_points(
        self,
        reconstruction: GeometricReconstructionOutput,
        box: torch.Tensor,
        selected_index: int = 0,
    ) -> AgentToolOutput:
        """
        Projects a 2D bounding box from a specific image into the unified 3D world coordinate system.
        Args:
            reconstruction (GeometricReconstructionOutput): The unified 3D reconstruction from the reconstruct method.
            box (torch.Tensor): Shape `(4,)`. The bounding box for the target object.
            selected_index (int): The index of the image where the box is located.
        """
        if box.squeeze().ndim != 1 or box.squeeze().shape[0] != 4:
            err_msg = f'Bad shape of input box. Expected shape `(4,)`, get {box.shape}'
            return self.error(msg=err_msg)

        box = box.squeeze()
        if reconstruction._reconstruction_model == 'vggt':
            vggt_reconstruction = reconstruction.to_vggt()
            output = await self.vggt.project_box_to_3d_points.remote(
                reconstruction=vggt_reconstruction,
                box=box,
                selected_index=selected_index
            )
            if output.err:
                return output
            output: VGGTModelProjectionOutput = output.result

            output = GeometricProjectionOutput.from_vggt(output)
            return self.success(result=output)

        else:
            raise NotImplemented
