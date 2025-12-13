import asyncio
from dataclasses import dataclass
from typing import Dict, Set, Tuple

import numpy as np
from ray import serve
from scipy.spatial.transform import Rotation as R
import torch

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.oriany_model import (
    OrientationAnythingModel, 
    OrientationAnythingModelOutput
)
from tools.apis.reconstructor import (
    GeometricReconstructor, 
    GeometricReconstructionOutput, 
    GeometricProjectionOutput
)


@dataclass
class ObjPoseEstimatorOutput(AgentContext):
    """
    T_obj2world (torch.Tensor): Shape `(4, 4)`. **Object-to-world** SE(3) homogeneous transformation matrix. It transforms a point from the object's local coordinate system to the unified world coordinate system. The object's local frame is defined with its origin at the centroid, `+Z_[obj_name]` pointing towards the semantic "front", `+Y_[obj_name]` pointing towards the semantic "bottom", and `+X_[obj_name]` derived from the right-hand rule.
    """
    T_obj2world: torch.Tensor

    _obj_obb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    _obj_label: str
    _obj_axes: Dict[str, torch.Tensor]

    def to_message_content(self) -> str:
        rotation_with_scale = self.T_obj2world[:3, :3]

        scale = torch.norm(rotation_with_scale[:, 0]).item()
        if not (0.95 < scale < 1.05):
             # This might indicate an issue if scale is not expected
            scale_warning = f' (Warning: Non-unit scale of {scale:.3f} detected)'
        else:
            scale_warning = ''

        return f"Successfully estimated semantic 6D pose.{scale_warning}"

    def get_computation_doc(self) -> Set[str]:
        return set(['obj_coord_sys', 'obj_pose', 'rotation', 'homo_coord'])
    

@serve.deployment
class ObjPoseEstimator(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 4

    def __init__(
        self, 
        reconstructor: GeometricReconstructor, 
        ori_any: OrientationAnythingModel
    ):
        super().__init__()
        self.reconstructor = reconstructor
        self.ori_any = ori_any

        import open3d as o3d
        self.o3d = o3d

    async def _get_obj_centroid(
        self,
        points_3d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points_3d)
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        centroid = pcd_clean.get_center().copy()
        return centroid
    
    async def _estimate_obj_obb_in_world(
        self,
        points_3d: np.ndarray,
        R_obj2world: torch.Tensor,
        centroid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        R_world2obj = R_obj2world.T.numpy()
        local_points_3d = (R_world2obj @ (points_3d - centroid).T).T

        local_pcd = self.o3d.geometry.PointCloud()
        local_pcd.points = self.o3d.utility.Vector3dVector(local_points_3d)
        local_aabb = local_pcd.get_axis_aligned_bounding_box()

        local_center = local_aabb.get_center().copy()
        extent = local_aabb.get_extent().copy()
        world_center = centroid + (R_obj2world.numpy() @ local_center)

        half_extent = extent / 2.0
        corners_local = np.array([
            [local_center[0]-half_extent[0], local_center[1]-half_extent[1], local_center[2]-half_extent[2]],
            [local_center[0]+half_extent[0], local_center[1]-half_extent[1], local_center[2]-half_extent[2]],
            [local_center[0]-half_extent[0], local_center[1]+half_extent[1], local_center[2]-half_extent[2]],
            [local_center[0]-half_extent[0], local_center[1]-half_extent[1], local_center[2]+half_extent[2]],
            [local_center[0]+half_extent[0], local_center[1]+half_extent[1], local_center[2]-half_extent[2]],
            [local_center[0]+half_extent[0], local_center[1]-half_extent[1], local_center[2]+half_extent[2]],
            [local_center[0]-half_extent[0], local_center[1]+half_extent[1], local_center[2]+half_extent[2]],
            [local_center[0]+half_extent[0], local_center[1]+half_extent[1], local_center[2]+half_extent[2]],
        ])
        corners_world = (R_obj2world.numpy() @ corners_local.T).T + centroid

        return (world_center, extent, corners_world)
    
    async def _get_orientation_in_world(
        self,
        orientation: OrientationAnythingModelOutput,
        extrinsic: torch.Tensor,
    ) -> torch.Tensor:
        azimuth_deg = orientation.azimuth
        polar_deg = orientation.polar
        rotation_deg = orientation.rotation

        # T_obj2world = T_cam2world @ T_oa2cam @ T_fix
        # 1. R_cam2world (camera -> world)
        R_cam2world = extrinsic[:3, :3].T

        # 2. R_oa2cam (orientation anything to camera)
        rot = R.from_euler('XYZ', [azimuth_deg, polar_deg, rotation_deg], degrees=True)
        R_oa2cam = torch.tensor(rot.as_matrix(), dtype=torch.float32)

        # 3. R_fix
        R_fix = torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=torch.float32)
        
        R_obj2world = R_cam2world @ R_oa2cam @ R_fix
        return R_obj2world

    @AgentTool.document_output_class(ObjPoseEstimatorOutput)
    async def predict_obj_pose(
        self, 
        reconstruction: GeometricReconstructionOutput,
        box: torch.Tensor,
        selected_index: int,
        obj_label: str = None,
        **kwargs,
    ) -> AgentToolOutput:
        """
        Predicts object-to-camera transformation matrix of an object relative to the corresponding camera. It is essential for establishing the pose of an object anchor, which is then used to define a calculation frame.
        Args:
            reconstruction (GeometricReconstructionOutput): 3D reconstruction.
            box (torch.Tensor): Box for target object.
            selected_index (int): The index of the image in the reconstruction to use. Defaults to 0.
        """
        projection_ref = self.reconstructor.project_box_to_3d_points.remote(
            reconstruction=reconstruction,
            box=box,
            selected_index=selected_index
        )
        orientation_ref = self.ori_any.predict_obj_orientation.remote(
            image_source=reconstruction._image[selected_index],
            box=box
        )
        projection_output, orientation_output = await asyncio.gather(
            projection_ref, orientation_ref
        )
        if projection_output.err:
            return projection_output
        if orientation_output.err:
            return orientation_output
         
        projection_output: GeometricProjectionOutput = projection_output.result
        if projection_output.points_3d.shape[0] == 0:
            return self.error(msg='Failed to project 3D points for object')
         
        # 1. Get object's centroid
        obj_centroid = await self._get_obj_centroid(
            projection_output.points_3d.cpu().numpy()
        )
        # 2. Get object's orientation in world frame
        orientation_output: OrientationAnythingModelOutput = orientation_output.result
        R_obj_to_world = await self._get_orientation_in_world(
            orientation_output,
            reconstruction.extrinsic[selected_index].cpu(),
        )
        # 3. Get object's pose
        T_obj2world = torch.eye(4, dtype=torch.float32)
        T_obj2world[:3, :3] = R_obj_to_world
        T_obj2world[:3, 3] = torch.tensor(obj_centroid, dtype=torch.float32)
        # 4. Prepare other output for visualization
        obj_axes = {
            '+X': R_obj_to_world[:, 0] / torch.linalg.norm(R_obj_to_world[:, 0]),
            '+Y': R_obj_to_world[:, 1] / torch.linalg.norm(R_obj_to_world[:, 1]),
            '+Z': R_obj_to_world[:, 2] / torch.linalg.norm(R_obj_to_world[:, 2]),
        }
        obj_obb = tuple(
            torch.from_numpy(out)
            for out in await self._estimate_obj_obb_in_world(
                projection_output.points_3d.cpu().numpy(),
                R_obj_to_world,
                obj_centroid
            )
        )
        output = ObjPoseEstimatorOutput(
            T_obj2world=T_obj2world.cpu(),
            _obj_obb=obj_obb,
            _obj_label=obj_label or 'object',
            _obj_axes=obj_axes
        )
        return self.success(result=output)
