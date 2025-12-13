import asyncio
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import ray
from ray import serve
from ray.util.actor_pool import ActorPool
import torch

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.vggt_model import (
    VGGTModel, 
    VGGTModelReconstructOutput, 
    VGGTModelProjectionOutput
)

__ALL__ = ['SceneAligner', 'SceneAlignerOutput']


@dataclass
class SceneAlignerOutput(AgentContext):
    """
    align_transform (torch.Tensor): Shape `(4, 4)`. The estimated SE(3) homogeneous transformation matrix that aligns the `source_scene` to the coordinate system of the `reference_scene`.
    """
    align_transform: Optional[torch.Tensor]

    # not exposed to planner
    _fitness: float
    _inlier_rmse: float
    _alignment_method: Optional[str] = None
    _ref_static_object_points: Optional[List[torch.Tensor]] = None
    _src_static_object_points: Optional[List[torch.Tensor]] = None

    def to_message_content(self) -> str:
        if self.align_transform is None:
            return (
                'Failed: This may be due to poor 3D reconstruction of the selected object(s). You should:\n'
                '1. Consider providing other static object(s) and re-planning.\n'
                '2. Providing at least 3 distinct static objects for Constellation Alignment.'
            )

        return (
            f'Alignment method: {self._alignment_method}, Fitness={self._fitness:.2f}, Inlier '
            f'RMSE={self._inlier_rmse:.4f}.'
        )

    def get_computation_doc(self) -> Set[str]:
        return set(['align_transform', 'rotation', 'homo_coord'])


@ray.remote(num_cpus=1)
class AlignmentWorker:

    def __init__(self):
        # Loading open3d in independent Actor process
        import open3d as o3d
        self.o3d = o3d

    def _preprocess_point_cloud(self, pcd, voxel_size: float):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            self.o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        radius_feature = voxel_size * 5
        pcd_fpfh = self.o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            self.o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, pcd_fpfh
    
    @torch.no_grad()
    def align_dense_once(
        self, 
        ref_points_np: np.ndarray,
        src_points_np: np.ndarray
    ) -> SceneAlignerOutput:
        # --- 1. Data Preparation ---
        source_pcd = self.o3d.geometry.PointCloud()
        source_pcd.points = self.o3d.utility.Vector3dVector(src_points_np.astype(np.float64))
        target_pcd = self.o3d.geometry.PointCloud()
        target_pcd.points = self.o3d.utility.Vector3dVector(ref_points_np.astype(np.float64))

        # Use the average spacing of points as a heuristic for voxel size.
        # This makes the process more adaptive to different point cloud densities.
        avg_spacing = np.mean(target_pcd.compute_nearest_neighbor_distance())
        voxel_size = avg_spacing * 3.0 # A larger voxel size for downsampling

        source_down, source_fpfh = self._preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = self._preprocess_point_cloud(target_pcd, voxel_size)

        # --- 2. Global Registration with RANSAC ---
        distance_threshold = voxel_size * 1.5
        ransac_result = self.o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=source_down, 
            target=target_down, 
            source_feature=source_fpfh, 
            target_feature=target_fpfh, 
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=self.o3d.pipelines.registration.TransformationEstimationPointToPoint(True), # True for with_scaling
            ransac_n=3, 
            checkers=[
                self.o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                self.o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], 
            criteria=self.o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        # --- 3. Local Refinement with ICP ---
        # Use the RANSAC result as the initial guess for ICP
        # The refinement is done on the original, dense point clouds for high accuracy.
        icp_result = self.o3d.pipelines.registration.registration_icp(
            source=source_pcd, 
            target=target_pcd, 
            max_correspondence_distance=distance_threshold, 
            init=ransac_result.transformation,
            estimation_method=self.o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            criteria=self.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )

        return SceneAlignerOutput(
            align_transform=torch.from_numpy(icp_result.transformation.copy()).float(),
            _fitness=icp_result.fitness,
            _inlier_rmse=icp_result.inlier_rmse,
            _alignment_method='dense',
        )
    
    @torch.no_grad()
    def align_constellation_once(
        self, 
        ref_centroids: np.ndarray,
        src_centroids: np.ndarray,
    ) -> SceneAlignerOutput:
        pcd_ref = self.o3d.geometry.PointCloud()
        pcd_ref.points = self.o3d.utility.Vector3dVector(ref_centroids)
        pcd_src = self.o3d.geometry.PointCloud()
        pcd_src.points = self.o3d.utility.Vector3dVector(src_centroids)
        
        corr = self.o3d.utility.Vector2iVector(np.array([[i, i] for i in range(len(ref_centroids))]))
        
        result = self.o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd_src, pcd_ref, corr, 0.1,
            self.o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            3,
            [self.o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.1)],
            self.o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        return SceneAlignerOutput(
            align_transform=torch.from_numpy(result.transformation.copy()).float(),
            _fitness=result.fitness,
            _inlier_rmse=result.inlier_rmse,
            _alignment_method='constellation'
        )


@serve.deployment
class SceneAligner(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 0
    AUTOSCALING_MAX_REPLICAS = 1

    NUM_WORKERS = 16
    NUM_ENSEMBLE_RUNS = 16
    FITNESS_THRESHOLD = 0.5
    CLUSTER_DISTANCE_THRESHOLD = 0.1
    CENTROID_ERROR_THRESHOLD = 0.2

    def __init__(self, vggt: VGGTModel):
        super().__init__()
        self.vggt = vggt

        worker_actors = [AlignmentWorker.remote() for _ in range(self.NUM_WORKERS)]
        self.actor_pool = ActorPool(worker_actors)

        self.failure_cache = {}
    
    async def _get_point_clouds(
        self, 
        scene: VGGTModelReconstructOutput, 
        boxes: List[torch.Tensor],
        selected_index: List[int],
    ) -> List[np.ndarray]:
        project_refs = [
            self.vggt.project_box_to_3d_points.remote(
                reconstruction=scene,
                box=box,
                selected_index=index,
            )
            for box, index in zip(boxes, selected_index)
        ]
        project_results = await asyncio.gather(*project_refs)

        points = []
        for project_result in project_results:
            if project_result.get.err:
                raise RuntimeError(project_result.err['msg'])
        
            output: VGGTModelProjectionOutput = project_result.result
            points.append(output.points_3d.cpu().numpy())

        return points
    
    def _verify_dense_alignment(
        self, 
        dense_result: SceneAlignerOutput,
        ref_points_list: List[np.ndarray],
        src_points_list: List[np.ndarray],
    ) -> bool:
        ref_centroids = [np.mean(pts, axis=0) for pts in ref_points_list]
        src_centroids = [np.mean(pts, axis=0) for pts in src_points_list]
        
        ref_centroids_h = torch.from_numpy(np.array(ref_centroids)).float()
        ref_centroids_h = torch.nn.functional.pad(ref_centroids_h, (0, 1), 'constant', 1.0)
        
        # Project reference centroids to source space and check error
        transformed_ref_centroids = (dense_result.align_transform @ ref_centroids_h.T).T[:, :3]
        
        centroid_error = torch.mean(torch.norm(
            transformed_ref_centroids - torch.from_numpy(np.array(src_centroids)).float(), dim=1
        ))
        
        # Check if both fitness and centroid consistency are acceptable
        return (dense_result._fitness > self.FITNESS_THRESHOLD and 
                centroid_error < self.CENTROID_ERROR_THRESHOLD)

    def _vote(self, valid_results: List[SceneAlignerOutput]) -> SceneAlignerOutput:
        clusters: List[List[SceneAlignerOutput]] = []
        remaining_results = valid_results.copy()
        while remaining_results:
            current_result = remaining_results.pop(0)
            new_cluster = [current_result]

            still_remaining = []
            for other_result in remaining_results:
                distance = torch.linalg.norm(
                    current_result.align_transform - other_result.align_transform
                )
                if distance < self.CLUSTER_DISTANCE_THRESHOLD:
                    new_cluster.append(other_result)
                else:
                    still_remaining.append(other_result)
            remaining_results = still_remaining
            clusters.append(new_cluster)

        # Find the final cluster
        max_cluster_len = max(len(c) for c in clusters)
        largest_cluster = [c for c in clusters if len(c) == max_cluster_len]

        if len(largest_cluster) == 1:
            final_cluster = largest_cluster[0]
        else:
            final_cluster = max(
                largest_cluster, 
                key=lambda c: max(r._fitness / (r._inlier_rmse + 1e-9) for r in c)
            )

        best_result = max(
            final_cluster, 
            key=lambda r: r._fitness / (r._inlier_rmse + 1e-9)
        )
        return best_result

    async def _align_dense(
        self, 
        ref_points_np: np.ndarray,
        src_points_np: np.ndarray,
    ) -> SceneAlignerOutput:
        if src_points_np.shape[0] < 100 or ref_points_np.shape[0] < 100:
            raise ValueError('Not enough points for alignment (minimum 100 required).')
        
        # Share with ray Actors
        ref_points_ref = ray.put(ref_points_np)
        src_points_ref = ray.put(src_points_np)

        # Dispatch tasks
        dispatcher = self.actor_pool.map_unordered(
            lambda actor, _: actor.align_dense_once.remote(ref_points_ref, src_points_ref),
            [None] * self.NUM_ENSEMBLE_RUNS
        )
        results = [r for r in dispatcher]

        # Filter low-quality results
        valid_results = [r for r in results if r._fitness > self.FITNESS_THRESHOLD]
        if not valid_results:
            return max(results, key=lambda r: r._fitness, default=None)
        
        return self._vote(valid_results)
    
    async def _align_constellation(
        self, 
        ref_centroids_np: np.ndarray, 
        src_centroids_np: np.ndarray
    ) -> SceneAlignerOutput:
        self.actor_pool.submit(
            lambda actor, args: actor.align_constellation_once.remote(*args),
            (ref_centroids_np, src_centroids_np)
        )
        return self.actor_pool.get_next()

    @AgentTool.document_output_class(SceneAlignerOutput)
    async def align(
        self,
        reference_scene: VGGTModelReconstructOutput,
        source_scene: VGGTModelReconstructOutput,
        reference_static_boxes: torch.Tensor | List[torch.Tensor],
        source_static_boxes: torch.Tensor | List[torch.Tensor],
        reference_selected_index: int | List[int] = None,
        source_selected_index: int | List[int] = None,
    ) -> AgentToolOutput:
        """
        Aligns the `source_scene` to the coordinate system of the `reference_scene` using common static objects. It is the primary tool for creating a common coordinate frame between two separate 3D reconstructions (two single image reconstruction).
        Args:
            reference_scene (VGGTModelReconstructOutput): Reference scene, which comes from `VGGTModel.reconstruct`'s output.
            source_scene (VGGTModelReconstructOutput): The source scene to be aligned.
            reference_static_boxes (torch.Tensor | List[torch.Tensor]): A single bounding box (Tensor of shape [4,]) or a list of bounding boxes (a list of Tensor shaped [4,]) for common static objects in the corresponding image of `reference_scene`.
            source_static_boxes (torch.Tensor | List[torch.Tensor]): A single bounding box or a list of bounding boxes for the same static objects in the corresponding image of `source_scene`. **CRITICAL**: If providing a list, it MUST have the same length as `reference_static_boxes`, and the box at each index `i` in both lists MUST correspond to the same object.
            reference_selected_index (int | List[int]): When the `reference_scene` is generated from multiple images, this tool will select the corresponding image from the reconstructed image sequences, and then perform from that image based on the `reference_static_boxes` to obtain the 3d points for alignment. It MUST have the same length as `reference_static_boxes`, and each `reference_selected_index[i]` MUST correspond to the reconstructed image where the `reference_static_boxes[i]` is located. By default, this tool will select the first reconstructed image (i.e. image[0]) for each static box.
            source_selected_index (int | List[int]): Similar to `reference_selected_index`, this parameter is used for `source_scene` and `source_static_boxes`.
        """
        try:
            if not isinstance(reference_static_boxes, List):
                reference_static_boxes = [reference_static_boxes]
            if not isinstance(source_static_boxes, List):
                source_static_boxes = [source_static_boxes]

            if len(reference_static_boxes) <= 0 or len(source_static_boxes) <= 0:
                raise ValueError(
                    f'At least one static box must be provided for alignment, but got '
                    f'{len(reference_static_boxes)} reference static boxes and '
                    f'{len(source_static_boxes)} source static boxes.'
                )
            if len(reference_static_boxes) != len(source_static_boxes):
                raise ValueError(
                    f'The number of reference static boxes ({len(reference_static_boxes)}) must '
                    f'match the number of source static boxes ({len(source_static_boxes)}).'
                )
            
            if reference_selected_index is None:
                reference_selected_index = [0 for _ in range(len(reference_static_boxes))]
            if source_selected_index is None:
                source_selected_index = [0 for _ in range(len(source_static_boxes))]
            if not isinstance(reference_selected_index, List):
                reference_selected_index = [reference_selected_index]
            if not isinstance(source_selected_index, List):
                source_selected_index = [source_selected_index]
            if len(reference_selected_index) != len(reference_static_boxes) or \
                  len(source_selected_index) != len(source_static_boxes):
                raise ValueError(
                    f'The number of selected index must match the number of static boxes. '
                    f'len(reference_static_boxes): {len(reference_static_boxes)}, '
                    f'len(reference_selected_index): {len(reference_selected_index)}. '
                    f'len(source_static_boxes): {len(source_static_boxes)}',
                    f'len(source_selected_index): {len(source_selected_index)}',
                )
            
            num_boxes = len(reference_static_boxes)
            cache_key = (id(reference_scene), id(source_scene))
            force_constellation = self.failure_cache.get(cache_key, False) and num_boxes >= 3

            ref_points_list = await self._get_point_clouds(
                reference_scene, reference_static_boxes, reference_selected_index
            )
            src_points_list = await self._get_point_clouds(
                source_scene, source_static_boxes, source_selected_index
            )

            if not force_constellation:
                # Attempt 1: Dense Alignment
                combined_ref_points = np.concatenate(ref_points_list)
                combined_src_points = np.concatenate(src_points_list)
                dense_result = await self._align_dense(combined_ref_points, combined_src_points)

                is_verified = False
                if dense_result:
                    is_verified = self._verify_dense_alignment(
                        dense_result, ref_points_list, src_points_list
                    ) if num_boxes > 1 else dense_result._fitness > self.FITNESS_THRESHOLD

                if is_verified:
                    dense_result._ref_static_object_points = [torch.from_numpy(pts) for pts in ref_points_list]
                    dense_result._src_static_object_points = [torch.from_numpy(pts) for pts in src_points_list]
                    return self.success(result=dense_result)
            
            if num_boxes < 3:
                self.failure_cache[cache_key] = True
                output = SceneAlignerOutput(
                    align_transform=None, 
                    _fitness=0, 
                    _inlier_rmse=1.0,
                )
                return self.success(result=output)

            # Attempt 2: Constellation Alignment (Fallback)
            ref_centroids, src_centroids = [], []
            for ref_pts, src_pts in zip(ref_points_list, src_points_list):
                if ref_pts.size > 0 and src_pts.size > 0:
                    ref_centroids.append(np.mean(ref_pts, axis=0))
                    src_centroids.append(np.mean(src_pts, axis=0))
            if len(ref_centroids) != num_boxes:
                output = SceneAlignerOutput(
                    align_transform=None, 
                    _fitness=0, 
                    _inlier_rmse=1.0,
                )
                return self.success(result=output)
            
            constellation_result = await self._align_constellation(
                np.array(ref_centroids), 
                np.array(src_centroids)
            )
            constellation_result._ref_static_object_points = [torch.from_numpy(pts) for pts in ref_points_list]
            constellation_result._src_static_object_points = [torch.from_numpy(pts) for pts in src_points_list]

            if constellation_result and cache_key in self.failure_cache:
                del self.failure_cache[cache_key]

            return self.success(result=constellation_result)

        except Exception as e:
            err_msg = f'An error occured during points alignment: {str(e)}'
            return self.error(msg=err_msg)
