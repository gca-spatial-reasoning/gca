import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from PIL import Image
from ray import serve
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.io import ImageLoader
from tools.apis.sam2_model import SAM2Model, SAM2ModelOutput
from workflow.config import get_config

__ALL__ = [
    'VGGTModel', 
    'VGGTModelReconstructOutput', 
    'VGGTModelProjectionOutput',
]


def load_and_preprocess_images(image_list: List[Image.Image], mode='pad'):
    # Check for empty list
    if len(image_list) == 0:
        raise ValueError('At least 1 image is required')

    # Validate mode
    if mode != 'crop' and mode != 'pad':
        raise ValueError('Mode must be either "crop" or "pad"')

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518
    transform_info_list = []

    # First process all images and collect their shapes
    for img in image_list:
        # If there's an alpha channel, blend onto white background:
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to 'RGB' (this step assigns white for transparent areas)
        img = img.convert('RGB')

        width, height = img.size
        # Original behavior: set width to 518px
        transform_info = {
            'original_shape': (width, height),
            'preprocessed_shape': None,
            'crop_box': None,
            'pad_box': None,
            'multi_image_padding': None,
        }
        if mode == 'crop':
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)

        else:
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14

        resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        transform_info['preprocessed_shape'] = (new_width, new_height)
        img_tensor = to_tensor(resized_img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        crop_box = None
        if mode == 'crop' and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_tensor = img_tensor[:, start_y : start_y + target_size, :]
            crop_box = (0, start_y, new_width, start_y + target_size)
            transform_info['crop_box'] = crop_box

        if mode == "pad":
            h_padding = target_size - img_tensor.shape[1]
            w_padding = target_size - img_tensor.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=0)
                pad_box = (pad_left, pad_right, pad_top, pad_bottom)
                img_tensor = torch.nn.functional.pad(
                    img_tensor, pad_box, mode="constant", value=1.0
                )
                transform_info['pad_box'] = pad_box
            

        shapes.add((img_tensor.shape[1], img_tensor.shape[2]))
        images.append(img_tensor)
        transform_info_list.append(transform_info)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f'Warning: Found images with different shapes: {shapes}')
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for i, img in enumerate(images):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                pad_info = (pad_left, pad_right, pad_top, pad_bottom)
                transform_info_list[i]['multi_image_padding'] = pad_info

                img = torch.nn.functional.pad(
                    img, pad_info, mode='constant', value=1.0
                )
            else:
                transform_info_list[i]['multi_image_padding'] = None

            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images, transform_info_list


@dataclass
class VGGTModelReconstructOutput(AgentContext):
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

        return (
            f'Reconstruction from {num_images} image(s) successful. Total_points: {len(total_points)}. '
            f'Scene bounding box size: (W: {scene_dims[0]:.2f}, H: {scene_dims[1]:.2f}, D: {scene_dims[2]:.2f}) meters. '
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


@dataclass
class VGGTModelTensorTransformOutput(AgentContext):
    """
    transformed_tensor (torch.Tensor): The transformed tensor. Its spatial dimensions (H, W) now align with the VGGT model's preprocessed input space, making it suitable for direct use with outputs like `world_points`. The other dimensions are preserved.
    """
    transformed_tensor: torch.Tensor

    def to_message_content(self) -> str:
        shape_str = 'x'.join(map(str, self.transformed_tensor.shape))
        return f'Successfully transformed tensor to shape: {shape_str}.'


@dataclass
class VGGTModelProjectionOutput(AgentContext):
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


@serve.deployment
class VGGTModel(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = 35.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 4

    MODEL_ID = 'facebook/VGGT-1B'
    DEVICE = 'cuda'
    DTYPE = torch.bfloat16

    def __init__(self, image_loader: ImageLoader, sam2: SAM2Model) -> None:
        super().__init__()
        self.model = VGGT.from_pretrained(
            self.MODEL_ID, cache_dir=get_config().cache_dir,
        ).to(self.DEVICE)
        self.model.eval()

        self.image_loader = image_loader
        self.sam2 = sam2
    
    @torch.no_grad()
    def _reconstruct(
        self, 
        images: List[Image.Image],
    ) -> VGGTModelReconstructOutput:
        image_tensor, transform_info = load_and_preprocess_images(images)
        image_tensor = image_tensor.to(self.DEVICE)

        with torch.amp.autocast(self.DEVICE, dtype=self.DTYPE):
            predictions = self.model(image_tensor)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions['pose_enc'], image_tensor.shape[-2:]
        )
        extrinsic_homo = torch.zeros((extrinsic.shape[1], 4, 4)).to(extrinsic.device)
        extrinsic_homo[:, :3, :4] = extrinsic.squeeze(0)
        extrinsic_homo[:, 3, 3] = 1.0

        return VGGTModelReconstructOutput(
            world_points=predictions['world_points'][0].cpu().detach(),
            world_points_conf=predictions['world_points_conf'][0].cpu().detach(),
            extrinsic=extrinsic_homo.cpu().detach(),
            intrinsic=intrinsic[0].cpu().detach(),
            _image=images,
            _image_tensor=predictions['images'][0].cpu().detach(),
            _depth=predictions['depth'][0].cpu().detach(),
            _depth_conf=predictions['depth_conf'][0].cpu().detach(),
            _transform_info=transform_info,
        )

    @torch.no_grad()
    async def _tensor_transform(
        self,
        tensor: torch.Tensor,
        transform_info: Dict[str, Any],
        interpolation: str = 'auto',
    ) -> AgentToolOutput:
        if interpolation == 'auto':
            interp_mode = 'nearest' if tensor.dtype == torch.bool else 'bilinear'
        else:
            interp_mode = interpolation

        if tensor.dim() == 2: # (H, W)
            tensor_4d = tensor.unsqueeze(0).unsqueeze(0).float()
        elif tensor.dim() == 3: # (H, W, C)
            tensor_4d = tensor.permute(2, 0, 1).unsqueeze(0).float()
        else:
            err_msg = f'Unsupported tensor dimension: {tensor.dim()}'
            return self.error(msg=err_msg)

        preprocessed_shape_hw = transform_info['preprocessed_shape'][::-1] # (H, W)
        resized_tensor = F.interpolate(
            tensor_4d, 
            size=preprocessed_shape_hw, 
            mode=interp_mode, 
            align_corners=(False if interp_mode != 'nearest' else None)
        )
        # print(f'resized {resized_tensor.sum()}')

        crop_box = transform_info.get('crop_box', None)
        if crop_box:
            x1, y1, x2, y2 = crop_box
            cropped_tensor = resized_tensor[..., y1:y2, :]
        else:
            cropped_tensor = resized_tensor

        pad_box = transform_info.get('pad_box', None)
        if pad_box:
            padded_tensor = torch.nn.functional.pad(
                cropped_tensor, pad_box, mode="constant", value=0
            )
        else:
            padded_tensor = cropped_tensor

        # print(f'cropped {cropped_tensor.sum()}')

        padding = transform_info.get('multi_image_padding', None)
        if padding:
            final_tensor_4d = F.pad(padded_tensor, padding, mode='constant', value=0)
        else:
            final_tensor_4d = padded_tensor

        final_tensor = final_tensor_4d.squeeze(0)
        if tensor.dim() == 2: # for 2d input, (1, H', W') -> (H', W')
            final_tensor = final_tensor.squeeze(0)
        elif tensor.dim() == 3: # for 3d input (C, H', W') -> (H', W', C)
            final_tensor = final_tensor.permute(1, 2, 0)

        if tensor.dtype == torch.bool:
            final_tensor = (final_tensor > 0.5).bool()
        else:
            final_tensor = final_tensor.to(tensor.dtype)
        output = VGGTModelTensorTransformOutput(transformed_tensor=final_tensor)
        return self.success(result=output)
    
    @torch.no_grad()
    async def _project_2d_mask_to_3d(
        self, 
        reconstruction: VGGTModelReconstructOutput, 
        mask_2d: torch.Tensor,
        selected_index: int = 0,
    ) -> AgentToolOutput:
        world_points = reconstruction.world_points[selected_index] # Shape: (H, W, 3)
        world_points_conf = reconstruction.world_points_conf[selected_index] # Shape: (H, W)
        image_tensor = reconstruction._image_tensor[selected_index].permute(1, 2, 0) # Shape: (C, H, W) -> (H, W, C)
        transform_info = reconstruction._transform_info[selected_index]

        # print(mask_2d.shape, world_points.shape[:2])
        if mask_2d.shape == world_points.shape[:2]:
            output = VGGTModelProjectionOutput(
                points_3d=world_points[mask_2d],
                points_confidence=world_points_conf[mask_2d],
                _points_rgb=image_tensor[mask_2d],
            )
            return self.success(result=output)

        transform_mask_output = await self._tensor_transform(
            tensor=mask_2d,
            transform_info=transform_info,
            interpolation='nearest'
        )
        if transform_mask_output.err:
            return transform_mask_output
        transform_mask_output = transform_mask_output.result
        
        transformed_mask = transform_mask_output.transformed_tensor.to(world_points.device)
        # print(transformed_mask)
        if transformed_mask.shape != world_points.shape[:2]:
            err_msg = (
                f'Shape mismatch after transform! Mask: {transformed_mask.shape}, '
                f'Points: {world_points.shape[:2]}'
            )
            return self.error(msg=err_msg)

        output = VGGTModelProjectionOutput(
            points_3d=world_points[transformed_mask],
            points_confidence=world_points_conf[transformed_mask],
            _points_rgb=image_tensor[transformed_mask],
        )
        return self.success(result=output)

    @AgentTool.document_output_class(VGGTModelReconstructOutput)
    async def reconstruct(
        self, 
        image_source: str | Image.Image | List[str | Image.Image],
    ) -> AgentToolOutput:
        """
        Reconstructs a 3D scene from one or more images. This function serves as the core geometric perception tool for the agent. It takes a flexible number of images depicting a scene and returns a comprehensive set of 3D data, including camera parameters and a 3D point cloud for every pixel in every image.
        Args:
            image_source (Image.Image | List[Image.Image]): A single image or a list of images (PIL object).
                - **Multi-Image Reconstruction (Recommended for Static Scenes)**: Provide a list of images (`List[Image.Image]`). It should be used when all images are **different views of the same static scene**. By using multiple perspectives, the model can generate a more robust, and coherent 3D scene.
                - **Single-Image Reconstruction (For Non-Static Scenes)**: Provide a single image (`Image.Image`). It will reconstruct a scene from a single viewpoint. Crucially, if you have images that represent non-static scenes (for example, photos of a room *before* and *after* an object has been moved), you must call this `reconstruct` function separately for each image. This will correctly generate independent reconstructions for each distinct scene.
        """
        if not isinstance(image_source, List):
            image_source = [image_source]

        images = [None] * len(image_source)
        load_tasks, indices_to_load = [], []
        for i, source in enumerate(image_source):
            if isinstance(source, str):
                indices_to_load.append(i)
                load_tasks.append(self.image_loader.load_image.remote(source))
            else:
                images[i] = source
        
        if load_tasks:
            loaded_results = await asyncio.gather(*load_tasks)
            for i, result in enumerate(loaded_results):
                if result.err:
                    return result
                images[indices_to_load[i]] = result.result

        vggt_output = await asyncio.to_thread(self._reconstruct, images)
        return self.success(result=vggt_output)

    @AgentTool.document_output_class(VGGTModelProjectionOutput)
    async def project_box_to_3d_points(
        self, 
        reconstruction: VGGTModelReconstructOutput,
        box: torch.Tensor,
        selected_index: int = 0,
    ) -> AgentToolOutput:
        """
        Projects a 2D bounding box to a 3D point cloud in world frame (defined by camera 0).
        Args:
            reconstruction (VGGTModelReconstructOutput): 3D reconstruction from `VGGTModel.reconstruct`.
            box (torch.Tensor): The bounding box for the target object.
            selected_index (int): Defaults to 0. When the `reconstruction` is generated from multiple images, the `box` MUST correspond to the image indexed by `selected_index`.
        """
        # 1. Call SAM2Model.segment to get the pixel mask
        sam2_output = await self.sam2.segment.remote(
            image_source=reconstruction._image[selected_index],
            box=box
        )
        if sam2_output.err:
            return sam2_output
         
        sam2_output: SAM2ModelOutput = sam2_output.result
        mask_2d = sam2_output.mask

        # 2. Project the mask to 3d points
        projection_output = await self._project_2d_mask_to_3d(
            reconstruction=reconstruction,
            mask_2d=mask_2d,
            selected_index=selected_index,
        )
        if projection_output.err:
            return projection_output
         
        return projection_output
