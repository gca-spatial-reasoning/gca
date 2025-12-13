import asyncio
from dataclasses import dataclass
import os

from huggingface_hub import hf_hub_download
from PIL import Image
from ray import serve
import torch
from transformers import AutoImageProcessor, Dinov2Config

third_party_dir = os.path.join(os.path.dirname(__file__), '..', 'third_party')
from tools.utils.misc import add_sys_path
with add_sys_path(os.path.join(third_party_dir, 'Orient-Anything')):
    from inference import get_3angle_infer_aug
    from paths import DINO_LARGE
    from vision_tower import DINOv2_MLP, FLIP_DINOv2
    from utils import background_preprocess

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.io import ImageLoader
from workflow.config import get_config


__ALL__ = ['OrientationAnythingModel', 'OrientationAnythingModelOutput']


def infer_func(image, dino, val_preprocess, device):
    rm_bkg_img = background_preprocess(image, True)
    angles = get_3angle_infer_aug(image, rm_bkg_img, dino, val_preprocess, device)
    return angles


class DINOv2_MLP_Adapter(DINOv2_MLP):

    def __init__(self, **kwargs):
        _ = kwargs.pop('dino_mode')
        super().__init__(dino_mode=None, **kwargs)

        config = Dinov2Config.from_pretrained(
            DINO_LARGE,
            cache_dir=get_config().cache_dir
        )
        self.dinov2 = FLIP_DINOv2(config)


@dataclass
class OrientationAnythingModelOutput(AgentContext):
    '''
    azimuth (float): Horizontal viewing angle relative to the object's front face.
    polar (float): Vertical viewing angle relative to the object's front face.
    rotation (float): The camera's rotation around its viewing axis.
    '''
    azimuth: float
    polar: float
    rotation: float
    
    # not exposed to planner
    _confidence: float
    _cropped_image: Image.Image


@serve.deployment
class OrientationAnythingModel(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = 10.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 4

    REPO_ID = 'Viglong/Orient-Anything'
    FILENAME = 'ronormsigma1/dino_weight.pt'
    DEVICE = 'cuda'

    def __init__(self, img_loader: ImageLoader):
        super().__init__()

        self.model, self.preprocessor = self._load_model()
        self.img_loader = img_loader

    def _load_model(self):
        dino = DINOv2_MLP_Adapter(
            dino_mode='large',
            in_dim=1024,
            out_dim=360+180+360+2,
            evaluate=True,
            mask_dino=False,
            frozen_back=False
        ).eval()

        ckpt_path = hf_hub_download(
            repo_id=self.REPO_ID, 
            filename=self.FILENAME, 
            repo_type='model',
            cache_dir=get_config().cache_dir,
        )
        dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        dino = dino.to(self.DEVICE)

        val_preprocess = AutoImageProcessor.from_pretrained(
            DINO_LARGE,
            cache_dir=get_config().cache_dir,
        )
        return dino, val_preprocess
    
    def _crop_image_with_padding(
        self, 
        image: Image.Image, 
        box: torch.Tensor, 
        padding_percent: float = 0.1
    ) -> Image.Image:
        """
        Crops the image based on the box, adding optional padding.
        """
        x1, y1, x2, y2 = box.unbind()
        width = x2 - x1
        height = y2 - y1
        padding_x = int(width * padding_percent)
        padding_y = int(height * padding_percent)

        img_width, img_height = image.size
        crop_x1 = max(0, int(x1 - padding_x))
        crop_y1 = max(0, int(y1 - padding_y))
        crop_x2 = min(img_width, int(x2 + padding_x))
        crop_y2 = min(img_height, int(y2 + padding_y))

        cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        return cropped_image
    
    @AgentTool.document_output_class(OrientationAnythingModelOutput)
    async def predict_obj_orientation(
        self,
        image_source: Image.Image | str,
        box: torch.Tensor,
    ) -> AgentToolOutput:
        """
        Predicts the 3DoF semantic orientation of a single object in an image.
        Args:
            image_source: The `PIL.Image.Image` object.
            box: The bounding box of the object.
        """
        if isinstance(image_source, Image.Image):
            image = image_source
        else:
            image_result = await self.img_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result

        cropped_image = self._crop_image_with_padding(image, box, padding_percent=0.1)
        angles = await asyncio.to_thread(
            infer_func, 
            cropped_image, 
            self.model, 
            self.preprocessor, 
            self.DEVICE
        )

        output = OrientationAnythingModelOutput(
            azimuth=float(angles[0]),
            polar=float(angles[1]),
            rotation=float(angles[2]),
            _confidence=float(angles[3]),
            _cropped_image=cropped_image
        )
        return self.success(result=output)
