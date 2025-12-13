import asyncio
import base64
import io

import aiohttp
import aiofiles
from PIL import Image
from ray import serve
from torchvision import transforms as TF

from tools.apis import AgentTool, AgentToolOutput
from tools.utils.proxy_manager import ProxyManager

__ALL__ = ['ImageLoader', 'ImageBase64Encoder']


@serve.deployment
class ImageLoader(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 4
    AUTOSCALING_MAX_REPLICAS = 8

    def __init__(self) -> None:
        super().__init__()
        proxy_url = ProxyManager().get_proxy('img_loader')
        self.session = aiohttp.ClientSession(proxy=proxy_url)

    async def load_image(
        self, 
        image_source: str, 
        to_tensor: bool = False, 
        to_rgb: bool = True
    ) -> AgentToolOutput:
        try:
            if image_source.startswith(('http://', 'https://')):
                async with self.session.get(image_source) as response:
                    response.raise_for_status()
                    image_bytes = await response.read()
            else:
                async with aiofiles.open(image_source, 'rb') as f:
                    image_bytes = await f.read()

            img = Image.open(io.BytesIO(image_bytes))
            if to_tensor:
                img = TF.ToTensor()(img)
            if to_rgb:
                img = img.convert("RGB")
            return self.success(result=img)
    
        except Exception as e:
            err_msg = f'Fail to load image from "{image_source}": {str(e)}'
            return self.error(msg=err_msg)


@serve.deployment
class ImageBase64Encoder(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 4
    AUTOSCALING_MAX_REPLICAS = 8

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()
        self.image_loader = image_loader

    def _encode(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='png')
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{encoded_string}'

    async def encode_image(self, image_source: str | Image.Image) -> AgentToolOutput:
        if isinstance(image_source, str):
            image_result = await self.image_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result
        else:
            image = image_source

        base64_uri = await asyncio.to_thread(self._encode, image)
        return self.success(result=base64_uri)
