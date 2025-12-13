import asyncio
import base64
import io
import sys
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from PIL import Image
from ray import serve
import shortuuid
import time

third_party_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'third_party')
)
from tools.utils.misc import add_sys_path
with add_sys_path(os.path.join(third_party_dir, 'RoboBrain2.0')):
    from inference import UnifiedInference


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.\n"
    "IMPORTANT: Your final answer in the <answer> tags MUST be enclosed in \\boxed{{}} format. "
    "For numerical answers, use Arabic numerals (1, 2, 3...) NOT words (one, two, three...). "
    "Examples: <answer>\\boxed{{42.3}}</answer>, <answer>\\boxed{{1}}</answer>, <answer>\\boxed{{A}}</answer>"
)

QUESTION_TYPE_TEMPLATES = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
}


def _image_to_data_uri(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


def _extract_text_and_images_from_messages(messages: List[Dict]) -> (str, List[str]):
    if not messages:
        return '', []

    last = messages[-1]
    content = last.get('content')
    text_parts: List[str] = []
    image_uris: List[str] = []

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = item.get('type')
            if typ == 'text':
                t = item.get('text')
                if isinstance(t, str) and t:
                    text_parts.append(t)
            elif typ in ('image_url', 'image'):
                if typ == 'image_url':
                    url_obj = item.get('image_url') or {}
                    url = url_obj.get('url', '')
                else:
                    url = item.get('image', '')
                if isinstance(url, str) and url:
                    image_uris.append(url)
    else:
        if content is not None:
            text_parts.append(str(content))

    prompt_text = '\n'.join([t for t in text_parts if t])
    return prompt_text, image_uris


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0.5})
class RoboBrainDeployment:

    def __init__(self, model_id: Optional[str] = None, device_map: str = "auto"):
        self._req_lock = asyncio.Lock()
        
        if model_id is None:
            model_id = os.environ.get("ROBOBRAIN_MODEL_ID", "BAAI/RoboBrain2.0-7B")
        self._model_id = model_id
        self.inference_engine = UnifiedInference(model_id=self._model_id, device_map=device_map)
        self._tempdir = tempfile.TemporaryDirectory()

    def __del__(self):
        self._tempdir.cleanup()

    def _save_pil_image_to_temp_file(self, image: Image.Image) -> str:
        filename = os.path.join(self._tempdir.name, f"{uuid.uuid4()}.png")
        image.save(filename, "PNG")
        return filename

    def _save_data_uri_to_temp_file(self, uri: str) -> Optional[str]:
        try:
            _, encoded = uri.split(',', 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(binary_data)).convert("RGB")
            return self._save_pil_image_to_temp_file(image)
        except Exception:
            return None

    def _prepare_image_paths(self, messages: List[Dict], images: Optional[List[Image.Image]]) -> List[str]:
        image_paths = []

        if images:
            for img in images:
                image_paths.append(self._save_pil_image_to_temp_file(img))

        _, image_uris = _extract_text_and_images_from_messages(messages)
        for uri in image_uris:
            if uri.startswith('data:image'):
                path = self._save_data_uri_to_temp_file(uri)
                if path:
                    image_paths.append(path)
            elif uri.startswith('file://'):
                image_paths.append(uri[len('file://'):])
            elif uri.startswith('http://') or uri.startswith('https://'):
                image_paths.append(uri)
        
        return list(set(image_paths))

    def _prepare_text(self, messages: List[Dict]) -> str:
        text, _ = _extract_text_and_images_from_messages(messages)
        return text
    
    def _format_prompt(self, prompt_text: str, question_type: str = "free-form") -> str:
        if not prompt_text:
            return ""
        if "<think>" in prompt_text.lower() or "<answer>" in prompt_text.lower():
            return prompt_text
        return QUESTION_TEMPLATE.format(Question=prompt_text)

    async def generate(
        self,
        messages: List[Dict],
        images: Optional[List[Image.Image]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        prompt_text = self._prepare_text(messages)
        question_type = kwargs.get("question_type", "free-form")
        prompt_text = self._format_prompt(prompt_text, question_type)
        image_paths = self._prepare_image_paths(messages, images)

        if not image_paths:
            raise TypeError('RoboBrainDeployment.generate() missing required images. Provide `images` parameter or messages with image_url/image.')

        do_sample = kwargs.get('do_sample', True)
        temperature = kwargs.get('temperature', 0.7)
        enable_thinking = kwargs.get('enable_thinking')

        inference_kwargs = {
            "do_sample": do_sample,
            "temperature": temperature,
        }
        if enable_thinking is not None:
            inference_kwargs["enable_thinking"] = enable_thinking
        
        def _run():
            return self.inference_engine.inference(
                text=prompt_text,
                image=image_paths,
                **inference_kwargs
            )

        async with self._req_lock:
            result = await asyncio.to_thread(_run)
        
        answer = result.get('answer', '') or ''
        thinking = (result.get('thinking') or '').strip()

        content = "" if not answer.strip() else (f"<think>{thinking}</think>{answer}" if thinking else answer)
        completion = ChatCompletion(
            id=f"chatcmpl-{shortuuid.random()}",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=content,
                        role="assistant"
                    ),
                )
            ],
            created=int(time.time()),
            model="robobrain_local",
            object="chat.completion",
        )
        return completion.model_dump()
