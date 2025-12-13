import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image
from ray import serve

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.io import ImageBase64Encoder
from tools.llm_client import LLMClientFactory
from tools.utils.mm_utils import add_label_to_image

__ALL__ = ['CoTReasoner', 'CoTReasonerOutput']


@dataclass
class CoTReasonerOutput(AgentContext):
    """
    content (str): The final, user-facing content or summary.
    reasoning_content (Optional[str]): The intermediate "thinking" steps of the model, often containing the plan, analysis, or breakdown of the problem. It can be `None` if the model does not produce a distinct thinking block.
    """
    content: str
    reasoning_content: Optional[str] = None


@serve.deployment
class CoTReasoner(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = None
    AUTOSCALING_MAX_REPLICAS = None

    MAX_TOKENS=32768
    TEMPERATURE=0.6
    TOP_P=0.95

    def __init__(self, image_encoder: ImageBase64Encoder) -> None:
        super().__init__()

        self.client, self.model = LLMClientFactory().create_client(
            client_name='cot_reasoner',
        )
        self.image_encoder = image_encoder

    @AgentTool.document_output_class(CoTReasonerOutput)
    async def cot_reason(
        self, 
        prompt: str, 
        input_images: Optional[Image.Image | List[Image.Image]] = None,
        other_images: Optional[Dict[str, Image.Image]] = None,
        add_label: bool = True,
    ) -> AgentToolOutput:
        """
        Acts as the agent's central reasoning and planning engine.

        This tool takes a user query or a complex task description and uses a large language model to perform Chain-of-Thought (CoT) reasoning. It can be used to break down problems, formulate plans for other tools, or generate final answers based on provided context.

        Args:
            prompt (str): The text prompt that requires reasoning. This can be a direct question, a task to be planned, or a request for code generation.
            image_sources (Optional[Image.Image | List[Image.Image]]): An optional single image or list of images to provide visual context for multimodal reasoning.
        """
        images_to_encode = []
        if input_images:
            if not isinstance(input_images, List):
                input_images = [input_images]
            for i, input_image in enumerate(input_images):
                images_to_encode.append(
                    add_label_to_image(input_image, f'input_images.images[{i}]')
                    if add_label else input_image
                )
        if other_images:
            for label, other_image in other_images.items():
                images_to_encode.append(
                    add_label_to_image(other_image, label) 
                    if add_label else other_image
                )

        base64_uris = []
        if images_to_encode:
            encode_result_refs = [
                self.image_encoder.encode_image.remote(image)
                for image in images_to_encode
            ]

            encode_results = await asyncio.gather(*encode_result_refs)
            for result in encode_results:
                if result.err:
                    return result
                base64_uris.append(result.result)

        content = [{'type': 'text', 'text': prompt}]
        for uri in base64_uris:
            content.append({
                'type': 'image_url', 
                'image_url': {'url': uri}
            })

        try:
            messages = [{'role': 'user', 'content': content}]
            if 'gpt-4' in self.model.lower():
                max_token_num = 16384
            else:
                max_token_num = self.MAX_TOKENS
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_token_num,
                temperature=self.TEMPERATURE,
                top_p=self.TOP_P
            )

            output_message = response.choices[0].message
            content = output_message.content
            if hasattr(output_message, 'reasoning_content'):
                reasoning_content = output_message.reasoning_content
            else:
                reasoning_content = None

            if reasoning_content is not None:
                reasoning_content = reasoning_content.replace('<think>', '').replace('</think', '')
                cot_output = CoTReasonerOutput(
                    content=content,
                    reasoning_content=reasoning_content
                )
                return self.success(result=cot_output)

            thinking_part, sep, content_part = content.partition('</think>')
            if sep:
                reasoning_content = thinking_part.replace('<think>', '')
                cot_output = CoTReasonerOutput(
                    content=content_part,
                    reasoning_content=reasoning_content
                )
                return self.success(result=cot_output)

            cot_output = CoTReasonerOutput(
                content=content,
                reasoning_content=None
            )
            return self.success(result=cot_output)
         
        except Exception as e:
            err_msg = f'An error occurred: {str(e)}'
            return self.error(msg=err_msg)
