import json
import re
from typing import List, Tuple

from PIL import Image
import torch


def qwen3_prompt(category: str, **kwargs) -> str:
    return (
        f'locate every instance that belongs to the following categories: "{category}". '
        f'For each "{category}", report bbox coordinates in JSON format like this: '
        f'{{"bbox_2d": [x1, y1, x2, y2], "label": "{category}"}}'
    )


async def qwen3_parse_detection(
    output, 
    image: Image.Image
) -> Tuple[torch.Tensor, List[str]]:
    content = output.content.strip()
    pattern = r'```json\s*([\s\S]*?)\s*```'
    json_match = re.search(pattern, content, re.DOTALL)
    if not json_match:
        return torch.empty(0, 4), []
    
    json_output = json.loads(json_match.group(1))
    
    boxes, labels = [], []
    width, height = image.size
    for bbox in json_output:
        abs_y1 = int(bbox['bbox_2d'][1] / 1000 * height)
        abs_x1 = int(bbox['bbox_2d'][0] / 1000 * width)
        abs_y2 = int(bbox['bbox_2d'][3] / 1000 * height)
        abs_x2 = int(bbox['bbox_2d'][2] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        boxes.append(torch.tensor(
            [abs_x1, abs_y1, abs_x2, abs_y2], dtype=torch.float32
        ))
        labels.append(bbox['label'])

    boxes = torch.stack(boxes) if boxes else torch.empty(0, 4)
    return boxes, labels


def glm_prompt(category: str, **kwargs) -> str:
    return (
        f'Please pinpoint the bounding box [[x1,y1,x2,y2], …] in the image as '
        f'per the given description "{category}".'
    )


async def glm_parse_detection(
    output,
    image: Image.Image
) -> Tuple[torch.Tensor, List[str]]:
    pattern = r'\<\|begin_of_box\|\>([\s\S]*?)\<\|end_of_box\|\>'
    bbox_match = re.search(pattern, output.content, re.DOTALL)
    if not bbox_match:
        return torch.empty(0, 4), []

    bbox = eval(bbox_match.group(1).strip())
    boxes = []
    width, height = image.size
    for bbox_i in bbox:
        abs_y1 = int(bbox_i[1] / 1000 * height)
        abs_x1 = int(bbox_i[0] / 1000 * width)
        abs_y2 = int(bbox_i[3] / 1000 * height)
        abs_x2 = int(bbox_i[2] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        boxes.append(torch.tensor(
            [abs_x1, abs_y1, abs_x2, abs_y2], dtype=torch.float32
        ))
    
    boxes = torch.stack(boxes) if boxes else torch.empty(0, 4)
    return boxes, None


VLM_AS_DETECTOR = {
    'qwen/qwen3-vl-235b-a22b-thinking': (qwen3_prompt, qwen3_parse_detection),
    'zai-org/glm-4.5v-fp8': (glm_prompt, glm_parse_detection),
}
