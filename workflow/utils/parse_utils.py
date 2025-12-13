import json
import re
from typing import Dict


def parse_json_str(content: str) -> Dict:
    content = content.strip()

    # GLM-4.5V
    if 'begin_of_box' in content: 
        pattern = r'\<\|begin_of_box\|\>([\s\S]*?)\<\|end_of_box\|\>'
        content_match = re.search(pattern, content, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
    
        else:
            content = content.strip().split('begin_of_box')[1].strip()
            if 'end_of_box' in content:
                content = content.split('end_of_box')[0].strip()

    pattern = r'```json\s*([\s\S]*?)\s*```'
    json_match = re.search(pattern, content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = content

    return json.loads(json_str), json_str
