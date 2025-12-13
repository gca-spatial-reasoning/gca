from dataclasses import dataclass, is_dataclass, fields
import inspect
import torch
from typing import Any, Dict, List, Optional

from PIL import Image


AGENT_CONTEXT_REGISTRY = set()


class AgentContext:
    """
    Base class for all structured data objects that provide context for the agent's 
    reasoning and tool-use pipeline.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AGENT_CONTEXT_REGISTRY.add(cls)
    
    def to_message_content(self) -> str:
        return str(self)
    
    def get_computation_doc(self) -> Optional[Dict[str, str]]:
        return None
    
    def _get_obj_size_bytes(self, obj: Any):
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        if isinstance(obj, Image.Image):
            return obj.height * obj.width * len(obj.getbands())
        if is_dataclass(obj):
            return sum(self._get_obj_size_bytes(getattr(obj, f.name)) for f in fields(obj))
        if isinstance(obj, List):
            return sum(self._get_obj_size_bytes(obj_i) for obj_i in obj)
        if isinstance(obj, Dict):
            return sum(self._get_obj_size_bytes(value) for value in obj.values())
        return 0

    def estimate_payload_size_mb(self):
        total_bytes = self._get_obj_size_bytes(self)
        return total_bytes / (1024 ** 2)


@dataclass
class AgentToolOutput:
    result: Optional[Any] = None
    err_msg: Optional[str] = None
    err_src: Optional[str] = None

    @property
    def err(self) -> Optional[Dict[str, str]]:
        if self.err_msg:
            return {'msg': self.err_msg, 'src': self.err_src}
        return None


class AgentTool:
    """
    Base class for all agent tools to provide common functionality.
    """

    @staticmethod
    def document_output_class(output_class: AgentContext):
        """
        Concatenate the document of the output class after the document of the method.
        """
        def decorator(method):
            method._output_class = output_class
            return method
        return decorator

    @classmethod
    def get_doc(cls) -> Dict[str, str]:
        """
        Dynamically retrieves docstrings for class's public methods.

        Returns:
            A dictionary where keys are method names and values are their docstrings.
        """
        docs = {}

        # Inspect the class for functions (not bound methods)
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Exclude private methods and this method itself to avoid recursion
            if name.startswith('_') or name in ['get_doc', 'document_output_class']:
                continue

            method_doc = inspect.getdoc(method) or ''
            # concatenate the document of output class
            if hasattr(method, '_output_class'):
                output_class = method._output_class
                class_name = output_class.__name__
                class_doc = inspect.getdoc(output_class) or ''
                if class_doc:
                    class_doc = class_doc.strip()

                addition_doc = f"""

Returns:
    `{class_name}` dataclass, which contains:\n{class_doc}
"""
                method_doc = method_doc + addition_doc

            if method_doc:
                docs[name] = method_doc
        
        return docs
    
    def success(self, result: Any) -> AgentToolOutput:
        return AgentToolOutput(result=result)
    
    def error(self, msg: str, src: Optional[str] = None) -> AgentToolOutput:
        if src is None:
            try:
                stack = inspect.stack()
                caller_frame = stack[1]
                cls_name = caller_frame[0].f_locals.get('self', '__class__').__class__.__name__
                func_name = caller_frame.function
                src = f'{cls_name}.{func_name}'
            except:
                src = 'Unknown'
        
        return AgentToolOutput(err_msg=msg, err_src=src)

    
@dataclass
class InputImages(AgentContext):
    """User's input images.
    Attributes:
        images (List[Image.Image]): A list of `PIL.Image.Image` objects.
    """
    images: List[Image.Image]
    
    # not exposed to planner
    _sources: List[str]


@dataclass
class Instruction(AgentContext):
    """User's input text request.
    Attributes:
        text (str): Text of the user's request.
    """
    text: str


@dataclass
class InputBBoxes2D(AgentContext):
    """A list of 2D bounding boxes of the image mentioned in user's text request.
    Attributes:
        bbox2d (torch.Tensor): the value of `bounding box`, tensor([[x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ...]).
    """
    bbox2d: torch.Tensor