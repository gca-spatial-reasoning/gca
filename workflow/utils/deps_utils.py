import inspect
import re
from typing import Any, Dict, get_origin, get_args, Union, Optional

from ray import serve

from tools.apis import AGENT_TOOL_REGISTRY
from tools.apis.base import AgentTool


def extract_dependency(annotation: Any):
    if isinstance(annotation, str):
        if annotation in AGENT_TOOL_REGISTRY.keys():
            return False, AGENT_TOOL_REGISTRY[annotation]
        
        matches = re.match(r'Optional\[([\w\W\d]+)\]', annotation.strip())
        if matches is None:
            return False, annotation

        dependency_class = matches.group(1)
        if dependency_class in AGENT_TOOL_REGISTRY.keys():
            return True, AGENT_TOOL_REGISTRY[dependency_class]
        return True, annotation

    origin = get_origin(annotation)
    if origin is Optional or origin is Union:
        args = get_args(annotation)
        for arg in args:
            if arg is not type(None):
                return True, arg
        return False, annotation
    else:
        return False, annotation
    

def check_optional_dependency(
    param: inspect.Parameter, 
    tool_class: AgentTool
) -> bool:
    check_func_name = f'CHECK_OPTIONAL_{param.name.upper()}'
    if hasattr(tool_class, check_func_name):
        check_func = getattr(tool_class, check_func_name)
        return check_func()
    return True


def discover_dependencies() -> Dict[str, Any]:
    dependency_graph = {}
    
    for tool_name, tool_class in AGENT_TOOL_REGISTRY.items():
        if isinstance(tool_class, serve.Deployment):
            original_class = tool_class.func_or_class
        elif isinstance(tool_class, serve.Application):
            original_class = tool_class._bound_deployment.func_or_class

        init_signature = inspect.signature(original_class.__init__)
        
        dependencies = []
        for param in init_signature.parameters.values():
            if param.name == 'self' or param.annotation is inspect.Parameter.empty:
                continue

            is_optional, dependency_class = extract_dependency(param.annotation)
            if getattr(dependency_class, 'name', None) in AGENT_TOOL_REGISTRY.keys():
                if is_optional and not check_optional_dependency(param, original_class):
                    continue
                dependencies.append(dependency_class.name)
        
        if not dependencies:
            dependency_graph[tool_name] = tool_name
        else:
            dependency_graph[tool_name] = (tool_name, *dependencies)
    
    return dependency_graph
