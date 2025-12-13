from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Tuple

from tools.apis.base import AgentContext


@dataclass
class ToolCall:
    """Represents a single, independent tool call protocal planned by the LLM.

    Attributes:
        step_id (int): The step number in the plan. Calls with the same ID
            can be executed in parallel.

        tool_name (str): The full name of the tool to be called, e.g.,
            'vggt_model.reconstruct'.

        output_variable (str): The variable name under which the tool's output
            will be saved in the workspace.

        args (Dict[str, Any]): A dictionary of arguments for the tool. Values 
            can be reference strings pointing to AgentContext stored in the 
            workspace, e.g., 'variable.attribute[0]', or primitive types.
    """
    step_id: int
    tool_name: str
    output_variable: str
    args: Dict[str, Any] = field(default_factory=dict)

    def get_call_id(self) -> str:
        return f'{self.step_id}/{self.tool_name}/{self.output_variable}'

    @staticmethod
    def parse_call_id(call_id: str) -> Tuple[str, str, str]:
        parts = call_id.split('/')
        assert len(parts) >= 3
        step_id, tool_name, output_variable = parts[0], parts[1], '/'.join(parts[2:])
        return step_id, tool_name, output_variable


class ParameterResolver:

    def resolve(
        self, 
        args: Any,
        workspace: Dict[str, AgentContext],
    ) -> Any:
        if isinstance(args, Dict):
            return {key: self.resolve(value, workspace) for key, value in args.items()}
        
        if isinstance(args, (List, Tuple)):
            return [self.resolve(item, workspace) for item in args]
        
        if isinstance(args, str):
            return self.resolve_reference_string(args, workspace)
        
        # for boolean, int, etc., return as is.
        return args

    def resolve_reference_string(
        self, 
        ref_string: str, 
        workspace: Dict[str, AgentContext]
    ) -> Any:
        """
        Parses and resolves a reference string if it starts with '$'.
        Otherwise, returns the string as is. e.g., '$variable.attribute[0, 1]'
        """
        if not ref_string.strip().startswith('$'):
            return ref_string
        clean_ref_string = ref_string.strip()[1:]

        var_value, ops_chain_str = self.parse_var_name(
            clean_ref_string, workspace
        )
        if not ops_chain_str:
            return var_value
        
        try:
            return self.parse_ops_chain(var_value, ops_chain_str)
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            err_msg = f'Could not resolve reference "{ref_string}". Error: {e}.'
            raise ValueError(err_msg)
    
    def parse_var_name(
        self, 
        ref_string: str, 
        workspace: Dict[str, AgentContext]
    ) -> Tuple[str, str]:
        # Regex match: variable name (group 1) + subsequent operation chain (group 2)
        matches = re.match(r'^([a-zA-Z0-9_]+)(.*)$', ref_string)
        if not matches:
            raise ValueError(f'Invalid reference format after "$": "{ref_string}"')

        var_name, ops_chain_str = matches.groups()
        if var_name not in workspace:
            raise ValueError(
                f'Reference error: Variable "{var_name}" not found in workspace. '
                f'Available variables: {list(workspace.keys())}'
            )

        var_value = workspace[var_name]
        return var_value, ops_chain_str
    
    def parse_ops_chain(self, var_value: AgentContext, ops_chain_str: str) -> Any:
        # Regex match: '.attribute' or '[index]'
        ops = re.findall(r'(\.[a-zA-Z0-9_]+|\[.*?\])', ops_chain_str)
        for op in ops:
            if op.startswith('.'):
                attr_name = op[1:]
                var_value = getattr(var_value, attr_name)
            elif op.startswith('['):
                index_str = op[1:-1]
                indices = self.parse_indices(index_str)
                var_value = var_value[indices]
        return var_value
        
    def parse_indices(self, index_str: str) -> Any:
        if not index_str:
            raise ValueError('Index cannot be empty.')
        
        if ',' in index_str:
            parts = [self.parse_single_index(p.strip()) for p in index_str.split(',')]
            return tuple(parts)
        else:
            return self.parse_single_index(index_str)
    
    def parse_single_index(self, part: str) -> Any:
        if ':' in part:
            # Slice operation, e.g., ':', '1:', ':5', '1:5'
            slice_parts = part.split(':')
            if len(slice_parts) > 3:
                raise ValueError(f'Invalid slice format: {part}')
            
            slice_args = [int(p) if p else None for p in slice_parts]
            return slice(*slice_args)
        else:
            # Integer index
            return int(part)
