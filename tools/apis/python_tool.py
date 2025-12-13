import asyncio
import base64
from dataclasses import dataclass
import functools
import os
import re
import sys
from typing import Any, Dict, Optional, Set, Tuple

import cloudpickle
from PIL import Image
from ray import serve
from torchvision import transforms as T

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext, AGENT_CONTEXT_REGISTRY
from tools.apis.objective import ObjectiveConstraint
from tools.apis.ref_frame import ReferenceFrameConstraint
from tools.llm_client import LLMClientFactory
from tools.utils.llm_invoke import invoke_with_retry
from workflow.prompts.coder import build_coder_prompt

__ALL__ = [
    'PythonCodeGenerator', 
    'PythonCodeInterpreter', 
    'PythonTool', 
    'PythonToolOutput',
    'PythonCodeGeneratorOutput',
]
WRAPPER_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'interpreter_wrapper.py')


@dataclass
class PythonToolOutput(AgentContext):
    """
    code_string (str): The generated Python code string.
    execution_result (Any): The value returned by the executed code.
    """
    code_string: str
    execution_result: Any
    _ctx_desc: str

    def to_message_content(self) -> str:
        return f'Execution result: {self.execution_result}\nGenerated Code: {self.code_string}'


@dataclass
class PythonCodeGeneratorOutput(AgentContext):
    content: str
    reasoning_content: Optional[str] = None


@serve.deployment
class PythonCodeGenerator(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 4
    AUTOSCALING_MAX_REPLICAS = 16

    MAX_TOKENS=32768
    TEMPERATURE = 0.0
    TOP_P = 0.95

    def __init__(self) -> None:
        super().__init__()

        self.client, self.model = LLMClientFactory().create_client(
            client_name='code_generator',
        )

    async def generate_code(self, prompt: str) -> Dict:
        try:
            messages = [{'role': 'user', 'content': prompt}]
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
                code_output = PythonCodeGeneratorOutput(
                    content=content,
                    reasoning_content=reasoning_content,
                )
                return self.success(result=code_output)
            
            thinking_part, sep, content_part = content.partition('</think>')
            if sep:
                code_output = PythonCodeGeneratorOutput(
                    content=content_part,
                    reasoning_content=thinking_part,
                )
                return self.success(result=code_output)
            
            code_output = PythonCodeGeneratorOutput(
                content=content,
                reasoning_content=None
            )
            return self.success(result=code_output)
        
        except Exception as e:
            err_msg = f'An error occurred during code generation: {str(e)}'
            return self.error(msg=err_msg)


@serve.deployment
class PythonCodeInterpreter(AgentTool):
    CPU_CONSUMED = 1.0
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 8

    TIMEOUT = 60.0

    def __init__(self):
        super().__init__()

    async def run(self, code_string: str, context_vars: Dict) -> AgentToolOutput:
        input_data = (code_string, context_vars)
        pickled_input = cloudpickle.dumps(input_data)
        encoded_input = base64.b64encode(pickled_input)

        cmd = [sys.executable, WRAPPER_SCRIPT_PATH]

        env = os.environ.copy()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        env['PYTHONPATH'] = f'{project_root}{os.pathsep}{env.get("PYTHONPATH", "")}'

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=encoded_input),
                timeout=self.TIMEOUT,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

            err_msg = f'Code execution timed out after {self.TIMEOUT} seconds.'
            return self.error(msg=err_msg)
        
        if process.returncode != 0:
            err_msg = f'Interpreter process failed with exit code {process.returncode}: {stderr.decode()}'
            return self.error(msg=err_msg)
        
        if not stdout:
            err_msg = f'Interpreter process produced no output: {stderr.decode()}'
            return self.error(msg=err_msg)

        try:
            pickled_output = base64.b64decode(stdout)
            result_dict = cloudpickle.loads(pickled_output)
            if result_dict['err']:
                return self.error(msg=result_dict['err']['msg'], src=result_dict['err']['src'])
            else:
                return self.success(result=result_dict['result'])
        except Exception as e:
            err_msg = f'Failed to deserialize result from interpreter: {str(e)}'
            return self.error(msg=err_msg)


def build_context_documentation(
    context_vars: Dict[str, Any],
    workspace: Dict[str, AgentContext],
) -> Dict[str, Tuple[str, Set[str]]]:
    docs = {}
    for name, value in context_vars.items():
        if not isinstance(value, str) or not value.strip().startswith('$'):
            continue

        clean_ref_string = value.strip()[1:]
        # Regex to find the base variable name (e.g., "detection" from "detection.boxes[0]")
        matches = re.match(r'^([a-zA-Z0-9_]+)(.*)$', clean_ref_string)
        if not matches:
            continue

        var_name, ops_chain_str = matches.groups()
        if var_name not in workspace:
            continue

        var_obj = workspace[var_name]
        var_type = type(var_obj)
        comp_doc = var_obj.get_computation_doc()

        if var_type in AGENT_CONTEXT_REGISTRY:
            class_doc = var_type.__doc__.strip()

            if var_name == clean_ref_string:
                # Case 1: Full AgentContext is passed (e.g., '$detection')
                docs[name] = (
                    f'A dataclass `{var_type.__name__}`. Attributes:\n{class_doc}',
                    comp_doc,
                )
            else:
                # Case 2: An attribute is passed (e.g., '$detection.boxes[0]')
                attr_matches = re.match(r'\.([a-zA-Z0-9_]+).*', ops_chain_str)
                if not attr_matches:
                    continue
                attr = attr_matches.group(1).strip()

                valid_attrs = [
                    k.strip() 
                    for k in list(var_type.__dict__['__dataclass_fields__'].keys()) 
                    if not k.strip().startswith('_')
                ]
                if attr in valid_attrs:
                    doc_lines = class_doc.strip().split('\n')
                    attr_lines, cur_attr = [], None

                    for line in doc_lines:
                        for other_attr in valid_attrs:
                            if line.strip().startswith(other_attr):
                                cur_attr = other_attr                        
                        if cur_attr == attr:
                            attr_lines.append(line)

                    attr_doc = '\n'.join(attr_lines)

                    docs[name] = (
                        f'This variable is an attribute `{ops_chain_str[1:]}` derived from dataclass `{var_type.__name__}`. Attribute:\n{attr_doc.strip()}',
                        comp_doc,
                    )
        
    return docs


@serve.deployment
class PythonTool(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = 4
    AUTOSCALING_MAX_REPLICAS = 16

    MAX_RETRIES = 3

    def __init__(
        self, 
        code_generator: PythonCodeGenerator, 
        interpreter: PythonCodeInterpreter,
    ) -> None:
        super().__init__()
        self.code_generator = code_generator
        self.interpreter = interpreter

    def _extract_python_code(self, string: str) -> Optional[str]:
        match_results = re.search(r'```python(.*?)```', string, re.DOTALL)
        if not match_results:
            err_msg = (
                'Not code block ```python ... ``` found in generation output. '
                'Maybe bad format or maybe think too much.'
            )
            raise RuntimeError(err_msg)
        
        code = match_results.group(1).strip()

        # check import sentences
        code_lines = code.split('\n')
        before_execute, imports_before_execute, sents_after_execute = True, [], []
        def_sign_line_id = -1
        for i, line in enumerate(code_lines):
            if not line.strip():
                continue
            
            if line.strip().startswith('def execute'):
                before_execute = False
                def_sign_line_id = i
                continue        
            
            if before_execute:
                imports_before_execute.append(line)
            else:
                sents_after_execute.append(line)
        
        if not imports_before_execute:
            return code
    
        new_code_lines = [
            code_lines[def_sign_line_id],
            *[' ' * 4 + line for line in imports_before_execute],
            *sents_after_execute
        ]
        new_code = '\n'.join(new_code_lines)
        return new_code
    
    async def _execute_code(
        self, 
        output: PythonCodeGeneratorOutput,
        context_desc: str,
        context_vars: Dict[str, AgentContext],
    ):
        code_string = self._extract_python_code(output.content)
        interpreter_output = await self.interpreter.run.remote(
            code_string=code_string, 
            context_vars=context_vars
        )
        if interpreter_output.err:
            raise RuntimeError(interpreter_output.err['msg'])
        
        return PythonToolOutput(
            code_string=code_string,
            execution_result=interpreter_output.result,
            _ctx_desc=context_desc,
        )

    @AgentTool.document_output_class(PythonToolOutput)
    async def code(
        self,
        context_vars: Dict[str, AgentContext],
        context_desc: str,
        user_request: str,
        context_vars_documentation: Dict[str, str],
        ref_frame_constraint: Optional[ReferenceFrameConstraint] = None,
        objective_constraint: Optional[ObjectiveConstraint] = None,
    ) -> AgentToolOutput:
        """ 
        Generates and executes Python code for geometric transformations and calculations.
        Args:
            context_vars (Dict[str, Any]): A dictionary of variables the generated code needs to access. Values MUST be references to workspace variables (e.g., `$detection.boxes[0]`) or primitives.
            context_desc (str): A **CRITICAL** natural language description explaining the role of each variable for the Coder. It **MUST** detail:
                1. **Reference Frame Constraint**: How to use the variables to build the world-to-reference transformation matrix.
                2. **Objective Constraint**: What the target data represents and that it must be transformed into the Reference Frame.
                **DO NOT** include procedural steps, formulas, or the plan itself. Only describe the context and the purpose of the inputs.
                **[GOOD] Example**:
                - `context_vars`: {"toaster_pose": "$toaster_pose", "reconstruction": "$reconstruction", "knife_points": "$knife_points_3d"},
                - `context_desc`:
                "**Reference Frame Constraint**: `toaster_pose` (the anchor's pose in its camera frame) and `reconstruction` (containing camera extrinsics) are required to establish the toaster-centric reference frame in the world.
                **Objective Constraint**: `knife_points` are the world coordinates of the knife that must be transformed into the toaster's frame for analysis.
        """

        # Process PIL.Image.Image to torch.Tensor
        to_tensor = T.ToTensor()
        for key, value in context_vars.items():
            if isinstance(value, Image.Image):
                context_vars[key] = to_tensor(value)

        invoker = self.code_generator.generate_code.remote

        prompter = functools.partial(
            build_coder_prompt,
            user_request=user_request,
            context_desc=context_desc,
            context_vars=context_vars,
            context_vars_documentation=context_vars_documentation,
            ref_frame_constraint=ref_frame_constraint,
            objective_constraint=objective_constraint,
        )

        parser = functools.partial(
            self._execute_code,
            context_desc=context_desc,
            context_vars=context_vars
        )

        try:
            output = await invoke_with_retry(
                invoker=invoker,
                prompter=prompter,
                parser=parser,
                max_retries=self.MAX_RETRIES,
            )
            return self.success(result=output)

        except Exception as e:
            err_msg = f'Failed to generate/execute the code. Last error: {e}'
            return self.error(msg=err_msg)
