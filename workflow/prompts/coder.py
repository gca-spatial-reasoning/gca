from typing import Any, Dict, Optional, Set, Tuple

from tools.apis.ref_frame import ReferenceFrameConstraint
from tools.apis.objective import ObjectiveConstraint
from workflow.config import get_config
from workflow.prompts.computation_docs import COMPUTATION_DOCS_REGISTRY


CODER_PROMPT_V1 = '''
You are an expert Python programmer. Your goal is to write a single Python function that correctly implements the computational objective based on the provided context and documentation.

**[User's Overall Question]**
This provides the high-level context for your task. You **MUST** write code to help answer this question.
{user_request}

**[Reference Frame]**
All geometric data are defined in the world frame (defined by camera 0) unless specified. All final interpretations **MUST** be expressed in the reference frame. The reference frame is defined:
{ref_frame_desc}

**[Objective]**
The ultimate goal from the high-level question. You MUST write code to calculate this objective to answer the question.
{objective_desc}

**[Available Variable]**
This description explain the origin/purpose of the available variable. 
{context_desc}

Documentation of Available Variables
{var_docs}

**[System Knowledge]**
This information is always true for the environment your code runs in.
{system_knowledge}
**[Available Libraries]**
You can use `numpy`, `torch`, `scipy`, `math`, and other standard Python libraries. All imports MUST be inside the `execute` function.
{feedback_prompt}
**[Critical Rules and Output Format]**
1. **Synthesize and Self-Correct**: Your primary duty is to write correct code. Use the objective as your goal, but critically verify and implement the logic using the provided documentation.
2. **Handle Multiple-Choice Questions**: If the user's question is multiple-choice, your code **MUST** systematically evaluate the conditions for every option (e.g., A, B, C, D). Besides, the logic of your code should focus ONLY on the given options. For instance, if the options are about X-axis and Y-axis rotation, you MUST determine the dominant direction from `rx` and `ry` only, even if another component is globally larger.
3. **Robust Answer Selection**: If this tool is used to determine or select the **final answer** among multiple candidates, you **MUST** design a logically sound and **robust decision algorithm**.If multiple options are available, you should systematically compare them and **choose the one that is closest to the expected or optimal value**.In some cases, there may be **no perfectly correct answer**. You must carefully consider the given problem and prior analytical context to produce the **best-effort optimal answer** under the available information.
4. **DO NOT** add any explanation in your final output. Your output **MUST** follow this format:
```python
def execute({func_signature}):
    # import sentences in execute 
    import ...

    # Your code here
    ...

    return serializable_value  # return value MUST be a serializable type
```
'''.strip()


def build_coder_prompt(
    user_request: str,
    context_desc: str, 
    context_vars: Dict[str, Any],
    context_vars_documentation: Dict[str, Tuple[str, Set[str]]],
    ref_frame_constraint: Optional[ReferenceFrameConstraint] = None,
    objective_constraint: Optional[ObjectiveConstraint] = None,
    err_msg: str = None,
    response: str = None,
) -> str:
    var_docs, comp_doc_keys = [], set()
    for name, var in context_vars.items():
        var_type = type(var)
        if name in context_vars_documentation:
            var_doc = context_vars_documentation[name][0]
            formatted_doc = '\n'.join(f'  # {line.strip()}' for line in var_doc.strip().split('\n'))
            var_doc = f'- `{name}` ({var_type.__name__})\n{formatted_doc}'

            class_comp_doc_keys = context_vars_documentation[name][1]
            if class_comp_doc_keys is not None:
                for key in class_comp_doc_keys:
                    comp_doc_keys.add(key)
        else:
            var_doc = f'- `{name}` ({var_type.__name__})'
        var_docs.append(var_doc)

    var_docs = '\n'.join(var_docs)
    func_signature = ', '.join(f'{name}' for name in context_vars.keys())

    if ref_frame_constraint is None:
        ref_frame_desc = '- Identical to world frame (defined by camera 0).'
    else:
        if ref_frame_constraint._type == 'object_axes':
            primitive = 'object\'s axes'
        elif ref_frame_constraint._type == 'camera_axes':
            primitive = 'camera\'s axes'
        else:
            primitive = 'direction vector between objects'
        ref_frame_desc = f'''
- Reference frame is defined by a(n) {primitive}: {ref_frame_constraint._anchor}.
- Reference frame's mathematic formalization: {ref_frame_constraint.formalization}.
'''.strip()

        ref_frame_comp_doc_keys = ref_frame_constraint.get_computation_doc()
        if ref_frame_comp_doc_keys is not None:
            for key in ref_frame_comp_doc_keys:
                comp_doc_keys.add(key)
    
        for cardinal_dir in ['north', 'south', 'east', 'west']:
            if cardinal_dir in ref_frame_constraint.formalization.lower():
                comp_doc_keys.add('cardinal_dir')
                break

    if objective_constraint is None:
        objective_desc = 'See question.'
    else:
        objective_desc = objective_constraint.objective

    system_knowledge = '''- **Coordinate System**:
    - **Default Coordinate System**: All geometric data in input variable is in the **OpenCV coordinate system** (+X right, +Y down, +Z forward) and follow right-handed rule.
'''
    if 'obj_coord_sys' in comp_doc_keys:
        system_knowledge += f'    - {COMPUTATION_DOCS_REGISTRY["obj_coord_sys"]}\n'
        comp_doc_keys.remove('obj_coord_sys')

    if get_config().use_knowledge_augmented_code_generation:
        cats = ['basic_transform', 'ref_frame_transform', 'interpretation', 'other']
        for cat in cats:
            for key in comp_doc_keys:
                if key in COMPUTATION_DOCS_REGISTRY[cat]:
                    system_knowledge += f'- {COMPUTATION_DOCS_REGISTRY[cat][key]}\n'

    if err_msg is None:
        feedback_prompt = ""
    else:
        feedback_prompt = f'''
**[Feedback on Last Execution]**
Please revise the code based on the following error message and source.
last_response: {response}\n
error: {err_msg}\n
'''

    return CODER_PROMPT_V1.format(
        user_request=user_request,
        ref_frame_desc=ref_frame_desc,
        objective_desc=objective_desc,
        context_desc=context_desc,
        var_docs=var_docs,
        system_knowledge=system_knowledge,
        feedback_prompt=feedback_prompt,
        func_signature=func_signature,
    )
