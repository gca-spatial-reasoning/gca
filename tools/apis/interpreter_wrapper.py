import base64
import sys
import traceback

import cloudpickle


ALLOWED_MODULES = {
    'numpy',
    'torch',
    'torchvision',
    'sympy',
    'pandas',
    'scipy',
    'sklearn',
    'einops',
    'regex',
    'matplotlib',
    'math',
    'collections',
    'random',
    're',
    'itertools',
    'functools',
    'datetime',
}

_original_import = __import__

def create_safe_importer(allowed_modules):
    """
    Create a secure, whitelist-based importer.
    """
    def safe_importer(name, globals=None, locals=None, fromlist=(), level=0):
        main_module = name.split('.')[0]
        if main_module not in allowed_modules:
            raise ImportError(
                f'Disallowed module import: "{name}". '
                f'Only modules from the following list are allowed: {list(allowed_modules)}'
            )
        return _original_import(name, globals, locals, fromlist, level)
    
    return safe_importer


SAFE_BUILTINS = {
    'print': print, 'len': len, 'sum': sum, 'min': min, 'max': max,
    'range': range, 'abs': abs, 'round': round, 'sorted': sorted,
    'all': all, 'any': any, 'bin': bin, 'chr': chr, 'divmod': divmod,
    'hasattr': hasattr, 'hash': hash, 'hex': hex, 'iter': iter,
    'next': next, 'oct': oct, 'ord': ord, 'enumerate': enumerate, 
    'isinstance': isinstance, 'issubclass': issubclass, 'pow': pow, 
    'filter': filter, 'map': map, 'reversed': reversed, 'zip': zip, 
    'list': list, 'dict': dict, 'set': set, 'str': str, 'int': int,
    'float': float, 'bool': bool, 'bytes': bytes, 'bytearray': bytearray,
    'complex': complex, 'tuple': tuple, 'slice': slice,
    'None': None, 'True': True, 'False': False,
    '__import__': create_safe_importer(ALLOWED_MODULES),
}


def run():
    encoded_input = sys.stdin.buffer.read()
    pickled_input = base64.b64decode(encoded_input)
    code_string, context_vars = cloudpickle.loads(pickled_input)

    local_scope = {}
    try:
        exec(code_string, {'__builtins__': SAFE_BUILTINS}, local_scope)
        
        if 'execute' not in local_scope:
            raise NameError('Code did not define an "execute" function.')
            
        execute_func = local_scope['execute']
        result = execute_func(**context_vars)
        output_data = {'err': None, 'result': result}

    except Exception:
        traceback_str = traceback.format_exc()
        err_msg = f'Error during execute() of LLM-generated code:\n---\n{traceback_str}\n---'
        output_data = {'err': {'msg': err_msg, 'src': 'interpreter_wrapper.execute'}}

    pickled_output = cloudpickle.dumps(output_data)
    encoded_output = base64.b64encode(pickled_output)
    sys.stdout.buffer.write(encoded_output)
    sys.stdout.flush()


if __name__ == '__main__':
    run()
