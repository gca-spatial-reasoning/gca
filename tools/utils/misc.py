import contextlib
from functools import partial
import importlib
import os
import sys
from typing import Any, Dict, List


@contextlib.contextmanager
def add_sys_path(path: str | List[str]):
    """A context manager to temporarily add path(s) to the beginning of sys.path.

    This ensures that only paths not already in sys.path are added, and only
    those added paths are removed upon exit, making it safe and idempotent.

    Args:
        path (str | List[str]): A single path or a list of paths to add.
    """
    if isinstance(path, str):
        path = [path]
    
    paths_to_add = [p for p in path if p not in sys.path and os.path.exists(p)]

    for p in paths_to_add[::-1]:
        sys.path.insert(0, p)

    try:
        yield
    finally:
        for p in paths_to_add:
            if p in sys.path:
                sys.path.remove(p)


@contextlib.contextmanager
def mock_with_mappings(mappings: Dict[str, Any]):
    """A general-purpose context manager to temporarily mock/patch attributes in modules.

    Args:
        mappings (Dict[str, Any]): A dictionary where keys are the full
            string path to the attribute (e.g., 'sys.argv', 'torch.hub.load')
            and values are the new functions or values to replace them with.
    """
    original_values = {}
    for key, value in mappings.items():
        try:
            module_path, attr_name = key.rsplit('.', 1)
            module = importlib.import_module(module_path)
            original_values[key] = getattr(module, attr_name)
            setattr(module, attr_name, value)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f'Failed to mock "{key}": {e}')
    
    try:
        yield
    finally:
        for key, value in original_values.items():
            try:
                module_path, attr_name = key.rsplit('.', 1)
                module = importlib.import_module(module_path)
                setattr(module, attr_name, value)
            except (ImportError, AttributeError):
                pass


# A specialized context manager to temporarily ignore command-line arguments,
# particularly useful for preventing conflicts with Ray Serve's internal args.
ignore_ray_argv = partial(mock_with_mappings, {'sys.argv': [sys.argv[0]]})
