from argparse import Namespace
from dataclasses import asdict, dataclass, field, fields
import json
import os
from typing import Dict, List, Optional


@dataclass
class AgentConfig:
    """Defines the global configuration for an agent workflow."""
    
    # ========================================================================
    # Benchmark Configuration
    # ========================================================================
    benchmark: str = 'mmsi'
    """
    The name of the benchmark dataset to evaluate on. Options: 'mindcube', 'mmsi', 'omnispatial', 'cvbench', 'spbench', 'none'.
    If 'none', the agent runs in an eager/interactive mode without a specific dataset.
    """

    tools_to_use: List[str] = field(default_factory=list)
    """
    A list of tool names to register and make available to the Agent Planner.
    These must match the keys in `tools.apis.AGENT_TOOL_REGISTRY` (e.g., 'GeometricReconstructor', 'EasyOCR').
    Only tools listed here will be initialized and exposed to the LLM.
    """

    question_type: Optional[List[str]] | Dict[str, List[str]] = None
    """
    Filters the benchmark dataset to only evaluate specific question types.
    - For MMSI: ['Positional Relationship (Cam.–Cam.)', 'MSR', ...]
    - For CVBench: ['Count', 'Relation', ...]
    - If None, all available samples in the benchmark are used.
    """

    # ========================================================================
    # VLM / LLM Model Configuration
    # ========================================================================
    cache_dir: Optional[str] = None
    """
    Local directory path for caching downloaded models (e.g., HuggingFace cache).
    Used by tools that load local weights (like VGGT, SAM2, GroundingDINO).
    """

    cot_reasoner_model: str = ''
    """
    The model identifier for the main reasoning agent (Planner/Analyst).
    Example: 'gpt-4o', 'Qwen/Qwen2-VL-72B-Instruct'.
    """

    cot_reasoner_base_url: str = ''
    """
    The API base URL for the reasoning model.
    - Standard URL: e.g., 'https://api.openai.com/v1'
    - Special values: 'vllm' (uses internal vLLM deployment).
    """

    cot_reasoner_api_key: str = ''
    """API key for the reasoning model service."""

    code_generator_model: str = ''
    """
    The model identifier for the Python code generation tool.
    Often distinct from the reasoner to optimize for coding capability (e.g., 'deepseek-coder').
    """

    code_generator_base_url: str = ''
    """The API base URL for the code generation model."""

    code_generator_api_key: str = ''
    """API key for the code generation model service."""

    # ========================================================================
    # Prompt Engineering Configuration
    # ========================================================================
    prompt_version: str = 'v1'
    """
    Specifies the version of prompt templates to use.
    """

    use_visual_in_context_examples: bool = True
    """
    If True, injects visual in-context examples (images) into the prompt for tasks like Reference Frame Analysis. 
    This helps the VLM understand complex spatial definitions by "seeing" examples rather than just reading text descriptions.
    """

    # ========================================================================
    # Agent Capability Switches
    # ========================================================================
    use_knowledge_augmented_code_generation: bool = True
    """
    If True, injects domain-specific geometric formulas and documentation (COMPUTATION_DOCS) into the PythonTool's prompt. 
    This helps the LLM generate correct code for coordinate transformations, rotation handling, etc.
    """

    use_meta_planner: bool = False
    """
    If True, enables the 'MetaPlanner' node at the start of the workflow.
    The MetaPlanner decides whether the query is 'text-driven' (solvable by pure logic) or 'image-driven' (requires vision tools), routing the flow accordingly.
    """

    use_reasoner_for_detection: bool = False
    """
    If True, uses the main VLM (CoTReasoner) to perform open-vocabulary object detection instead of the specialized GroundingDINO model. 
    Useful if the VLM has strong grounding capabilities (e.g., Qwen3-VL).
    """

    enable_visual_feedback: bool = True
    """
    If True, intermediate visual outputs from tools (e.g., detection boxes drawn on images) are fed back into the Planner's context in subsequent turns.
    Crucial for the agent to "self-correct" (e.g., seeing that a detection is wrong).
    """

    max_visuals_to_load: int = 8
    """
    The maximum number of visual feedback images to keep in the conversation history.
    Prevents the context window from overflowing with too many intermediate debug images.
    """

    # ========================================================================
    # Network / Proxy Configuration
    # ========================================================================
    img_loader_proxy: Optional[str] = ''
    """HTTP proxy URL for downloading images from the web (used by ImageLoader)."""

    cot_reasoner_proxy: Optional[str] = ''
    """HTTP proxy URL for connecting to the reasoner model API."""

    code_generator_proxy: Optional[str] = ''
    """HTTP proxy URL for connecting to the code generator model API."""

    # ========================================================================
    # Logging & Reporting Configuration
    # ========================================================================
    generate_report_on_completion: bool = True
    """
    If True, generates a comprehensive HTML session report (`session_report.html`) at the end of execution. 
    The report includes the instruction, interactive logs, visualizations, and the final verdict.
    """

    enable_logging: bool = True
    """
    If True, writes detailed execution logs (`trace.jsonl`, `msg.jsonl`) to the session directory. 
    These logs are the data source for the HTML report.
    """

    # ========================================================================
    # Runtime Configuration
    # ========================================================================
    work_dir: Optional[str] = None
    """
    The root directory where all session outputs, logs, and visualizations will be stored.
    If None, a default directory based on the benchmark and model name is created.
    """

    concurrency: int = 1
    """
    The number of concurrent samples to evaluate in parallel.
    Controls the size of the asyncio semaphore during batch evaluation.
    """

    enable_serve_autoscaling: bool = True
    """
    If True, enables Ray Serve's autoscaling for tool deployments.
    Tools like VGGT or SAM2 will scale up/down replicas based on request load.
    """

    def _load_from_envs(self):
        for f in fields(self):
            env_var_name = f'AGENT_{f.name.upper()}'
            value_str = os.getenv(env_var_name)

            if value_str is None:
                continue

            target_type = f.type
            try:
                if target_type == bool:
                    converted_value = value_str.lower() in ('true', '1', 'yes')
                elif target_type == int:
                    converted_value = int(value_str)
                elif target_type == float:
                    converted_value = float(value_str)
                elif target_type == List[str]:
                    converted_value = [item.strip() for item in value_str.split(',')]
                else:
                    converted_value = value_str
                
                setattr(self, f.name, converted_value)
            except (ValueError, TypeError) as e:
                print(
                    f'[Warning] Could not convert env var {env_var_name}="{value_str}" '
                    f'to type {target_type}: {e}'
                )

    def _validate_config(self):
        from evals import BENCHMARK_REGISTRY
        from tools.apis import AGENT_TOOL_REGISTRY

        # validate benchmark
        if self.benchmark not in list(BENCHMARK_REGISTRY.keys()):
            raise ValueError(
                f'Benchmark "{self.benchmark}" not supported. Available benchmarks: '
                f'{list(BENCHMARK_REGISTRY.keys())}'
            )
        
        # validate tools_to_use
        for tool_name in self.tools_to_use:
            if tool_name not in AGENT_TOOL_REGISTRY:
                raise ValueError(
                    f'Tool "{tool_name}" specified in config is not a valid, registered tool. '
                    f'Available tools are: {list(AGENT_TOOL_REGISTRY.keys())}'
                )

    def __post_init__(self):
        self._load_from_envs()

    def update_from_json(self, config_path: str):
        with open(config_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self._validate_config()

    def update_from_args(self, args: Namespace):
        args_dict = vars(args)
        for key, value in args_dict.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self._validate_config()
    
    def to_json(self) -> Dict:
        return asdict(self)


global_config = None


def get_config() -> AgentConfig:
    global global_config
    if global_config is None:
        global_config = AgentConfig()

        config_file = os.getenv('AGENT_CONFIG_FILE')
        if config_file:
            global_config.update_from_json(config_file)

    return global_config
