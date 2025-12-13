# Usage of GCA

## Experiment Configuration

You can view all experiment configuration options and their meanings [here](../workflow/config.py). You can customize the agent's behavior using JSON files, Environment Variables, or CLI Arguments. 

* JSON Configuration Files

    The most common way is to modify the existing configuration files located in the `config/` directory (e.g., `config/agent_mmsi.json`) or create a new one. 
    
    You can then specify which config file to use when launching the agent:

    ```bash
    python -m entrypoints.agent --benchmark mmsi --config config/my_custom_config.json
    ```

* Environment Variables

    Every configuration option defined in `AgentConfig` can be set via environment variables. The naming convention is `AGENT_{OPTION_NAME_UPPERCASE}`. 
    
    For example, to change the `work_dir` and `concurrency`, you can set:

    ```bash
    export AGENT_WORK_DIR="./my_experiment_results"
    export AGENT_CONCURRENCY=8
    ```

* Command Line Arguments

    Frequently used parameters can be passed directly as command-line arguments when starting the agent. These are defined in the argument parser.

> Priority order: CLI Argmuments > JSON > Environment Variables.

## Setting up VLM/LLM

The agent utilizes two distinct roles for Large Language Models:

* `cot_reasoner`: Responsible for high-level analysis, planning, and tool orchestration (typically a VLM like GPT-4o or Qwen3-VL-Think).

* `code_generator`: Responsible for generating executable Python code based on the plan.

You can configure these roles using the following options in your config file or environment variables:

* `cot_reasoner_model` / `code_generator_model`

* `cot_reasoner_base_url` / `code_generator_base_url`

* `cot_reasoner_api_key` / `code_generator_api_key`

### Using Local vLLM Deployment

Apart from standard external APIs (OpenAI/Google), we provide built-in utilities to easily deploy and utilize local open-source models via vLLM with automatic discovery and load balancing.

**Step 1: Launch vLLM Server(s)**

You can use the provided helper scripts:

```bash
bash scripts/serve_qwen3_vl_235b_thinking.sh
```

or manually use the `entrypoints.launch_vllm` script:

```bash
python -m entrypoints.launch_vllm --model Qwen/Qwen3-VL-235B-A22B-Thinking --tp 8
```

This registers the model service in a local registry file `logs/serve.json`. You can launch multiple instances of the same model on different GPUs/machines to increase throughput.


**Step 2: Configure Agent to use vLLM**

Set the `base_url` to `"vllm"`. The agent will automatically discover the IP and port of your running vLLM instances and perform load balancing.

```json
{
    "cot_reasoner_model": "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "cot_reasoner_base_url": "vllm",
    "cot_reasoner_api_key": "bearer",
    ...
}
```

## Launching Experiments

To start an evaluation run with GCA, use the `entrypoints.agent` script.

**Basic Command**

```bash
python -m entrypoints.agent --benchmark [BENCHMARK]
```

**Other Supported Arguments**

* `--config`: Path to the JSON config file.

* `--question_type`: Filter specific question types to run.

* `--concurrency`: Number of parallel workers.

* `--work_dir`: Directory to save logs and results.

* `--resume`: Resume from a previous run.

### Launching CoT Baseline 

```bash
python -m entrypoints.cot_baseline --benchmark [BENCHMARK]
```
