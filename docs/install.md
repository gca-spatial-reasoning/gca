# Installation of GCA

## Prerequisites

We recommend creating a Conda environment and installing PyTorch within it.

```bash
conda create -n gca python=3.11
conda activate gca
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

## Prepare Third Party Repositories

We employ several Visual Foundation Models (VFMs) for agent to parameterize the visual world. 

```bash
mkdir -p visual-agent/tools/third_party
cd visual-agent/tools/third_party
```

1. VGGT

    ```bash
    git clone --depth=1 https://github.com/facebookresearch/vggt.git && cd vggt
    pip install .
    cd ..
    ```

2. SAM2

    ```bash
    git clone --depth=1 https://github.com/facebookresearch/sam2.git && cd sam2
    pip install .

    # Download checkpoints
    # Note that we only use `sam2.1_hiera_large.pt`, you can comment out the download commands for other checkpoints.
    cd checkpoints
    sh download_ckpts.sh 

    cd ../..
    ```

3. OrientAnything

    ```bash
    git clone --depth=1 https://github.com/SpatialVision/Orient-Anything.git
    ```

4. MoGE

    ```bash
    git clone --depth=1 https://github.com/microsoft/MoGe.git
    ```

## Install Requirements

```bash
cd visual-agent/
pip install -r requirements/gca.txt
```

## Install Requirements for vLLM (Optional)

In addition to using external APIs (GPT/Gemini), GCA also supports deploying local models such as Qwen3-Thinking or GLM4.5V using vLLM for agentic reasoning.

To this end, we write some utilities to facilitate the vLLM deployment of local models and invocation of these models in GCA. If you wish to deploy local models based on these vLLM utilities, you need to install the following packages.

```bash
cd visual-agent

conda create -n gca-vllm python=3.11
conda activate gca-vllm
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install requirements/vllm.txt
```
