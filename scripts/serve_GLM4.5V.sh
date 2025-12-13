#!/bin/bash
set -ex

export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=zai-org/GLM-4.5V-FP8
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=64
NUM_GPUS=4

nohup python -m entrypoints.launch_vllm \
    --model $MODEL \
    --tp $NUM_GPUS \
    --max_model_len $MAX_MODEL_LEN \
    --max_num_seqs $MAX_NUM_SEQS > /dev/null 2>&1 &
