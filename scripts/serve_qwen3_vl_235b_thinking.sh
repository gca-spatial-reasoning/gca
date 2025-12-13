#!/bin/bash
set -ex

export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=Qwen/Qwen3-VL-235B-A22B-Thinking
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=64
NUM_GPUS=4

nohup python -m entrypoints.launch_vllm \
    --model $MODEL \
    --served_model_name $SERVED_NAME \
    --tp $NUM_GPUS \
    --max_model_len $MAX_MODEL_LEN \
    --max_num_seqs $MAX_NUM_SEQS > /dev/null 2>&1 &
