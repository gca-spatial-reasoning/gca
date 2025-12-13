#!/bin/bash
set -ex

export AGENT_COT_REASONER_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_COT_REASONER_API_KEY='bearer'
export AGENT_COT_REASONER_BASE_URL='vllm'
export AGENT_COT_REASONER_PROXY=''
export AGENT_CODE_GENERATOR_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_CODE_GENERATOR_API_KEY='bearer'
export AGENT_CODE_GENERATOR_BASE_URL='vllm'
export AGENT_CODE_GENERATOR_PROXY=''

BENCHMARK=$1
MODEL_NAME=${AGENT_COT_REASONER_MODEL##*/}

mkdir -p logs/
START_TIME=`date +%Y%m%d-%H:%M:%S`
LOG_FILE=logs/agent_${BENCHMARK}_${MODEL_NAME}_${START_TIME}.log

python -m entrypoints.agent --benchmark $BENCHMARK --concurrency 24 --resume \
    2>&1 | tee -a $LOG_FILE > /dev/null &

sleep 1s
tail -f $LOG_FILE
