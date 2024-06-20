#!/bin/bash

GPU_DEVICES="0,1,2,3"
MODEL_PATH="<< Model Path>>"
PARALLEL_SIZE=4
PORT=9000

export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_SIZE \
    --seed 42 \
    --port $PORT
