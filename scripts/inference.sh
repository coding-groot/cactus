#!/bin/bash

PROJECT_ROOT=$(dirname $(dirname $(realpath "$0")))

############ Your Parameters ############
INPUT_FILE="$PROJECT_ROOT/resource/dataset/evaluation.json"
OUTPUT_DIR="$PROJECT_ROOT/output/cactus-chatgpt"
COUNSELOR_TYPE="cactus"
LLM_TYPE="chatgpt"
MAX_TURNS=20
#########################################

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT" || exit

# Activate the Python virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

python "$PROJECT_ROOT/src/inference.py" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_processes 50 \
    --counselor_type "$COUNSELOR_TYPE" \
    --llm_type "$LLM_TYPE" \
    --max_turns "$MAX_TURNS"
