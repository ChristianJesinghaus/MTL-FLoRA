#!/usr/bin/env bash
#
# Dynamic evaluation launcher for TinyLlama multi‑task mLoRA models.
#
# This script derives the global LoRA rank, the global number of B adapters
# and the block size for softmax normalisation by reading the
# run_config.json file saved during training.  It then calls the
# evaluation Python program with the appropriate hyperparameters.
#
# Usage:
#   ./eval_tinyllama_mtl_mlora_dynamic.sh <LOAD_DIR> <OUT_DIR> [extra args]
#
# If LOAD_DIR/run_config.json is not found, the script falls back to
# reasonable defaults (16 for lora_r, 6 for global_num_B and 3 for block_size).

set -euo pipefail

# Determine the repository root (the directory containing this script).
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

# Parse positional arguments
LOAD_DIR="${1:-./outputs_tinyllama_train/checkpoints/ckpt_best}"
OUT_DIR="${2:-./outputs_tinyllama_eval}"
shift 2 || true
EXTRA_ARGS=("$@")

# Load container/env helpers
source script/common_env.sh

# Ensure output directory exists
mkdir -p "${OUT_DIR}"

# Resolve the path to the config file (one directory above checkpoint)
CONFIG_JSON="$(dirname "${LOAD_DIR}")/run_config.json"

# Default values in case config is missing
GLOBAL_LORA_R_DEFAULT=32
GLOBAL_NUM_B_DEFAULT=12
BLOCK_SIZE_DEFAULT=3

# Extract hyperparameters from the config if present
if [[ -f "${CONFIG_JSON}" ]]; then
    # Use Python to parse JSON and compute values
    read -r GLOBAL_LORA_R GLOBAL_NUM_B BLOCK_SIZE <<< $(python3 - <<'PY'
import json, os
cfg_path = os.environ['CONFIG_JSON']
with open(cfg_path, 'r') as f:
    cfg = json.load(f)
lora_r = cfg.get('lora_r', 4)
num_B = cfg.get('num_B', 1)
num_clients = cfg.get('num_clients', 1)
num_fl_rounds = cfg.get('num_fl_rounds', 1)
global_lora_r = lora_r * (num_clients ** num_fl_rounds)
global_num_B = num_B * (num_clients ** num_fl_rounds)
block_size = num_B
print(global_lora_r, global_num_B, block_size)
PY
)
else
    GLOBAL_LORA_R=${GLOBAL_LORA_R_DEFAULT}
    GLOBAL_NUM_B=${GLOBAL_NUM_B_DEFAULT}
    BLOCK_SIZE=${BLOCK_SIZE_DEFAULT}
fi

# Python evaluation script
SCRIPT="run_glue_tinyllama_mtl_mlora_eval_single_gpu.py"

# Base arguments
ARGS=(
  --output_dir "${OUT_DIR}"
  --load_dir "${LOAD_DIR}"
  --eval_batch_size 16
  --max_length 256
  --num_workers 2

  --lora_r "${GLOBAL_LORA_R}"
  --lora_alpha 16
  --lora_dropout 0.05
  # local number of B matrices per client (block size)
  --num_B "${BLOCK_SIZE}"
  # aggregated number of B matrices
  --global_num_B "${GLOBAL_NUM_B}"
  # block size for softmax normalisation
  --block_size "${BLOCK_SIZE}"

  --temperature 0.1
  --fp16

  --save_eval_details
)

# Append extra CLI args (override defaults if duplicated)
ARGS+=("${EXTRA_ARGS[@]}")

# Build and run command inside container
CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

run_in_container "${CMD_STR}"