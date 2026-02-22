#!/usr/bin/env bash
#
# Top-level launcher for evaluating a TinyLlama multi‑task mLoRA model.
#
# This script runs the evaluation Python program inside the Apptainer
# container via `run_in_container`.  It accepts two optional
# positional arguments: the directory containing your trained model
# (`LOAD_DIR`) and the output directory for evaluation results
# (`OUT_DIR`).  If omitted, reasonable defaults are provided.
# Additional flags override the default evaluation settings below.

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

# Python evaluation script
SCRIPT="run_glue_tinyllama_mtl_mlora_eval_single_gpu.py"

# Base arguments (must match training settings for LoRA hyperparameters)
ARGS=(
  --output_dir "${OUT_DIR}"
  --load_dir "${LOAD_DIR}"
  --eval_batch_size 16
  --max_length 256
  --num_workers 2

  --lora_r 32
  --lora_alpha 16
  --lora_dropout 0.05
  --num_B 3
  --temperature 0.1

  --fp16

  --save_eval_details
  #--eval_details_max_examples 200
)

# Append extra CLI args (override defaults if duplicated)
ARGS+=("${EXTRA_ARGS[@]}")

# Build and run command inside container
CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

run_in_container "${CMD_STR}"