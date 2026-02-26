#!/usr/bin/env bash
#
# Top-level launcher for evaluating a TinyLlama multi-task mLoRA model.
#
# This script mirrors the training launcher workflow:
# - sources `script/common_env.sh` (defines run_in_container + HF cache env)
# - runs the Python eval program inside the Apptainer container
#
# IMPORTANT:
# - Pass the *training output directory* as LOAD_DIR (e.g. outputs/tinyllama_train_...).
#   The Python eval script will automatically load the FINAL GLOBAL model:
#     - checkpoints/ckpt_global_final.pt (preferred) OR
#     - adapter_state_final.pt + heads_state_final.pt
# - Hyperparameters are auto-resolved from <LOAD_DIR>/run_config.json when present.
#
# Usage:
#   bash eval_tinyllama_mtl_mlora.sh <LOAD_DIR> <OUT_DIR> [extra python args...]
#
# Example:
#   bash eval_tinyllama_mtl_mlora.sh outputs/tinyllama_train_federated_epoch3_flround3_numB3_p1_0 outputs/tinyllama_eval_federated_epoch3_flround3_numB3_p1_0

set -euo pipefail

# Determine the repository root (the directory containing this script).
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

# Positional arguments
LOAD_DIR="${1:-}"
OUT_DIR="${2:-}"
if [[ -z "${LOAD_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: $0 <LOAD_DIR> <OUT_DIR> [extra python args...]" >&2
  echo "  LOAD_DIR: training output dir (contains run_config.json + ckpt_global_final/adapter_state_final)" >&2
  echo "  OUT_DIR : directory for evaluation results" >&2
  exit 1
fi

if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "LOAD_DIR '${LOAD_DIR}' not found or not a directory." >&2
  exit 1
fi

shift 2 || true
EXTRA_ARGS=("$@")

# Load container/env helpers
source script/common_env.sh

# Ensure output directory exists
mkdir -p "${OUT_DIR}"

# Python evaluation script
SCRIPT="run_glue_tinyllama_mtl_mlora_eval_single_gpu.py"

# Base arguments (most model hyperparams are auto-loaded from run_config.json)
ARGS=(
  --output_dir "${OUT_DIR}"
  --load_dir "${LOAD_DIR}"

  --eval_batch_size 16
  --num_workers 2

  --save_eval_details
  # --eval_details_max_examples 200
)

# Append extra CLI args (override defaults if duplicated)
ARGS+=("${EXTRA_ARGS[@]}")

# Build and run command inside container
CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

run_in_container "${CMD_STR}"