#!/usr/bin/env bash
#
# Top-level launcher for evaluating a TinyLlama multi-task mLoRA model.
#
# Usage:
#   bash eval_tinyllama_mtl_mlora.sh <LOAD_PATH> <OUT_DIR> [extra args...]
#
# LOAD_PATH can be either:
#   - a checkpoint file (*.pt)  -> passed as --load_ckpt
#   - a directory with adapter_state.pt + heads_state.pt -> passed as --load_dir
#

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

# Positional args
LOAD_PATH="${1:-./outputs_tinyllama_train}"
OUT_DIR="${2:-./outputs_tinyllama_eval}"
shift 2 || true
EXTRA_ARGS=("$@")

# Make absolute paths
abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}
LOAD_PATH="$(abspath "${LOAD_PATH}")"
OUT_DIR="$(abspath "${OUT_DIR}")"

# Decide whether to use --load_ckpt or --load_dir
LOAD_ARGS=()
if [[ -f "${LOAD_PATH}" ]]; then
  LOAD_ARGS=(--load_ckpt "${LOAD_PATH}")
elif [[ -d "${LOAD_PATH}" ]]; then
  LOAD_ARGS=(--load_dir "${LOAD_PATH}")
else
  echo "[ERR] LOAD_PATH does not exist: ${LOAD_PATH}" >&2
  exit 1
fi

# Load container/env helpers
source "${REPO_DIR}/script/common_env.sh"

mkdir -p "${OUT_DIR}"

SCRIPT="run_glue_tinyllama_mtl_mlora_eval_single_gpu.py"

# Base args – NO lora_r / num_B / global_num_B / block_size defaults here!
# These MUST come from EXTRA_ARGS so there's no risk of wrong defaults winning.
ARGS=(
  --output_dir "${OUT_DIR}"
  "${LOAD_ARGS[@]}"

  --eval_batch_size 16
  --max_length 256
  --num_workers 2

  --lora_alpha 16
  --lora_dropout 0.05

  --temperature 0.1
  --fp16

  --save_eval_details
)

# Append extra CLI args (these now SET lora_r, num_B, global_num_B, block_size)
ARGS+=("${EXTRA_ARGS[@]}")

CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

echo "[RUN] CMD=${CMD_STR}"

run_in_container "${CMD_STR}"