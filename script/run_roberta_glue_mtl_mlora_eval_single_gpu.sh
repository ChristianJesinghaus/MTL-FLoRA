#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script/run_roberta_glue_mtl_mlora_eval_single_gpu.sh <LOAD_DIR> <OUT_DIR> [extra args...]
#
# Examples:
#   bash script/run_roberta_glue_mtl_mlora_eval_single_gpu.sh outputs/roberta_run_123 outputs/roberta_eval_456 \
#        --eval_details_max_examples -1

LOAD_DIR="${1:?Usage: $0 <LOAD_DIR> <OUT_DIR> [extra args...] }"
OUT_DIR="${2:?Usage: $0 <LOAD_DIR> <OUT_DIR> [extra args...] }"
shift 2 || true
EXTRA_ARGS=("$@")

# Load container/env helpers
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

mkdir -p "${OUT_DIR}"

SCRIPT="run_glue_roberta_mtl_mlora_eval_single_gpu.py"

# Base args
ARGS=(
  --output_dir "${OUT_DIR}"
  --load_dir "${LOAD_DIR}"
  --eval_batch_size 32
  --max_length 256
  --num_workers 2

  # mLoRA MUST match training (override via EXTRA_ARGS if needed)
  --lora_r 32
  --lora_alpha 16
  --lora_dropout 0.05
  --num_B 3
  --temperature 0.1

  # Mixed precision
  --fp16

  # eval details dump (override via EXTRA_ARGS)
  --save_eval_details
  --eval_details_max_examples 200
  --load_global_model
)

# Append extra args from CLI
ARGS+=("${EXTRA_ARGS[@]}")

CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

echo "[RUN] LOAD_DIR=${LOAD_DIR}"
echo "[RUN] OUT_DIR=${OUT_DIR}"
echo "[RUN] HF_HOME=${HF_HOME}"
echo "[RUN] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "[RUN] CMD=${CMD_STR}"

run_in_container "cd '${REPO_DIR}' && ${CMD_STR}"
