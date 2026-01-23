#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script/run_roberta_glue_mtl_mlora_train_single_gpu.sh <OUT_DIR> [extra args...]

OUT_DIR="${1:?Usage: $0 <OUT_DIR> [extra args...] }"
shift || true
EXTRA_ARGS=("$@")

# Load container/env helpers
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

mkdir -p "${OUT_DIR}"

SCRIPT="run_glue_roberta_mtl_mlora_train_single_gpu.py"

# Default args (you can override by passing the same flags in EXTRA_ARGS)
ARGS=(
  --output_dir "${OUT_DIR}"
  --epochs 1
  --train_batch_size 8
  --eval_batch_size 32
  --grad_accum_steps 2
  --learning_rate 2e-4
  --warmup_ratio 0.06
  --max_length 256
  
  --freeze_bias
  --freeze_layernorm

  # mLoRA
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.05
  --num_B 3
  --temperature 0.1

  # Mixed precision
  --fp16

  # Save a ckpt before eval each epoch
  --save_pre_eval_ckpt

  # Eval details dump (optional)
  --save_eval_details
  --eval_details_max_examples 200

  # Federated learning settings
  --num_fl_rounds 1
  --num_clients 2
  --dirichlet_alpha 1.0

  # Test mode (set via EXTRA_ARGS if needed)
  --test
)

CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}" "${EXTRA_ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

echo "[RUN] OUT_DIR=${OUT_DIR}"
echo "[RUN] REPO_DIR=${REPO_DIR}"
echo "[RUN] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "[RUN] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "[RUN] CMD=${CMD_STR}"

run_in_container "cd '${REPO_DIR}' && ${CMD_STR}"
