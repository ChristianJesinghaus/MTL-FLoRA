#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script/run_roberta_glue_mtl_lora.sh <OUT_DIR> [extra args...]
#
# Examples:
#   bash script/run_roberta_glue_mtl_lora.sh outputs/roberta_glue_mlora_${SLURM_JOB_ID}
#   bash script/run_roberta_glue_mtl_lora.sh outputs/eval_${SLURM_JOB_ID} --eval_only --load_dir outputs/roberta_glue_mlora_23446 --eval_details_max_examples -1

OUT_DIR="${1:-outputs/roberta_glue_mlora_run}"
shift || true
EXTRA_ARGS=("$@")

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/data/singularity_images/flash_llm_pytorch2_1_0.sif}"

mkdir -p "${OUT_DIR}"

# Determine how many GPUs are available (so we can decide between python vs torchrun)
NPROC=1
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  NPROC="${SLURM_GPUS_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra DEV_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#DEV_ARR[@]}"
fi

PY_ARGS=(
  python3 -u run_glue_roberta_mtl_lora.py
  --output_dir "${OUT_DIR}"
  --epochs 1
  --train_batch_size 8
  --eval_batch_size 32
  --grad_accum_steps 2
  --learning_rate 2e-4
  --warmup_ratio 0.06
  --max_length 256
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.05
  --num_B 3
  --temperature 0.1
  --fp16
  --save_pre_eval_ckpt

  --save_eval_details
  --eval_details_max_examples 200
)

# Append extra args from CLI (e.g. --eval_only --load_dir ...)
PY_ARGS+=("${EXTRA_ARGS[@]}")

if [[ "${NPROC}" -gt 1 ]]; then
  CMD=(torchrun --standalone --nproc_per_node="${NPROC}" "${PY_ARGS[@]}")
else
  CMD=("${PY_ARGS[@]}")
fi

CMD_STR="$(printf '%q ' "${CMD[@]}")"

echo "[RUN] OUT_DIR=${OUT_DIR}"
echo "[RUN] NPROC=${NPROC}"
echo "[RUN] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "[RUN] CMD=${CMD_STR}"

exec apptainer exec --nv \
  -B "${REPO_DIR}:${REPO_DIR}" \
  -B /data:/data \
  "${CONTAINER_IMAGE}" \
  bash -lc "cd '${REPO_DIR}' && ${CMD_STR}"
