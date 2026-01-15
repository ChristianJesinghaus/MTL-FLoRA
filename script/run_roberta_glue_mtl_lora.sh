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

# Container selection:
# Priority:
#   1) CONTAINER_IMAGE
#   2) MTL_LORA_CONTAINER
#   3) repo-local docker_1.sif
#   4) repo-local my-fluffy-cow.sif
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${MTL_LORA_CONTAINER:-}}"
if [[ -z "${CONTAINER_IMAGE}" ]]; then
  if [[ -f "${REPO_DIR}/docker_1.sif" ]]; then
    CONTAINER_IMAGE="${REPO_DIR}/docker_1.sif"
  elif [[ -f "${REPO_DIR}/my-fluffy-cow.sif" ]]; then
    CONTAINER_IMAGE="${REPO_DIR}/my-fluffy-cow.sif"
  else
    echo "[ERROR] No container image found. Set CONTAINER_IMAGE or put docker_1.sif/my-fluffy-cow.sif in repo root."
    exit 1
  fi
fi

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
  echo "[ERROR] Container image not found: ${CONTAINER_IMAGE}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

# HF caches (safe defaults)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Make sure env is available inside container even if sanitized
export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export APPTAINERENV_TOKENIZERS_PARALLELISM="false"
export APPTAINERENV_PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export APPTAINERENV_LANG="${LANG:-C}"
export APPTAINERENV_LC_ALL="${LC_ALL:-C}"

# Pass HF token if present (do NOT echo it)
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

# Determine how many GPUs are available (for torchrun)
NPROC=1
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  # SLURM_GPUS_ON_NODE is usually a number, but handle comma lists too
  if [[ "${SLURM_GPUS_ON_NODE}" == *","* ]]; then
    IFS=',' read -ra _ARR <<< "${SLURM_GPUS_ON_NODE}"
    NPROC="${#_ARR[@]}"
  else
    NPROC="${SLURM_GPUS_ON_NODE}"
  fi
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  IFS=',' read -ra _ARR <<< "${SLURM_JOB_GPUS}"
  NPROC="${#_ARR[@]}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_ARR[@]}"
fi
NPROC="${NPROC:-1}"

# Avoid CPU oversubscription in DDP
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${NPROC}" -gt 1 ]]; then
  per_proc=$(( SLURM_CPUS_PER_TASK / NPROC ))
  if [[ "${per_proc}" -lt 1 ]]; then per_proc=1; fi
  export OMP_NUM_THREADS="${per_proc}"
fi

SCRIPT="run_glue_roberta_mtl_lora.py"

# Base args (training defaults)
ARGS=(
  --output_dir "${OUT_DIR}"
  --epochs 2
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
  --save_pre_eval_ckpt

  # eval details dump (you can override via EXTRA_ARGS)
  --save_eval_details
  --eval_details_max_examples 200
)

# Append extra args from CLI
ARGS+=("${EXTRA_ARGS[@]}")


# - For torchrun, pass the TRAINING SCRIPT directly (NOT "python -u ...")
# - For single GPU, call python3 -u
if [[ "${NPROC}" -gt 1 ]]; then
  CMD=(torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT}" "${ARGS[@]}")
else
  CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}")
fi

CMD_STR="$(printf '%q ' "${CMD[@]}")"

echo "[RUN] OUT_DIR=${OUT_DIR}"
echo "[RUN] NPROC=${NPROC}"
echo "[RUN] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "[RUN] HF_HOME=${HF_HOME}"
echo "[RUN] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "[RUN] CMD=${CMD_STR}"

exec apptainer exec --nv \
  -B "${REPO_DIR}:${REPO_DIR}" \
  -B "${HF_HOME}:${HF_HOME}" \
  -B /data:/data \
  "${CONTAINER_IMAGE}" \
  bash -lc "cd '${REPO_DIR}' && ${CMD_STR}"
