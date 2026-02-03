#!/usr/bin/env bash
set -euo pipefail
#SBATCH --job-name=roberta_glue_mlora_1gpu
#SBATCH --partition=pascal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END,FAIL

trap 'echo "[ERROR] Script failed at line $LINENO: $BASH_COMMAND" >&2' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/outputs"

echo "=========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=========================================="
echo "[INFO] Job started on $(hostname) at $(date)"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"

# Threads
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Locale safe
export LANG=C
export LC_ALL=C

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0

# HF caches
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Optional token file
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

OUT_DIR="${REPO_ROOT}/outputs/roberta_glue_mlora_1gpu_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] Calling: bash \"${REPO_ROOT}/script/run_roberta_glue_mtl_mlora_train_single_gpu.sh\" \"${OUT_DIR}\""

bash "${REPO_ROOT}/script/run_roberta_glue_mtl_mlora_train_single_gpu.sh" "${OUT_DIR}"

echo "[INFO] Job finished at $(date)"
