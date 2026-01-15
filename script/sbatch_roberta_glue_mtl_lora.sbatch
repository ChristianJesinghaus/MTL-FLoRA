#!/usr/bin/env bash
#SBATCH --job-name=roberta_glue_mlora
#SBATCH --partition=pascal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=pascal-node01.l3s.intra,pascal-node03.l3s.intra
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

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

# CPU threads: total for job; the run script divides by NPROC for torchrun workers
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Locale safe
export LANG=C
export LC_ALL=C

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0

# HF caches (HOME ist immer schreibbar)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

OUT_DIR="${REPO_ROOT}/outputs/roberta_glue_mlora_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

bash "${REPO_ROOT}/script/run_roberta_glue_mtl_lora.sh" "${OUT_DIR}"

echo "[INFO] Job finished at $(date)"
