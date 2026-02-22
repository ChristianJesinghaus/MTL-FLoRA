#!/usr/bin/env bash
#SBATCH --job-name=roberta_glue_mlora_eval_1gpu
#SBATCH --partition=volta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
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

if [[ -z "${LOAD_DIR:-}" ]]; then
  echo "[ERROR] Please set LOAD_DIR (path to a training output directory)." >&2
  echo "Example: sbatch --export=ALL,LOAD_DIR=outputs/roberta_glue_mlora_1gpu_24163 script/sbatch_roberta_glue_mtl_mlora_eval_1gpu.sbatch" >&2
  exit 1
fi

# Where to write eval outputs
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/roberta_glue_mlora_eval_1gpu_${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

# Optional knobs via env
EVAL_DETAILS_MAX_EXAMPLES="${EVAL_DETAILS_MAX_EXAMPLES:-200}"
ONLY_ERRORS_FLAG=""
if [[ "${EVAL_DETAILS_ONLY_ERRORS:-0}" == "1" ]]; then
  ONLY_ERRORS_FLAG="--eval_details_only_errors"
fi

bash "${REPO_ROOT}/script/run_roberta_glue_mtl_mlora_eval_single_gpu.sh" "${LOAD_DIR}" "${OUT_DIR}" \
  --eval_details_max_examples "${EVAL_DETAILS_MAX_EXAMPLES}" \
  ${ONLY_ERRORS_FLAG}

echo "[INFO] Job finished at $(date)"
