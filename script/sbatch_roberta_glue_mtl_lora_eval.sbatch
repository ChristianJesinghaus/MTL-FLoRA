#!/usr/bin/env bash
#SBATCH --job-name=roberta_glue_mlora_eval
#SBATCH --partition=pascal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_roberta_glue_mlora_eval_%j.out
#SBATCH --error=slurm_roberta_glue_mlora_eval_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/outputs"

echo "=========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=========================================="
echo "[INFO] Job started on $(hostname) at $(date)"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Locale safe
export LANG=C
export LC_ALL=C

# Logging / tqdm in Slurm output
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0

# HF caches (HOME ist schreibbar)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Optional: HF token aus Datei laden
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Eval-details defaults (kannst du per sbatch --export überschreiben)
export SAVE_EVAL_DETAILS="${SAVE_EVAL_DETAILS:-1}"
export EVAL_DETAILS_MAX_EXAMPLES="${EVAL_DETAILS_MAX_EXAMPLES:-200}"   # -1 für alle
export EVAL_DETAILS_ONLY_ERRORS="${EVAL_DETAILS_ONLY_ERRORS:-0}"
export EVAL_DETAILS_TOPK="${EVAL_DETAILS_TOPK:-2}"
export STSB_ABS_ERR_THRESHOLD="${STSB_ABS_ERR_THRESHOLD:-0.5}"
export EVAL_FP16="${EVAL_FP16:-1}"

# Resolve LOAD_DIR robustly
LOAD_DIR="${LOAD_DIR:-${1:-}}"
if [[ -z "${LOAD_DIR}" ]]; then
  LOAD_DIR="$(ls -td "${REPO_ROOT}/outputs"/roberta_glue_mlora_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${LOAD_DIR}" ]]; then
  echo "[ERROR] Could not determine LOAD_DIR."
  echo "       Provide it as env var or argument:"
  echo "         sbatch script/sbatch_roberta_glue_mtl_lora_eval.sbatch outputs/roberta_glue_mlora_23446"
  exit 1
fi
if [[ "${LOAD_DIR}" != /* ]]; then
  LOAD_DIR="${REPO_ROOT}/${LOAD_DIR}"
fi
if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "[ERROR] LOAD_DIR does not exist: ${LOAD_DIR}"
  exit 1
fi

EVAL_OUT_DIR="${EVAL_OUT_DIR:-${LOAD_DIR}/eval_runs/eval_${SLURM_JOB_ID}}"
mkdir -p "${EVAL_OUT_DIR}"

echo "[INFO] LOAD_DIR=${LOAD_DIR}"
echo "[INFO] EVAL_OUT_DIR=${EVAL_OUT_DIR}"
echo "[INFO] SAVE_EVAL_DETAILS=${SAVE_EVAL_DETAILS} EVAL_DETAILS_MAX_EXAMPLES=${EVAL_DETAILS_MAX_EXAMPLES}"

bash "${REPO_ROOT}/script/run_roberta_glue_mtl_lora_eval.sh" "${LOAD_DIR}" "${EVAL_OUT_DIR}"

echo "[INFO] Eval job finished at $(date)"
