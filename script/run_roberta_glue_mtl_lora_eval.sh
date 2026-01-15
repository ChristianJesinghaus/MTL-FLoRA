#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script/run_roberta_glue_mtl_lora_eval.sh <LOAD_DIR> <EVAL_OUT_DIR>

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOAD_DIR="${1:?Missing LOAD_DIR (e.g., outputs/roberta_glue_mlora_23446)}"
EVAL_OUT_DIR="${2:?Missing EVAL_OUT_DIR}"

# Container selection:
# - if MTL_LORA_CONTAINER is set, use it
# - else prefer docker_1.sif, else my-fluffy-cow.sif
CONTAINER="${MTL_LORA_CONTAINER:-}"
if [[ -z "${CONTAINER}" ]]; then
  if [[ -f "${REPO_ROOT}/docker_1.sif" ]]; then
    CONTAINER="${REPO_ROOT}/docker_1.sif"
  else
    CONTAINER="${REPO_ROOT}/my-fluffy-cow.sif"
  fi
fi

if [[ ! -f "${CONTAINER}" ]]; then
  echo "[ERROR] Container not found at: ${CONTAINER}"
  exit 1
fi

mkdir -p "${EVAL_OUT_DIR}"

# HF caches (use HF_HOME only; TRANSFORMERS_CACHE is deprecated)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Propagate into container even if apptainer cleans env
export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export APPTAINERENV_TOKENIZERS_PARALLELISM="false"

# Pass HF token if present (do NOT echo it)
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

# Locale (avoid setlocale warnings)
export LANG="${LANG:-C}"
export LC_ALL="${LC_ALL:-C}"
export APPTAINERENV_LANG="${LANG}"
export APPTAINERENV_LC_ALL="${LC_ALL}"

# Logging
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export APPTAINERENV_PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"


# Eval-details controls via env

# Defaults:
#   SAVE_EVAL_DETAILS=1           schreibt JSONL Dumps
#   EVAL_DETAILS_MAX_EXAMPLES=200  pro Task/Split (global); -1 = alles
#   ONLY_ERRORS=0                   1 = nur falsche Beispiele speichern
#   EVAL_DETAILS_TOPK=2             top-k probs speichern
#   STSB_ABS_ERR_THRESHOLD=0.5      bei ONLY_ERRORS: abs err >= thresh
#   USE_FP16=1                      eval mit autocast fp16
SAVE_EVAL_DETAILS="${SAVE_EVAL_DETAILS:-1}"
EVAL_DETAILS_MAX_EXAMPLES="${EVAL_DETAILS_MAX_EXAMPLES:-200}"
# unterstütze beide Env-Namen:
ONLY_ERRORS="${ONLY_ERRORS:-${EVAL_DETAILS_ONLY_ERRORS:-0}}"
EVAL_DETAILS_TOPK="${EVAL_DETAILS_TOPK:-2}"
STSB_ABS_ERR_THRESHOLD="${STSB_ABS_ERR_THRESHOLD:-0.5}"
USE_FP16="${USE_FP16:-1}"

# Determine GPU count -> optionally torchrun for multi-GPU eval
NPROC=1
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  NPROC="${SLURM_GPUS_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra DEV_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#DEV_ARR[@]}"
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] CONTAINER=${CONTAINER}"
echo "[INFO] LOAD_DIR=${LOAD_DIR}"
echo "[INFO] EVAL_OUT_DIR=${EVAL_OUT_DIR}"
echo "[INFO] HF_HOME=${HF_HOME}"
echo "[INFO] HF token present? $([[ -n "${HF_TOKEN:-}" || -n "${HUGGINGFACE_HUB_TOKEN:-}" ]] && echo yes || echo no)"
echo "[INFO] SAVE_EVAL_DETAILS=${SAVE_EVAL_DETAILS} EVAL_DETAILS_MAX_EXAMPLES=${EVAL_DETAILS_MAX_EXAMPLES} ONLY_ERRORS=${ONLY_ERRORS}"
echo "[INFO] NPROC=${NPROC} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"

# Build python args
PY_ARGS=(run_glue_roberta_mtl_lora.py
  --eval_only
  --load_dir "${LOAD_DIR}"
  --output_dir "${EVAL_OUT_DIR}"
)

if [[ "${USE_FP16}" == "1" ]]; then
  PY_ARGS+=(--fp16)
fi

if [[ "${SAVE_EVAL_DETAILS}" == "1" ]]; then
  PY_ARGS+=(--save_eval_details
    --eval_details_max_examples "${EVAL_DETAILS_MAX_EXAMPLES}"
    --eval_details_topk "${EVAL_DETAILS_TOPK}"
    --stsb_abs_err_threshold "${STSB_ABS_ERR_THRESHOLD}"
  )
  if [[ "${ONLY_ERRORS}" == "1" ]]; then
    PY_ARGS+=(--eval_details_only_errors)
  fi
fi

# Filter any accidental empty args (belt & suspenders)
CLEAN_ARGS=()
for a in "${PY_ARGS[@]}"; do
  if [[ -n "${a}" ]]; then
    CLEAN_ARGS+=("${a}")
  fi
done
PY_ARGS=("${CLEAN_ARGS[@]}")

# Compose command (single GPU: python3 -u; multi GPU: torchrun)
if [[ "${NPROC}" -gt 1 ]]; then
  CMD=(torchrun --standalone --nproc_per_node="${NPROC}" "${PY_ARGS[@]}")
else
  CMD=(python3 -u "${PY_ARGS[@]}")
fi

CMD_STR="$(printf '%q ' "${CMD[@]}")"
echo "[INFO] Running: ${CMD_STR}"

exec apptainer exec --nv \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --bind /data:/data \
  "${CONTAINER}" \
  bash -lc "cd '${REPO_ROOT}' && ${CMD_STR}"
