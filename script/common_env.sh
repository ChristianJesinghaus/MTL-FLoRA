#!/usr/bin/env bash
set -euo pipefail

# Common helpers for Apptainer-based runs on the cluster.
# Sourced by the train/eval runner scripts.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Container selection priority:
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
    echo "[ERROR] No container image found. Set CONTAINER_IMAGE or put docker_1.sif/my-fluffy-cow.sif in repo root." >&2
    exit 1
  fi
fi
if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
  echo "[ERROR] Container image not found: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

# HF caches (safe defaults)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Ensure env vars are visible inside the container even if sanitized
export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export APPTAINERENV_TOKENIZERS_PARALLELISM="false"
export APPTAINERENV_PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export APPTAINERENV_LANG="${LANG:-C}"
export APPTAINERENV_LC_ALL="${LC_ALL:-C}"

# Pass HF token if present (do NOT echo it)
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

# CPU threads (avoid oversubscription)
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
fi

# Force single GPU visibility if user accidentally has multiple in CUDA_VISIBLE_DEVICES
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

run_in_container() {
  local cmd_str="$1"

  echo "[RUN] REPO_DIR=${REPO_DIR}"
  echo "[RUN] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
  echo "[RUN] HF_HOME=${HF_HOME}"
  echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[RUN] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
  echo "[RUN] CMD=${cmd_str}"

  exec apptainer exec --nv \
    -B "${REPO_DIR}:${REPO_DIR}" \
    -B "${HF_HOME}:${HF_HOME}" \
    -B /data:/data \
    "${CONTAINER_IMAGE}" \
    bash -lc "cd '${REPO_DIR}' && ${cmd_str}"
}
