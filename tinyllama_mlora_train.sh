#!/bin/bash
# Train MTL‑LoRA on the commonsense reasoning dataset using TinyLlama.
#
# This script wraps `mlora_finetune.py` with the environment settings required
# for Pascal GPUs and the university cluster.  It accepts two mandatory
# arguments: the LoRA rank `r` and the scaling factor `alpha`.  Optional
# arguments can override the dataset path, output directory and cache
# directory.
#
# Usage:
#   bash tinyllama_mlora_train.sh <r> <alpha> [DATA_PATH] [OUTPUT_DIR] [CACHE_DIR]

set -euo pipefail

# Resolve dataset and output directories from positional parameters or use
# sensible defaults.
r_value=${1:?"LoRA rank (r) must be supplied"}
alpha_value=${2:?"LoRA scaling factor (alpha) must be supplied"}
DATA_PATH=${3:-""}      # path to commonsense_170k_taskid.json
OUTPUT_DIR=${4:-"./tinyllama_mlora_output"}
CACHE_DIR=${5:-""}

# Define node‑local scratch base.  When running under Slurm,
# SLURM_TMPDIR points to a local filesystem.  Otherwise fall back to /tmp.
SCRATCH_BASE="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-local}}"

# Configure Hugging Face caches on the node‑local scratch.  This avoids
# corruption when multiple workers download datasets simultaneously on NFS.
export HF_HOME="$SCRATCH_BASE/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Remove cached Winogrande splits if present to handle the
# NonMatchingSplitsSizesError noted in the DoRA repository.  A fresh
# download will be triggered automatically.
rm -rf "$HF_DATASETS_CACHE/winogrande" "$HF_DATASETS_CACHE/downloads" 2>/dev/null || true
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Disable flash‑attention and memory‑efficient SDPA kernels.  These kernels
# rely on tensor cores that are not available on Pascal GPUs and will
# otherwise cause errors.  We also force full precision via USE_FP32.
export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1
export USE_FP32=1

# Set locale to avoid warnings about missing UTF‑8 locales inside Apptainer.
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Use a conservative batch size and sequence length for limited VRAM.
BATCH_SIZE=5
CUTOFF_LEN=256

# If a custom cache directory is provided, use it; otherwise reuse the scratch
# cache created above.
if [[ -n "$CACHE_DIR" ]]; then
  CACHE_ARG="--cache_dir $CACHE_DIR"
else
  CACHE_ARG="--cache_dir $HF_HOME/cache"
fi

# Base model: TinyLlama instead of LLaMA2.  You can replace this with any
# other TinyLlama variant available on Hugging Face.
BASE_MODEL='TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# Launch training with DeepSpeed.  Use the include list from Slurm
# automatically if provided; otherwise default to a single node.

python3 mlora_finetune.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $CACHE_ARG \
    --adapter_name mlora \
    --batch_size $BATCH_SIZE \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len $CUTOFF_LEN \
    --save_step 1000 \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]' \
    --lora_r $r_value \
    --lora_alpha $alpha_value \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \