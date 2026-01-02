#!/bin/bash
# Evaluate a TinyLlama MTL‑LoRA checkpoint on the commonsense benchmarks.
#
# Usage:
#   bash tinyllama_mlora_eval.sh <CHECKPOINT_PATH> <OUTPUT_DIR>
#
# The script reads the list of visible GPUs from CUDA_VISIBLE_DEVICES and
# distributes tasks across them in a round‑robin fashion.  Each evaluation
# runs as a background job and writes its output to a separate text file in
# the specified output directory.  Flash‑attention and memory‑efficient SDPA
# are disabled, and FP32 is enforced by default.

set -euo pipefail

CKPT=${1:?"Path to the trained LoRA checkpoint must be supplied"}
OUTDIR=${2:?"Path to the directory where logs should be written"}

mkdir -p "$OUTDIR"

# Node‑local scratch and Hugging Face caches
SCRATCH_BASE="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-local}}"
export HF_HOME="$SCRATCH_BASE/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Disable flash‑attention and memory‑efficient SDPA kernels
export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1
export USE_FP32=1

# Locale settings
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Base model and LoRA hyper‑parameters (as used in training)
BASE_MODEL='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
LORA_R=8
LORA_ALPHA=16
LAMBDA_NUM=8
NUM_B=3
TEMPERATURE=0.1
TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj"]'

# List of commonsense datasets used in the paper
TASKS=(
  boolq
  piqa
  social_i_qa
  hellaswag
  winogrande
  ARC-Challenge
  ARC-Easy
  openbookqa
)

# Read GPUs from CUDA_VISIBLE_DEVICES, fallback to a single GPU
IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=${#GPU_IDS[@]}

if (( NUM_GPUS == 0 )); then
  GPU_IDS=(0)
  NUM_GPUS=1
fi

for idx in "${!TASKS[@]}"; do
  dataset=${TASKS[$idx]}
  gpu_idx=$(( idx % NUM_GPUS ))
  gpu_id=${GPU_IDS[$gpu_idx]}

  echo "[tinyllama_mlora_eval] Launching $dataset on GPU $gpu_id"
  (
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python mlora_evaluate.py \
      --dataset "$dataset" \
      --model TinyLlama-1.1B \
      --adapter mlora \
      --base_model "$BASE_MODEL" \
      --lora_target_modules "$TARGET_MODULES" \
      --batch_size 1 \
      --lora_r $LORA_R \
      --lora_alpha $LORA_ALPHA \
      --lambda_num $LAMBDA_NUM \
      --num_B $NUM_B \
      --temperature $TEMPERATURE \
      --lora_weights "$CKPT" \
      | tee -a "$OUTDIR/${dataset}.txt"
  ) &
done

wait
echo "Evaluation completed.  Results written to $OUTDIR/*.txt"