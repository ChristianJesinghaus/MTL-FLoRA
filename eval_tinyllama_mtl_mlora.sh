#!/bin/bash
# Shell script to evaluate a TinyLlama multi‑task mLoRA model on a Slurm cluster.
#
# This script loads a trained TinyLlama mLoRA model and runs the
# evaluation script to compute metrics on the validation splits of
# the GLUE tasks.  It mirrors the training launcher but assumes
# that a checkpoint directory has already been produced.  Any
# additional command‑line arguments will be forwarded to the Python
# program.

# Activate your virtual environment or load necessary modules here.
# Modify the following lines to suit your cluster's configuration.

## Example: Load modules (comment out if not needed)
# module purge
# module load anaconda
# source activate flora_env

## Example: Activate a virtualenv in your home directory
# source ~/venvs/flora/bin/activate

# Change to the repository root.  This assumes the script lives in
# the project root; adjust the path if placed elsewhere.
cd "$(dirname "$0")"

# Optional: configure HuggingFace cache directories to avoid
# downloading models repeatedly.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}

# Directory containing the checkpoint to evaluate.  Override with
# `LOAD_DIR` environment variable or pass `--load_dir` as an
# additional argument when calling this script.
LOAD_DIR=${LOAD_DIR:-"./outputs_tinyllama_train/checkpoints/ckpt_best"}

# Output directory for evaluation results.  Override with
# `OUTPUT_DIR` or `--output_dir` via command‑line arguments.
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs_tinyllama_eval"}
mkdir -p "$OUTPUT_DIR"

# Run the evaluation.  The LoRA hyperparameters must match those
# used during training.  Additional arguments are accepted and
# forwarded via "$@".
python run_glue_tinyllama_mtl_mlora_eval_single_gpu.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --load_dir "$LOAD_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --num_B 2 \
  --temperature 0.1 \
  --eval_batch_size 16 \
  --fp16 \
  "$@"