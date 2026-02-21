#!/bin/bash
# Shell script to train TinyLlama with multi‑task mLoRA on a Slurm cluster.
#
# This script assumes you have cloned the `MTL‑FLoRA` repository and placed
# the `tinyllama_glue_mtl_mlora` package and training script into the
# repository root.  It activates your Python environment, navigates to
# the project directory and invokes the training script with sensible
# defaults for a 1080 Ti GPU.  Any additional arguments passed to
# this script will be forwarded to the Python program (e.g. to change
# epochs, batch size or LoRA hyperparameters).

# Activate your virtual environment or load necessary modules here.  The
# following lines are examples and should be adjusted to your cluster's
# setup.  If you manage Python environments via `module load` or
# `conda`, uncomment and edit accordingly.

## Example: Load modules (comment out if not needed)
# module purge
# module load anaconda
# source activate flora_env

## Example: Activate a virtualenv in your home directory
# source ~/venvs/flora/bin/activate

# Change to the repository root.  Replace this with the absolute path
# to your clone of MTL‑FLoRA on the compute node.
cd "$(dirname "$0")"

# Optional: configure HuggingFace cache directories.  These
# environment variables ensure that models and datasets are stored
# under your home directory rather than in a shared scratch location.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}

# Create an output directory if it does not exist
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs_tinyllama_train"}
mkdir -p "$OUTPUT_DIR"

# Run the training script.  Feel free to adjust the default
# hyperparameters below.  Additional arguments can be appended when
# invoking this shell script (they will be forwarded to the Python
# script via "$@").
python run_glue_tinyllama_mtl_mlora_train_single_gpu.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_dir "$OUTPUT_DIR" \
  --epochs 1 \
  --train_batch_size 2 \
  --eval_batch_size 16 \
  --grad_accum_steps 4 \
  --learning_rate 2e-4 \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --num_B 2 \
  --temperature 0.1 \
  --num_fl_rounds 1 \
  --num_clients 1 \
  "$@"
