#!/usr/bin/env bash
#
# Top-level launcher for training the TinyLlama multi‑task mLoRA model.
#
# This script sources the shared `script/common_env.sh` helper to
# identify and configure the Apptainer container.  It then uses
# `run_in_container` to execute the Python training program inside
# that container.  You can supply an output directory as the first
# positional argument; if omitted, a sensible default under
# `./outputs_tinyllama_train` will be used.  Additional flags
# override the default hyperparameters specified below.

set -euo pipefail

# Determine the repository root (the directory containing this script).
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

# Parse the first positional argument as the output directory, or set a default.
OUT_DIR="${1:-./outputs_tinyllama_train}"
shift || true
EXTRA_ARGS=("$@")

# Load common helpers from the script folder.  This defines
# `run_in_container` and configures HF cache environment variables.
source script/common_env.sh

# Ensure the output directory exists.
mkdir -p "${OUT_DIR}"

# Python script to be executed inside the container.
SCRIPT="run_glue_tinyllama_mtl_mlora_train_single_gpu.py"

# Default arguments tuned for a 1080 Ti GPU.  Feel free to override
# any of these by passing the same flag via EXTRA_ARGS on the
# command line.
ARGS=(
  --output_dir "${OUT_DIR}"
  --epochs 1
  --train_batch_size 2
  --eval_batch_size 16
  --grad_accum_steps 4
  --learning_rate 2e-4
  --warmup_ratio 0.06
  --max_length 256

  --freeze_bias
  --freeze_layernorm

  # mLoRA hyperparameters (small rank and few B matrices)
  --lora_r 4
  --lora_alpha 16
  --lora_dropout 0.05
  --num_B 2
  --temperature 0.1

  # Mixed precision
  --fp16

  # Save a checkpoint before evaluation each epoch
  --save_pre_eval_ckpt

  # Dump evaluation details after each epoch
  --save_eval_details
  --eval_details_max_examples 200

  # Federated learning settings
  --num_fl_rounds 1
  --num_clients 1
  --dirichlet_alpha 1.0

  # Enable test mode by default (override with --no-test in EXTRA_ARGS)
  --test
)

# Append any extra CLI arguments (these override defaults if duplicated).
ARGS+=("${EXTRA_ARGS[@]}")

# Construct the Python command to run inside the container.
CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

# Run the command within the Apptainer container.  `run_in_container`
# automatically mounts the repository and passes through relevant
# environment variables (HF caches, CUDA settings, etc.).
run_in_container "${CMD_STR}"
