#!/bin/bash
#SBATCH --job-name=tinyllama_mtl_eval
#SBATCH --output=logs/tinyllama_eval_%j.out
#SBATCH --error=logs/tinyllama_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=volta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-12:00:00

# Slurm submission script for evaluating a trained TinyLlama
# multi‑task mLoRA model.  This job typically requires less time
# and memory than training.  Adjust the resource requirements as
# needed.
#
# Usage:
#   sbatch eval_tinyllama_mtl_mlora.sbatch [--load_dir <checkpoint>]
#
# Any additional arguments will be forwarded to the evaluation
# shell script.  For example, to evaluate a specific checkpoint:
#
#   sbatch eval_tinyllama_mtl_mlora.sbatch --load_dir ./outputs_tinyllama_train/checkpoints/ckpt_5000

# Load modules or activate environments here if necessary.
# module purge
# module load anaconda
# source activate flora_env

# Ensure the log directory exists
mkdir -p logs

# Execute the TinyLlama evaluation script via Apptainer.  We derive
# default load and output directories if the user does not provide
# them explicitly.  The first non-option argument is treated as
# LOAD_DIR, the second as OUT_DIR.

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

# Parse positional arguments (those not starting with a dash)
positional=()
remaining=()
for arg in "$@"; do
  if [[ "$arg" == -* ]]; then
    remaining+=("$arg")
  else
    positional+=("$arg")
  fi
done

if [[ ${#positional[@]} -ge 1 ]]; then
  LOAD_DIR="${positional[0]}"
else
  LOAD_DIR="${REPO_ROOT}/outputs_tinyllama_train/checkpoints/ckpt_best"
fi
if [[ ${#positional[@]} -ge 2 ]]; then
  OUT_DIR="${positional[1]}"
else
  OUT_DIR="${REPO_ROOT}/outputs/tinyllama_eval_${SLURM_JOB_ID}"
fi

# Remaining arguments (options beginning with dash) are passed on
EXTRA_ARGS=("${remaining[@]}")

# Ensure output directory exists
mkdir -p "${OUT_DIR}"

# Call the top-level evaluation launcher which invokes the container
bash eval_tinyllama_mtl_mlora.sh "${LOAD_DIR}" "${OUT_DIR}" "${EXTRA_ARGS[@]}"