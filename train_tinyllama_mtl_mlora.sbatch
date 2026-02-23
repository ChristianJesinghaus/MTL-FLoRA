#!/bin/bash
#SBATCH --job-name=tinyllama_mtl_train
#SBATCH --output=logs/tinyllama_train_%j.out
#SBATCH --error=logs/tinyllama_train_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=volta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# Slurm submission script for training the TinyLlama multi‑task mLoRA
# model.  Adjust the resource requirements (cpus, memory, time)
# according to your cluster's policies and the size of your model.
#
# Usage:
#   sbatch train_tinyllama_mtl_mlora.sbatch [additional args]
#
# Any additional arguments provided to `sbatch` will be passed
# through to the underlying training shell script.  For example:
#
#   sbatch train_tinyllama_mtl_mlora.sbatch --epochs 2 --train_batch_size 4

# Load modules or activate environments here if necessary.
# For example:
# module purge
# module load anaconda
# source activate flora_env

# Create a directory for Slurm logs if it doesn't exist.
mkdir -p logs

# Execute the TinyLlama training script via Apptainer.  We derive a
# unique output directory from the Slurm job ID unless the user
# specifies one via the first positional argument to the script.

# Set the repository root to the directory where the job was
# submitted (assumed to contain the code and script folder)
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

# If the user supplied a first argument that does not begin with a dash,
# treat it as the desired output directory.  Otherwise derive one
# automatically under ./outputs.
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
  OUT_DIR="$1"
  shift
else
  OUT_DIR="${REPO_ROOT}/outputs/tinyllama_train_${SLURM_JOB_ID}"
fi
mkdir -p "${OUT_DIR}"

# Call the top-level training launcher which invokes the container.  Any
# remaining arguments are forwarded to override defaults.
bash train_tinyllama_mtl_mlora.sh "${OUT_DIR}" "$@"