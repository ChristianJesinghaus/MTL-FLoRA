#!/bin/bash
#SBATCH --job-name=tinyllama_mtl_eval
#SBATCH --output=logs/tinyllama_eval_%j.out
#SBATCH --error=logs/tinyllama_eval_%j.err
#SBATCH --ntasks=1
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

# Execute the evaluation shell script with any provided arguments
bash eval_tinyllama_mtl_mlora.sh "$@"