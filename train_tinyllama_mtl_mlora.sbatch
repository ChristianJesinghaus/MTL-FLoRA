#!/bin/bash
#SBATCH --job-name=tinyllama_mtl_train
#SBATCH --output=logs/tinyllama_train_%j.out
#SBATCH --error=logs/tinyllama_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00

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

# Execute the training shell script.  Use `bash` rather than
# `srun` so that the script can manage its own environment.  Pass
# all command‑line arguments through "$@".
bash train_tinyllama_mtl_mlora.sh "$@"