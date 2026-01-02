#!/bin/bash
#SBATCH --job-name=mlora_train
#SBATCH --partition=pascal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=mlora_train_%j.log
#SBATCH --error=mlora_train_%j.err

export SCRATCH_BASE="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
export HF_HOME="$SCRATCH_BASE/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Hier reale Pfade angeben!
CONTAINER="docker_1.sif"
SCRIPT="tinyllama_mlora_train.sh"


mkdir -p "$SCRATCH_BASE"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# LoRA-Rank und Alpha übergeben; ggf. Pfad zum Datensatz und Output-Dir ergänzen
apptainer exec --nv \
  --bind "$HOME:$HOME" \
  --bind "$SCRATCH_BASE:$SCRATCH_BASE" \
  "$CONTAINER" \
  bash "$SCRIPT" 8 16 "commonsense_170k_taskid.json" "tinyllama_output"
