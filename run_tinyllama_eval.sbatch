#!/bin/bash
#SBATCH --job-name=mlora_eval
#SBATCH --partition=pascal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=mlora_eval_%j.log
#SBATCH --error=mlora_eval_%j.err

export SCRATCH_BASE="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
export HF_HOME="$SCRATCH_BASE/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

CONTAINER="~/FL/lasttry/MTL-LoRA/docker_1.sif"
SCRIPT="$HOME/FL/lasttry/MTL-LoRA/tinyllama_mlora_eval.sh"
CHECKPOINT="~/FL/lasttry/MTL-LoRA/tinyllama_output/adapter_model.bin"
OUTDIR="~/FL/lasttry/MTL-LoRA/eval_logs"

apptainer exec --nv \
  --bind "$HOME:$HOME" \
  --bind "$SCRATCH_BASE:$SCRATCH_BASE" \
  "$CONTAINER" \
  bash "$SCRIPT" "$CHECKPOINT" "$OUTDIR"
