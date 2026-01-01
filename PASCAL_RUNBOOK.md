# Pascal (GTX 1080 Ti) runbook

These steps reproduce the paper-style mLoRA training and multiple-choice evaluation on the Pascal GPU partition (`pascal`). The defaults avoid bf16/flash attention, keep batch sizes small, and isolate Hugging Face caches to prevent corruption.

## Common environment defaults
Add these to your `~/.bashrc` or at the top of every Slurm script:

```bash
# Pascal-safe kernels
export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1

# Isolate Hugging Face caches per run
export HF_HOME=${HF_HOME:-$PWD/.hf_pascal/${SLURM_JOB_ID:-$$}}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE"
```

## 1) Export evaluation datasets (handles cache corruption)
Copy-paste to `export_eval_pascal.sbatch` and submit with `sbatch export_eval_pascal.sbatch`.

```bash
#!/bin/bash
#SBATCH -p pascal
#SBATCH -J eval-export
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -o logs/export-%j.out
#SBATCH -e logs/export-%j.err
set -euxo pipefail

export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1
export HF_HOME=${HF_HOME:-$PWD/.hf_pascal/${SLURM_JOB_ID:-$$}}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE" logs

run_export() {
  python script/export_eval_datasets_paperlike.py --out_dir "$PWD/eval_datasets"
}

if ! run_export; then
  echo "Corrupted HF cache detected; clearing and retrying" >&2
  rm -rf "$HF_DATASETS_CACHE"
  mkdir -p "$HF_DATASETS_CACHE"
  run_export
fi
```

## 2) Train TinyLlama on Commonsense (paper-style mLoRA)
Copy-paste to `train_tinyllama_pascal.sbatch` and submit with `sbatch train_tinyllama_pascal.sbatch`.

```bash
#!/bin/bash
#SBATCH -p pascal
#SBATCH -J mlora-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o logs/train-%j.out
#SBATCH -e logs/train-%j.err
set -euxo pipefail

export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1
export HF_HOME=${HF_HOME:-$PWD/.hf_pascal/${SLURM_JOB_ID:-$$}}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE" logs

BASE_MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATA_PATH="$SLURM_SUBMIT_DIR/commonsense_170k_taskid.json"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/tinyllama-mlora"
DEEPSPEED_CONFIG="config/ds2.json"

srun python mlora_finetune.py \
  --base_model "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --adapter_name mlora \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --weight_decay 0.0 \
  --cutoff_len 512 \
  --save_step 1000 \
  --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]' \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lambda_num 8 \
  --num_B 3 \
  --temperature 0.1 \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --dtype float32
```

Notes:
- Batch size 1 with grad-accum 8 keeps VRAM within 1080 Ti limits while matching the paper’s effective batch size.
- `--dtype float32` ensures stable training on Pascal (bf16 is unsupported).
- If you resume training, set `--resume_from_checkpoint` to the latest checkpoint path.

## 3) Full evaluation on all 8 MC tasks
Copy-paste to `eval_full_pascal.sbatch` and submit with `sbatch eval_full_pascal.sbatch` after training. Assumes the latest checkpoint in `$OUTPUT_DIR` (adjust if you want a fixed checkpoint).

```bash
#!/bin/bash
#SBATCH -p pascal
#SBATCH -J mlora-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o logs/eval-%j.out
#SBATCH -e logs/eval-%j.err
set -euxo pipefail

export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1
export HF_HOME=${HF_HOME:-$PWD/.hf_pascal/${SLURM_JOB_ID:-$$}}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE" logs

BASE_MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/tinyllama-mlora"
LORA_WEIGHTS=$(ls -t $OUTPUT_DIR/*.bin | head -n1)
TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj"]'

DATASETS=(boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa)

for DATASET in "${DATASETS[@]}"; do
  srun --ntasks=1 --gres=gpu:1 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset "$DATASET" \
    --base_model "$BASE_MODEL" \
    --lora_target_modules "$TARGET_MODULES" \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights "$LORA_WEIGHTS" \
    --dtype float32
  echo "Finished $DATASET"
done
```

Notes:
- Uses the newest `.bin` checkpoint in `runs/tinyllama-mlora`; set `LORA_WEIGHTS` manually to evaluate a specific checkpoint.
- Keep `--dtype float32` to avoid precision issues on Pascal.

## 4) “Eval latest” during training
Copy-paste to `eval_latest_pascal.sbatch` and submit while training is running. It polls for the newest checkpoint and evaluates just one task (change `DATASET` as needed).

```bash
#!/bin/bash
#SBATCH -p pascal
#SBATCH -J mlora-eval-latest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o logs/eval-latest-%j.out
#SBATCH -e logs/eval-latest-%j.err
set -euxo pipefail

export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1
export HF_HOME=${HF_HOME:-$PWD/.hf_pascal/${SLURM_JOB_ID:-$$}}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE" logs

BASE_MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/tinyllama-mlora"
TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj"]'
DATASET="boolq"

latest_ckpt() {
  ls -t $OUTPUT_DIR/*.bin 2>/dev/null | head -n1
}

LORA_WEIGHTS=$(latest_ckpt)
if [ -z "$LORA_WEIGHTS" ]; then
  echo "No checkpoint found in $OUTPUT_DIR" >&2
  exit 1
fi

srun python mlora_evaluate.py \
  --model LLaMA-7B \
  --adapter mlora \
  --dataset "$DATASET" \
  --base_model "$BASE_MODEL" \
  --lora_target_modules "$TARGET_MODULES" \
  --batch_size 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lambda_num 8 \
  --num_B 3 \
  --temperature 0.1 \
  --lora_weights "$LORA_WEIGHTS" \
  --dtype float32
```

## 5) Quick checklist
- Submit Slurm jobs from the repo root (`MTL-LoRA`).
- Ensure `commonsense_170k_taskid.json` is present at repo root; adjust `DATA_PATH` if stored elsewhere.
- The provided settings match the paper’s mLoRA hyperparameters and are sized for Pascal GPUs.
- For other base models, only change `BASE_MODEL` and (if needed) `TARGET_MODULES` to match that architecture.
```
