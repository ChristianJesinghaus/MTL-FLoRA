#!/usr/bin/env python3
"""Train RoBERTa + (MTL-)mLoRA on multi-task GLUE with stacking Approach I / Ablation (SINGLE GPU).

This is the single-GPU refactor of the monolithic DDP script.
- no torch.distributed
- no DDP / torchrun
- keeps GLUE disk caching and the same evaluation prompts (unchanged)
- trains: LoRA params + task heads + (optionally) all bias + all LayerNorm

Outputs (in --output_dir):
- checkpoints/ckpt_*.pt
- adapter_state*.pt (trainable encoder params: LoRA (+bias/LN if enabled))
- heads_state*.pt
- eval_latest.json / eval_epoch_*.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import defaultdict
from typing import Dict

import logging

import torch

# Suppress RobertaTokenizerFast warning about using __call__ (it's already the preferred method)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.roberta_glue_mtl_mlora.data import build_dataloaders
from src.roberta_glue_mtl_mlora.factory import create_model, create_tokenizer
from src.roberta_glue_mtl_mlora.hf_utils import default_hf_home, get_hf_token
from src.roberta_glue_mtl_mlora.fed_utils import *
from src.roberta_glue_mtl_mlora.constants import GLUE_TASKS
from src.roberta_glue_mtl_mlora.eval_loop import evaluate


from src.roberta_glue_mtl_mlora.model import (
    cast_trainable_params_to_fp32,
  #  count_trainable_params,
    set_trainable_params,
)
from src.roberta_glue_mtl_mlora.train_loop import train
from src.roberta_glue_mtl_mlora.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Model / output
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--output_dir", type=str, default="./outputs_roberta_glue_mlora_sgpu")
    p.add_argument("--seed", type=int, default=42)

    # Training hyperparams (defaults tuned for 1080 Ti)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    # Mixed precision
    p.add_argument("--fp16", action="store_true", help="Enable CUDA AMP (fp16 autocast + GradScaler).")

    # LoRA / mLoRA hyperparams
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_B", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)

    # Which additional params to train
    p.add_argument(
        "--freeze_bias",
        action="store_true",
        help="By default, all bias params are TRAINED. Pass this flag to keep them frozen.",
    )
    p.add_argument(
        "--freeze_layernorm",
        action="store_true",
        help="By default, all LayerNorm params are TRAINED. Pass this flag to keep them frozen.",
    )

    # Checkpointing
    p.add_argument("--save_steps", type=int, default=2500, help="Save training checkpoint every N update steps.")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--save_pre_eval_ckpt", action="store_true")
    p.add_argument("--resume_from_ckpt", type=str, default=None)

    # HF / dataset cache
    p.add_argument("--offline", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--glue_disk_cache_dir", type=str, default=None)
    p.add_argument("--hf_datasets_cache_dir", type=str, default=None)

    # Eval details dump
    p.add_argument("--skip_eval", action="store_true", help="Skip evaluation after epochs.")
    p.add_argument("--save_eval_details", action="store_true", help="Write per-example JSONL.")
    p.add_argument("--eval_details_max_examples", type=int, default=200, help="Max examples per split. Use -1 for all.")
    p.add_argument("--eval_details_only_errors", action="store_true")
    p.add_argument("--eval_details_topk", type=int, default=2)
    p.add_argument("--stsb_abs_err_threshold", type=float, default=0.5)

    # Federated learning settings
    p.add_argument("--num_fl_rounds", type=int, default=1, help="Number of federated learning rounds.")
    p.add_argument("--num_clients", type=int, default=1, help="Number of clients to simulate (1 = no FL).")
    p.add_argument("--dirichlet_alpha", type=float, default=1.0, help="Dirichlet alpha for non-IID data split (lower = more heterogeneous).")
    p.add_argument("--strat", type=str, default="FLoRA", help="FL aggregation strategy. Default: FLoRA")

    # Test mode
    p.add_argument("--test", action="store_true", help="Enable test mode: only load 50 samples per client per task.")

    return p.parse_args()

def fine_tune_client(model: torch.nn.Module, client_data: dict, device: torch.device, args: argparse.Namespace, use_amp=False) -> Dict[str, torch.Tensor]:
    train(
    model=model,
    task_data=client_data,
    device=device,
    use_amp=use_amp,
    output_dir=args.output_dir,
    epochs=args.epochs,
    grad_accum_steps=args.grad_accum_steps,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    save_pre_eval_ckpt=args.save_pre_eval_ckpt,
    eval_every_epoch=(not args.skip_eval),
    save_eval_details=args.save_eval_details,
    eval_details_max_examples=args.eval_details_max_examples,
    eval_details_only_errors=args.eval_details_only_errors,
    eval_details_topk=args.eval_details_topk,
    stsb_abs_err_threshold=args.stsb_abs_err_threshold,
    resume_from_ckpt=args.resume_from_ckpt,
    args_for_ckpt=args,
    )
    client_lora_weights = {name: param.detach().cpu().clone() for name, param in model.named_parameters() if 'lora' in name}
    client_heads = {name: param.detach().cpu().clone() for name, param in model.named_parameters() if 'heads' in name}
    return client_lora_weights, client_heads

# Federated Averaging function for LoRA weights
def fed_avg(client_weights):
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = avg_weights[key] / len(client_weights)
    return avg_weights


# Apply aggregated weights to global model
def update_global_model(global_model, avg_weights):
    updated_count = 0
    shape_mismatches = []
    
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in avg_weights:
                weight = avg_weights[name]
                if param.shape != weight.shape:
                    shape_mismatches.append((name, param.shape, weight.shape))
                    print(f"[ERROR] Shape mismatch for {name}: model={param.shape}, weights={weight.shape}")
                else:
                    param.copy_(weight)
                    updated_count += 1
    
    if shape_mismatches:
        print(f"[ERROR] Found {len(shape_mismatches)} shape mismatches!")
    
    # Debug: print which parameters from avg_weights were NOT used
    updated_names = set()
    for name, _ in global_model.named_parameters():
        if name in avg_weights:
            updated_names.add(name)
    
    unused_weights = set(avg_weights.keys()) - updated_names
    if unused_weights:
        print(f"[WARNING] These aggregated weights were not used in model update:")
        for name in list(unused_weights)[:5]:
            print(f"  {name}: shape={avg_weights[name].shape}")

def check_bad_tensors(avg_weights):
    """
    Returns True if any tensor contains NaNs OR is all zeros.
    Otherwise returns False.
    """
    for name, t in avg_weights.items():
        if torch.isnan(t).any():
            print(f"[DEBUG] check_bad_tensors: NaNs found in {name}")
            return True
        if torch.count_nonzero(t) == 0:
            print(f"[DEBUG] check_bad_tensors: All zeros in {name}")
            return True
    return False


def transfer_non_lora_params(old_model, new_model, round_num=None):
    """
    Transfer non-LoRA parameters from old model to new model.
    Useful when increasing lora_r and recreating the model while preserving
    base encoder weights, task heads, bias, and LayerNorm parameters.
    
    Args:
        old_model: Model with old (smaller) lora_r
        new_model: Model with new (larger) lora_r
        round_num: Optional FL round number for debug logging
        
    Returns:
        int: Number of parameters transferred
    """
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    
    params_transferred = 0
    for name, param in old_state_dict.items():
        # Only transfer non-LoRA parameters (LoRA layers change shape and are updated separately)
        if 'lora' not in name and name in new_state_dict:
            if new_state_dict[name].shape == param.shape:
                new_state_dict[name].copy_(param)
                params_transferred += 1
            else:
                round_str = f" (FL round {round_num})" if round_num is not None else ""
                print(f"[WARNING] Shape mismatch for {name}{round_str}: old={param.shape}, new={new_state_dict[name].shape}")
    
    new_model.load_state_dict(new_state_dict)    
    return params_transferred


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    if torch.cuda.is_available():
        print(f"[INFO] torch.cuda.current_device()={torch.cuda.current_device()}")
        print(f"[INFO] torch.cuda.get_device_name()={torch.cuda.get_device_name()}")
    else:
        print(f"[WARNING] CUDA not available - running on CPU")
    
    # HF config
    hf_token = get_hf_token(args.hf_token)
    hf_home = default_hf_home()
    glue_disk_cache_dir = args.glue_disk_cache_dir or os.path.join(hf_home, "glue_disk_cache")
    hf_datasets_cache_dir = args.hf_datasets_cache_dir or os.environ.get("HF_DATASETS_CACHE") or None

    print(f"[INFO] device={device} use_amp={use_amp}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] HF_HOME={hf_home}")
    print(f"[INFO] glue_disk_cache_dir={glue_disk_cache_dir}")
    print(f"[INFO] hf_datasets_cache_dir={hf_datasets_cache_dir}")
    print(f"[INFO] hf_token_present={'yes' if hf_token else 'no'}")

    tokenizer = create_tokenizer(args.model_name, offline=args.offline)
    global_model = create_model(
        model_name=args.model_name,
        offline=args.offline,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_B=args.num_B,
        temperature=args.temperature,
    )

    # Trainable params selection
    train_bias = not args.freeze_bias
    train_ln = not args.freeze_layernorm
    set_trainable_params(global_model, train_bias=train_bias, train_layernorm=train_ln)
    #cast_trainable_params_to_fp32(global_model)

   # trainable, total, pct = count_trainable_params(model)
   # print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    print(f"[INFO] train_bias={train_bias} train_layernorm={train_ln}")

    # Data
    task_data = build_dataloaders(
        tokenizer,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        hf_datasets_cache_dir=hf_datasets_cache_dir,
        glue_disk_cache_dir=glue_disk_cache_dir,
        hf_token=hf_token,
        offline=args.offline,
        save_eval_details=args.save_eval_details,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha, # Lower = more non-IID (0.1 is very heterogeneous)
        test_mode=args.test,
    ) # -> List[Dict[str, TaskData]]

    # Train (optimizer/scheduler/scaler + resume logic live in train_loop.train)
    # FL training loop
    flora_r = args.lora_r  # Track current global model's LoRA rank
    for round in range(args.num_fl_rounds):
        print(f"[INFO] Starting FL round {round+1}/{args.num_fl_rounds}")
        client_weights = []
        client_heads = []
        for client_id in range(args.num_clients):
            print(f"[INFO] Fine-tuning client {client_id+1}/{args.num_clients}")
            client_model = copy.deepcopy(global_model)
            client_data = task_data[client_id]
            client_lora_weights, client_task_heads = fine_tune_client(client_model, client_data, device, args, use_amp)
            client_weights.append(client_lora_weights)
            client_heads.append(client_task_heads)
        
        # Aggregate weights
        print(f"[DEBUG] FL round {round+1}: Aggregating weights from {len(client_weights)} clients")
    
        avg_weights = dict()
        avg_heads = dict()

        if args.strat == "fedit":
            avg_weights = average_mtl_weights(client_weights)
            avg_heads = fed_avg(client_heads)
        elif args.strat == "centralized":
            avg_weights = client_weights[0]
            avg_heads = client_heads[0]
        else: # default to FLoRA
            flora_r *= args.num_clients
            avg_weights = aggregate_lora_parameters(client_weights)
            avg_heads = fed_avg(client_heads)

        # Create new global model with potentially expanded rank
        new_global_model = create_model(
            model_name=args.model_name,
            offline=args.offline,
            device=device,
            lora_r=flora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_B=args.num_B,
            temperature=args.temperature,
        )
        
        # Transfer non-LoRA parameters from old model to preserve training state
        transfer_non_lora_params(global_model, new_global_model, round_num=round+1)
        
        # Update global model with aggregated LoRA weights
        update_global_model(new_global_model, {**avg_weights, **avg_heads})
        
        # Ensure trainable params are set correctly for the new model
        set_trainable_params(new_global_model, train_bias=train_bias, train_layernorm=train_ln)
        cast_trainable_params_to_fp32(new_global_model)
        
        # Replace global model reference and update current rank
        global_model = new_global_model

        #evaluate global model (optional during training)
        results = evaluate(
            model=global_model,
            task_data=task_data[0],
            device=device,
            use_amp=use_amp,
            output_dir=args.output_dir,
            tag="eval_only",
            save_details=True,
            details_max_examples=200,
            details_only_errors=False,
            details_topk=2,
            stsb_abs_err_threshold=0.5,
        )

        print(f"[eval_only] results {round+1}/{args.num_fl_rounds}:")
        print(json.dumps(results, indent=2))
        with open(os.path.join(args.output_dir, f"eval_only_metrics{round+1}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"[INFO] Completed FL round {round+1}/{args.num_fl_rounds}")
    
    # Save global model after FL training
    print(f"[INFO] Saving global model after FL training")
    torch.save(global_model.state_dict(), os.path.join(args.output_dir, "global_model_final.pt"))
    
    # Save LoRA adapter weights separately
    adapter_state = {name: param.detach().cpu() for name, param in global_model.named_parameters() if 'lora' in name}
    torch.save(adapter_state, os.path.join(args.output_dir, "adapter_state_final.pt"))
    
    # Save task heads weights separately
    heads_state = {name: param.detach().cpu() for name, param in global_model.named_parameters() if 'heads' in name}
    torch.save(heads_state, os.path.join(args.output_dir, "heads_state_final.pt"))
    
    print(f"[INFO] Saved global model, adapter weights, and task heads to {args.output_dir}")
    
    # Save run config
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
