#!/usr/bin/env python3
"""Train RoBERTa + (MTL-)mLoRA on multi-task GLUE (SINGLE GPU).

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
    print(f"[DEBUG] fine_tune_client: Extracted {len(client_lora_weights)} LoRA weight matrices")
    for name, tensor in list(client_lora_weights.items())[:3]:  # Print first 3 for brevity
        print(f"  {name}: shape={tensor.shape}")
    return client_lora_weights

# Federated Averaging function for LoRA weights
def fed_avg(client_weights):
    print(f"[DEBUG] fed_avg: Averaging {len(client_weights)} client weight sets")
    avg_weights = copy.deepcopy(client_weights[0])
    print(f"[DEBUG] fed_avg: Total parameters to average: {len(avg_weights)}")
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = avg_weights[key] / len(client_weights)
    print(f"[DEBUG] fed_avg: Averaged weights (first 3 for brevity)")
    for name, tensor in list(avg_weights.items())[:3]:
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    return avg_weights

def stack_A(client_A, client_p, hidden, lora_r):
    device = next(iter(client_A[0].values())).device
    num_clients = len(client_A)
    
    stacked = dict()
    for layer in client_A[0]:
        stacked[layer] = torch.cat([client_p[i]*client_A[i][layer] for i in range(num_clients)], dim=1).to(device) # stack As along lora_r for each layer

    assert next(iter(stacked.values())).shape==torch.Size([1, lora_r, hidden]), f"As stacked incorrectly: {next(iter(stacked.values())).shape}"
    return stacked

def stack_B(client_B, num_B, hidden, lora_r):
    device = next(iter(client_B[0].values())).device
    num_clients = len(client_B)
    stacked = dict() #dict.fromkeys(client_B[0], torch.zeros([num_B, hidden, lora_r]))
    for layer in client_B[0]:
        stacked[layer] = torch.cat([client_B[i][layer] for i in range(num_clients)], dim=2).to(device) # stack Bs along lora_r for each layer
    # TODO testing
    assert next(iter(stacked.values())).shape==torch.Size([num_B, hidden, lora_r]), "Bs stacked incorrectly"
    return stacked


def stack_lambdas(client_lambdas, num_tasks, lora_r):
    device = next(iter(client_lambdas[0].values())).device
    dtype = next(iter(client_lambdas[0].values())).dtype
    num_clients = len(client_lambdas)
    stacked = dict.fromkeys(client_lambdas[0], torch.zeros([num_tasks, lora_r, lora_r], dtype=dtype))

    for layer in client_lambdas[0]:
        lambdas = [client_lambdas[i][layer] for i in range(num_clients)]
        sizes = [l.shape[1] for l in lambdas] # accounting for heterogeneous lora ranks
        offset = 0
        for l, r in zip(lambdas, sizes): # stack lambdas diagonally
            stacked[layer][:, offset:offset+r, offset:offset+r] = l
            offset += r

    return stacked

def avg_B_w(client_B_w, num_tasks, num_B):
    num_clients = len(client_B_w)
    avg = copy.deepcopy(client_B_w[0])

    for layer in client_B_w[0]:
        for i in range(1, num_clients):
            avg[layer] += client_B_w[i][layer]
        avg[layer] = avg[layer] / num_clients

    # Debug: check for any zero or problematic values
    for name, tensor in list(avg.items())[:2]:
        print(f"  {name}: shape={tensor.shape}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}, contains_nan={torch.isnan(tensor).any()}")
    
    return avg

def aggregate_mtl_weights(client_weights, client_p, hidden=768, num_B=3, num_tasks=2, lora_r=8):
    client_A = []
    client_B = []
    client_lambdas = []
    client_B_w = []

    for weights in client_weights:
        client_A.append({k: v for k, v in weights.items() if k.endswith("lora_A")})
        client_B.append({k: v for k, v in weights.items() if "lora_B" in k and not k.endswith("lora_B_w")})
        client_lambdas.append({k: v for k, v in weights.items() if k.endswith("lora_lambdas")})
        client_B_w.append({k: v for k, v in weights.items() if k.endswith("lora_B_w")})
    
    print(f"[DEBUG] aggregate_mtl_weights: Found {len(client_A[0])} lora_A, {len(client_B[0])} lora_B, {len(client_lambdas[0])} lora_lambdas, {len(client_B_w[0])} lora_B_w in first client")
    
    '''
    # Sample parameter names for debugging
    if client_A[0]:
        print(f"[DEBUG] Sample lora_A names: {list(client_A[0].keys())[:2]}")
    if client_B[0]:
        print(f"[DEBUG] Sample lora_B names: {list(client_B[0].keys())[:2]}")
    if client_lambdas[0]:
        print(f"[DEBUG] Sample lora_lambdas names: {list(client_lambdas[0].keys())[:2]}")
    if client_B_w[0]:
        print(f"[DEBUG] Sample lora_B_w names: {list(client_B_w[0].keys())[:2]}")
    '''

    a_stacked = stack_A(client_A, client_p, hidden, lora_r)
    b_stacked = stack_B(client_B, num_B, hidden, lora_r)
    lambdas_stacked = stack_lambdas(client_lambdas, num_tasks, lora_r)
    b_w_avg = avg_B_w(client_B_w, num_tasks, num_B)

    agg_weights = {**a_stacked, **b_stacked, **lambdas_stacked, **b_w_avg}
    print(f"[DEBUG] aggregate_mtl_weights: Created {len(agg_weights)} aggregated weights")
    
    return agg_weights

# Apply aggregated weights to global model
def update_global_model(global_model, avg_weights):
    updated_count = 0
    shape_mismatches = []
    
    print(f"[DEBUG] update_global_model: Global model has {len(list(global_model.named_parameters()))} parameters")
    print(f"[DEBUG] update_global_model: avg_weights has {len(avg_weights)} entries")
    
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
    
    print(f"[DEBUG] update_global_model: Updated {updated_count} parameters in global model")
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


def merge_and_expand_lora(global_model, aggregated_weights, old_r, new_r):
    """
    Merge aggregated weights into global model while expanding LoRA rank.
    
    Instead of just copying aggregated weights, this:
    1. Takes the global model's current LoRA weights
    2. Pads them with zeros to the new rank
    3. Merges in the aggregated weights from all clients
    
    This maintains structural coherence during rank expansion.
    
    Args:  
        global_model: Current global model with old rank
        aggregated_weights: Stacked/aggregated weights from clients
        old_r: Current LoRA rank
        new_r: New (expanded) LoRA rank
        
    Returns:
        Dict of merged weights ready to use in new model
    """
    old_state = global_model.state_dict()
    merged = {}
    
    device = next(iter(aggregated_weights.values())).device
    dtype = next(iter(aggregated_weights.values())).dtype
    
    for name, tensor in aggregated_weights.items():
        if "lora_A" in name:
            # For A: (1, new_r, in_feat) - aggregated already has new_r
            # Use the aggregated version (stacked from clients)
            merged[name] = tensor.to(device=device, dtype=dtype)
            print(f"[DEBUG] Using stacked lora_A: {name} shape={tensor.shape}")
            
        elif "lora_B" in name and "lora_B_w" not in name:
            # For B: (num_B, out_feat, new_r) - aggregated already has new_r
            # Use the aggregated version (stacked from clients)
            merged[name] = tensor.to(device=device, dtype=dtype)
            print(f"[DEBUG] Using stacked lora_B: {name} shape={tensor.shape}")
            
        elif "lora_lambdas" in name:
            # For lambdas: (num_tasks, new_r, new_r) - aggregated has diagonally stacked version
            # The aggregated version places client lambdas diagonally
            # Use as-is since this preserves per-client task-specific decisions
            merged[name] = tensor.to(device=device, dtype=dtype)
            print(f"[DEBUG] Using stacked lora_lambdas: {name} shape={tensor.shape}")
            
        else:
            # For other parameters like lora_B_w: use as-is
            merged[name] = tensor.to(device=device, dtype=dtype)
    
    return merged


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
    round_str = f" in FL round {round_num}" if round_num is not None else ""
    print(f"[DEBUG] Transferred {params_transferred} non-LoRA parameters{round_str}")
    
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
    cast_trainable_params_to_fp32(global_model)

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
        for client_id in range(args.num_clients):
            print(f"[INFO] Fine-tuning client {client_id+1}/{args.num_clients}")
            client_model = copy.deepcopy(global_model)
            client_data = task_data[client_id]
            client_lora_weights = fine_tune_client(client_model, client_data, device, args, use_amp)
            print(f"[DEBUG] FL round {round+1}: Client {client_id+1} returned {len(client_lora_weights)} LoRA matrices")
            for name, tensor in list(client_lora_weights.items())[:2]:
                print(f"  {name}: shape={tensor.shape}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}")
            client_weights.append(client_lora_weights)
        
        # Aggregate weights
        print(f"[DEBUG] FL round {round+1}: Aggregating weights from {len(client_weights)} clients")
    
        #avg_weights = dict()
        avg_weights = fed_avg(client_weights)
        '''
        if args.strat == "fedit":
            avg_weights = fed_avg(client_weights)
        elif args.strat == "centralized":
            avg_weights = client_weights
        else: # default to FLoRA
            flora_r *= args.num_clients
            avg_weights = aggregate_mtl_weights(
                client_weights, 
                client_p=[1.0/args.num_clients] * args.num_clients,
                hidden=768,  # RoBERTa-base hidden size
                num_B=args.num_B,
                num_tasks=len(GLUE_TASKS),
                lora_r=flora_r 
            )
        '''
        '''
        
        # Debug: Check aggregated weights
        print(f"[DEBUG] FL round {round+1}: Aggregated weights summary:")
        agg_lora_a_shapes = []
        agg_lora_b_shapes = []
        for name, tensor in avg_weights.items():
            if name.endswith("lora_A"):
                agg_lora_a_shapes.append(tensor.shape)
            elif "lora_B" in name and not name.endswith("lora_B_w"):
                agg_lora_b_shapes.append(tensor.shape)
        
        if agg_lora_a_shapes:
            print(f"  lora_A shape samples: {agg_lora_a_shapes[:2]}, expected: (1, {flora_r}, 768)")
        if agg_lora_b_shapes:
            print(f"  lora_B shape samples: {agg_lora_b_shapes[:2]}, expected: (3, 768, {flora_r}) or similar")
        '''
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
        print(f"[DEBUG] FL round {round+1}: Updating global model with aggregated LoRA weights")
        update_global_model(new_global_model, avg_weights)
        
        # Ensure trainable params are set correctly for the new model
        set_trainable_params(new_global_model, train_bias=train_bias, train_layernorm=train_ln)
        cast_trainable_params_to_fp32(new_global_model)
        
        # Debug: Verify updated weights
        print(f"[DEBUG] FL round {round+1}: Updated model weights stats:")
        for name, param in list(new_global_model.named_parameters()):
            if 'lora' in name and 'lora_A' in name:
                print(f"  {name}: shape={param.shape}, min={param.min():.6f}, max={param.max():.6f}, mean={param.mean():.6f}")
                break
        
        # Replace global model reference and update current rank
        global_model = new_global_model
        print(f"[INFO] Completed FL round {round+1}/{args.num_fl_rounds}")
    
    # Save run config
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
