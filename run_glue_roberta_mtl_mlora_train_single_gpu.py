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

import torch

from src.roberta_glue_mtl_mlora.data import build_dataloaders
from src.roberta_glue_mtl_mlora.factory import create_model, create_tokenizer
from src.roberta_glue_mtl_mlora.hf_utils import default_hf_home, get_hf_token
from src.roberta_glue_mtl_mlora.fed_utils import *

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

def stack_A(client_A, client_p, hidden, lora_r): # TODO Assertion fails: Check A shape
    device = next(iter(client_A[0].values())).device
    num_clients = len(client_A)
    #stacked = dict.fromkeys(client_A[0], torch.zeros([1, lora_r, hidden]))
    stacked = dict()
    for layer in client_A[0]:
        stacked[layer] = torch.cat([client_p[i]*client_A[i][layer] for i in range(num_clients)], dim=1).to(device) # stack As along lora_r for each layer
    assert next(iter(stacked.values())).shape==torch.Size([1, lora_r, hidden]), f"As stacked incorrectly: {next(iter(stacked.values())).shape}" # fails
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
    avg = copy.deepcopy(client_B_w[0])

    for layer in client_B_w[0]:
        for i in range(1, num_clients):
            avg[layer] += client_B_w[i][layer]
        avg[layer] = avg[layer] / num_clients

    return avg

def aggregate_mtl_weights(client_weights, client_p, hidden=768, num_B=3, num_tasks=2, lora_r=8):
    # TODO integrate & test
    client_A = []
    client_B = []
    client_lambdas = []
    client_B_w = []

    for weights in client_weights:
        client_A.append({k: v for k, v in weights.items() if k.endswith("lora_A")})
        client_B.append({k: v for k, v in weights.items() if k.endswith("lora_B")})
        client_lambdas.append({k: v for k, v in weights.items() if k.endswith("lora_lambdas")})
        client_B_w.append({k: v for k, v in weights.items() if k.endswith("lora_B_w")})
    
    print(f"[DEBUG] aggregate_mtl_weights: client_A[0]:{client_A[0]}")

    a_stacked = stack_A(client_A, client_p, hidden, lora_r)
    b_stacked = stack_B(client_B, num_B, hidden, lora_r)
    lambdas_stacked = stack_lambdas(client_lambdas, num_tasks, lora_r)
    b_w_avg = avg_B_w(client_B_w, nun_tasks, num_B)

    print(f"[DEBUG] aggregate_mtl_weights: a_stacked[0]:{a_stacked[0]}")


    agg_weights = {**a_stacked, **b_stacked, **lambdas_stacked, **b_w_avg}
    print(f"[DEBUG] aggregate_mtl_weights: agg_weights:{agg_weights}")

    return agg_weights

# Apply aggregated weights to global model
def update_global_model(global_model, avg_weights):
    updated_count = 0
        
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in avg_weights:
                param.copy_(avg_weights[name])
                updated_count += 1
    print(f"[DEBUG] update_global_model: Updated {updated_count} parameters in global model")

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


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

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
    model = create_model(
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
    set_trainable_params(model, train_bias=train_bias, train_layernorm=train_ln)
    cast_trainable_params_to_fp32(model)

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
    flora_r = args.lora_r
    for round in range(args.num_fl_rounds):
        flora_r *= args.num_clients
        print(f"[INFO] Starting FL round {round+1}/{args.num_fl_rounds}")
        client_weights = []
        for client_id in range(args.num_clients):
            print(f"[INFO] Fine-tuning client {client_id+1}/{args.num_clients}")
            client_model = copy.deepcopy(model)
            client_data = task_data[client_id]
            client_lora_weights = fine_tune_client(client_model, client_data, device, args, use_amp)
            print(f"[DEBUG] FL round {round+1}: Client {client_id+1} returned {len(client_lora_weights)} LoRA matrices")
            client_weights.append(client_lora_weights)
        # Aggregate weights
        print(f"[DEBUG] FL round {round+1}: Aggregating weights from {len(client_weights)} clients")
        #test = aggregate_mtl_weights(client_weights, 768, 8)
        #avg_weights = fed_avg(client_weights)
        avg_weights = aggregate_mtl_weights(client_weights, client_p=[0.5, 0.5], lora_r=flora_r)
        #avg_weights = aggregate_lora_parameters(client_weights, weights_dict={ "client_1": 0.5, "client_2": 0.5})
        # Update global model
        print(f"[DEBUG] FL round {round+1}: Updating global model with aggregated weights")
        model = create_model(
            model_name=args.model_name,
            offline=args.offline,
            device=device,
            lora_r=flora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_B=args.num_B,
            temperature=args.temperature,
        )
        update_global_model(model, avg_weights)

        assert check_bad_tensors(avg_weights) == False, "[DEBUG] Bad tensor found in avg_weights"
        print(f"[INFO] Completed FL round {round+1}/{args.num_fl_rounds}")

    # Save run config
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
