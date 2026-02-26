#!/usr/bin/env python3
"""Train TinyLlama + (MTL-)mLoRA on multi-task GLUE (single GPU).

This script mirrors the functionality of
``run_glue_roberta_mtl_mlora_train_single_gpu.py`` but targets
TinyLlama instead of RoBERTa.  It supports federated LoRA
aggregation (FLoRA) as well as classic federated averaging.  The
defaults are tuned for a single 1080 Ti GPU with limited VRAM, so
batch sizes and LoRA rank are smaller than in the RoBERTa run
script.  Adjust these values as needed for your hardware.

Outputs (in ``--output_dir``) include:

* ``checkpoints/ckpt_*.pt`` — full model checkpoints
* ``adapter_state*.pt`` — trainable encoder params (LoRA + optional bias/LN)
* ``heads_state*.pt`` — per-task classification heads
* ``eval_latest.json`` / ``eval_epoch_*.json`` — evaluation results
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

# Suppress tokenization warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Import generic utilities from the RoBERTa implementation; these
# functions operate on arbitrary models and are reused here.
from src.roberta_glue_mtl_mlora.data import build_dataloaders
# Note: ``aggregate_mtl_weights``, ``update_global_model`` and
# ``transfer_non_lora_params`` are defined below.  They were
# originally defined in the RoBERTa training script and are
# duplicated here because they are not exported from
# ``src.roberta_glue_mtl_mlora.fed_utils``.
from src.roberta_glue_mtl_mlora.hf_utils import default_hf_home, get_hf_token
from src.roberta_glue_mtl_mlora.train_loop import train
# Functions to save checkpoints and adapter/head weights.  We import these
# here rather than in fine_tune_client so they are available for saving the
# final aggregated model after the FL loop.
from src.roberta_glue_mtl_mlora.checkpoint import (
    save_checkpoint,
    save_adapter_and_heads,
)
from src.roberta_glue_mtl_mlora.utils import set_seed

from tinyllama_glue_mtl_mlora.constants import GLUE_TASKS
from tinyllama_glue_mtl_mlora.factory import create_model, create_tokenizer
from tinyllama_glue_mtl_mlora.model import (
    cast_trainable_params_to_fp32,
    set_trainable_params,
)

# -----------------------------------------------------------------------------
# Federated Averaging helper
#
# The RoBERTa training script defines fed_avg inline, but it is not exposed
# via ``src.roberta_glue_mtl_mlora.fed_utils``.  We therefore reimplement it
# here to support the ``fedit`` strategy or simple averaging across client
# LoRA weights.  Given a list of per-client weight dictionaries, it returns
# a new dictionary whose tensors are the elementwise average.  This function
# assumes that all clients provide the same keys and that tensors are on
# CPU; adjust as needed for your use case.
def fed_avg(client_weights):
    """Average LoRA weight dictionaries across clients (simple mean)."""
    if not client_weights:
        raise ValueError("fed_avg: No client weights provided")
    num_clients = len(client_weights)
    # Deep copy first client's weights to avoid modifying input
    avg = {k: v.clone() for k, v in client_weights[0].items()}
    for key in avg.keys():
        for i in range(1, num_clients):
            avg[key] += client_weights[i][key]
        avg[key] = avg[key] / num_clients
    return avg


# -----------------------------------------------------------------------------
# Federated stacking and averaging functions copied from the RoBERTa training
# script.  These functions combine the LoRA weights from multiple clients
# (when using the FLoRA strategy) by stacking the low-rank matrices along
# their rank dimension and averaging the task-specific B_w matrices.  They
# accept lists of per-client weight dictionaries and return a single
# aggregated dictionary.  See the original RoBERTa script for detailed
# commentary on the expected shapes.

def stack_A(client_A: list, client_p: list, hidden: int, lora_r: int) -> Dict[str, torch.Tensor]:
    """Stack A matrices from clients along the LoRA rank dimension."""
    device = next(iter(client_A[0].values())).device
    num_clients = len(client_A)
    stacked = {}
    for layer in client_A[0]:
        stacked[layer] = torch.cat(
            [client_p[i] * client_A[i][layer] for i in range(num_clients)], dim=1
        ).to(device)
    return stacked


def stack_B(client_B: list, num_B: int, hidden: int, lora_r: int) -> Dict[str, torch.Tensor]:
    """Stack B matrices from clients along the LoRA rank dimension."""
    device = next(iter(client_B[0].values())).device
    num_clients = len(client_B)
    stacked = {}
    for layer in client_B[0]:
        stacked[layer] = torch.cat(
            [client_B[i][layer] for i in range(num_clients)], dim=2
        ).to(device)
    return stacked


def stack_lambdas(client_lambdas: list, num_tasks: int, lora_r: int) -> Dict[str, torch.Tensor]:
    """Stack Lambda matrices from clients into a block-diagonal tensor."""
    device = next(iter(client_lambdas[0].values())).device
    dtype = next(iter(client_lambdas[0].values())).dtype
    num_clients = len(client_lambdas)
    stacked = {
        key: torch.zeros((num_tasks, lora_r, lora_r), dtype=dtype, device=device)
        for key in client_lambdas[0]
    }
    for layer in client_lambdas[0]:
        lambdas = [client_lambdas[i][layer] for i in range(num_clients)]
        sizes = [l.shape[1] for l in lambdas]
        offset = 0
        for lam, r in zip(lambdas, sizes):
            stacked[layer][:, offset:offset + r, offset:offset + r] = lam.to(device)
            offset += r
    return stacked


def avg_B_w(client_B_w: list, num_tasks: int, num_B: int) -> Dict[str, torch.Tensor]:
    """Average B_w matrices across clients (simple mean)."""
    num_clients = len(client_B_w)
    avg = {k: v.clone() for k, v in client_B_w[0].items()}
    for layer in avg:
        for i in range(1, num_clients):
            avg[layer] += client_B_w[i][layer]
        avg[layer] = avg[layer] / num_clients
    return avg


def aggregate_mtl_weights(
    client_weights: list,
    client_p: list,
    hidden: int,
    num_B: int,
    num_tasks: int,
    lora_r: int,
) -> Dict[str, torch.Tensor]:
    """Aggregate per-client LoRA weights for FLoRA by stacking and averaging."""
    # Separate LoRA parameters into A, B, lambdas and B_w blocks
    client_A = []
    client_B = []
    client_lambdas = []
    client_B_w = []
    for weights in client_weights:
        client_A.append({k: v for k, v in weights.items() if k.endswith("lora_A")})
        client_B.append({k: v for k, v in weights.items() if "lora_B" in k and not k.endswith("lora_B_w")})
        client_lambdas.append({k: v for k, v in weights.items() if k.endswith("lora_lambdas")})
        client_B_w.append({k: v for k, v in weights.items() if k.endswith("lora_B_w")})

    a_stacked = stack_A(client_A, client_p, hidden, lora_r)
    b_stacked = stack_B(client_B, num_B, hidden, lora_r)
    lambdas_stacked = stack_lambdas(client_lambdas, num_tasks, lora_r)
    b_w_avg = avg_B_w(client_B_w, num_tasks, num_B)
    # Merge into a single dictionary
    agg_weights = {**a_stacked, **b_stacked, **lambdas_stacked, **b_w_avg}
    return agg_weights


def update_global_model(global_model: torch.nn.Module, avg_weights: Dict[str, torch.Tensor]) -> None:
    """Copy aggregated LoRA weights into the global model (in-place)."""
    updated_count = 0
    shape_mismatches = []
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in avg_weights:
                weight = avg_weights[name]
                if param.shape != weight.shape:
                    shape_mismatches.append((name, param.shape, weight.shape))
                else:
                    param.copy_(weight.to(param.device))
                    updated_count += 1
    if shape_mismatches:
        for name, s_old, s_new in shape_mismatches:
            print(f"[WARNING] Shape mismatch for {name}: model={s_old}, weights={s_new}")


def transfer_non_lora_params(
    old_model: torch.nn.Module,
    new_model: torch.nn.Module,
    round_num: int | None = None,
) -> int:
    """Transfer non-LoRA parameters from the old model to the new model."""
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    params_transferred = 0
    for name, param in old_state_dict.items():
        if "lora" not in name and name in new_state_dict:
            if new_state_dict[name].shape == param.shape:
                new_state_dict[name].copy_(param)
                params_transferred += 1
            else:
                round_str = f" (FL round {round_num})" if round_num is not None else ""
                print(
                    f"[WARNING] Shape mismatch for {name}{round_str}: old={param.shape}, new={new_state_dict[name].shape}"
                )
    new_model.load_state_dict(new_state_dict)
    return params_transferred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TinyLlama + mLoRA on multi-task GLUE (single GPU)")

    # Model / output
    p.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Name of the TinyLlama model to fine-tune",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_tinyllama_glue_mlora_sgpu",
        help="Directory to store checkpoints and evaluation results",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training hyperparameters (defaults tuned for 1080 Ti)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    # Mixed precision
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Enable CUDA AMP (fp16 autocast + GradScaler)",
    )

    # LoRA / mLoRA hyperparams
    p.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (scaling)")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--num_B", type=int, default=2, help="Number of B matrices in mLoRA")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature scaling factor for mLoRA B matrices",
    )

    # Which additional params to train
    p.add_argument(
        "--freeze_bias",
        action="store_true",
        help="Keep bias parameters frozen.  By default biases are trained.",
    )
    p.add_argument(
        "--freeze_layernorm",
        action="store_true",
        help="Keep LayerNorm parameters frozen.  By default they are trained.",
    )

    # Checkpointing
    p.add_argument(
        "--save_steps",
        type=int,
        default=2500,
        help="Save training checkpoint every N update steps.",
    )
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--save_pre_eval_ckpt", action="store_true")
    p.add_argument("--resume_from_ckpt", type=str, default=None)

    # HF / dataset cache
    p.add_argument("--offline", action="store_true", help="Use offline (local) HF caches only")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--glue_disk_cache_dir", type=str, default=None)
    p.add_argument("--hf_datasets_cache_dir", type=str, default=None)

    # Eval details dump
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after each epoch",
    )
    p.add_argument(
        "--save_eval_details",
        action="store_true",
        help="Write per-example JSONL with evaluation details",
    )
    p.add_argument(
        "--eval_details_max_examples",
        type=int,
        default=200,
        help="Max examples per split to include in eval details (use -1 for all)",
    )
    p.add_argument("--eval_details_only_errors", action="store_true")
    p.add_argument("--eval_details_topk", type=int, default=2)
    p.add_argument(
        "--stsb_abs_err_threshold",
        type=float,
        default=0.5,
        help="Absolute error threshold for STS-B regression during evaluation",
    )

    # Federated learning settings
    p.add_argument(
        "--num_fl_rounds",
        type=int,
        default=1,
        help="Number of federated learning rounds (1 disables FL)",
    )
    p.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="Number of clients to simulate (1 = no FL)",
    )
    p.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=1.0,
        help="Dirichlet alpha for non-IID data split (lower = more heterogeneous)",
    )
    p.add_argument(
        "--strat",
        type=str,
        default="FLoRA",
        help="Federated aggregation strategy: 'FLoRA'/'federated', 'fedit' or 'centralized'",
    )

    # NEW: client_p (manual client weighting)
    # - One value: applied to ALL clients (e.g. --client_p 1.0 => [1.0, 1.0] for 2 clients)
    # - N values: one per client (e.g. --client_p 0.6 0.4 with --num_clients 2)
    # If omitted: defaults to 1.0 for each client (so Python does NOT auto-compute 1/num_clients anymore).
    p.add_argument(
        "--client_p",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Client weight(s) p used during FLoRA aggregation. "
            "Provide one value (applies to all clients) or one per client. "
            "If omitted, defaults to 1.0 for each client."
        ),
    )

    # Test mode
    p.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode: load only a small subset of data per task",
    )

    return p.parse_args()


def fine_tune_client(
    model: torch.nn.Module,
    client_data: dict,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool = False,
) -> Dict[str, torch.Tensor]:
    """Fine-tune a model on a single client's data and return LoRA weights.

    This function simply delegates to ``train_loop.train`` to perform
    the optimization.  After training it extracts and returns all
    parameters whose names contain ``'lora'``.  These weights are used
    for federated averaging or FLoRA aggregation.
    """
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
    client_lora_weights = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora" in name
    }
    print(f"[DEBUG] fine_tune_client: Extracted {len(client_lora_weights)} LoRA weight matrices")
    return client_lora_weights


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # Seed for reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    if torch.cuda.is_available():
        print(f"[INFO] torch.cuda.current_device()={torch.cuda.current_device()}")
        print(f"[INFO] torch.cuda.get_device_name()={torch.cuda.get_device_name()}")
    else:
        print("[WARNING] CUDA not available - running on CPU")

    # HF configuration
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

    # Load tokenizer and model
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

    # Select trainable parameters according to flags
    train_bias = not args.freeze_bias
    train_ln = not args.freeze_layernorm
    set_trainable_params(global_model, train_bias=train_bias, train_layernorm=train_ln)
    cast_trainable_params_to_fp32(global_model)
    print(f"[INFO] train_bias={train_bias} train_layernorm={train_ln}")

    # Build data loaders.  We reuse the RoBERTa data loader implementation
    # because it is model-agnostic: it uses the provided tokenizer and
    # simply tokenizes the raw GLUE datasets.
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
        dirichlet_alpha=args.dirichlet_alpha,
        test_mode=args.test,
    )  # -> List[Dict[str, TaskData]]

    # Federated learning loop
    flora_r = args.lora_r  # Track current global model's LoRA rank
    for round_idx in range(args.num_fl_rounds):
        print(f"[INFO] Starting FL round {round_idx + 1}/{args.num_fl_rounds}")
        client_weights = []
        for client_id in range(args.num_clients):
            print(f"[INFO] Fine-tuning client {client_id + 1}/{args.num_clients}")
            # Copy the current global model to avoid parameter sharing
            client_model = copy.deepcopy(global_model)
            client_data_dict = task_data[client_id]
            client_lora_weights = fine_tune_client(client_model, client_data_dict, device, args, use_amp)
            client_weights.append(client_lora_weights)

        # Aggregate weights across clients
        print(f"[DEBUG] FL round {round_idx + 1}: Aggregating weights from {len(client_weights)} clients")
        if args.strat == "fedit":
            avg_weights = fed_avg(client_weights)
        elif args.strat == "centralized":
            # Centralized learning: no averaging; simply pick the first client
            avg_weights = client_weights[0]
        else:  # Default: FLoRA / federated stacking
            # Increase the LoRA rank by the number of clients for stacking
            flora_r *= args.num_clients
            hidden_dim = global_model.encoder.config.hidden_size

            # Resolve client_p from CLI:
            # - None -> default to 1.0 for each client (no auto 1/num_clients computation anymore)
            # - one value -> replicate for all clients
            # - num_clients values -> use as-is
            if args.client_p is None:
                client_p = [1.0] * args.num_clients
            else:
                if len(args.client_p) == 1:
                    client_p = [args.client_p[0]] * args.num_clients
                elif len(args.client_p) == args.num_clients:
                    client_p = list(args.client_p)
                else:
                    raise ValueError(
                        f"--client_p expects 1 value or exactly --num_clients values "
                        f"(num_clients={args.num_clients}), got {len(args.client_p)}: {args.client_p}"
                    )

            print(f"[INFO] Using client_p={client_p} for FLoRA aggregation")

            avg_weights = aggregate_mtl_weights(
                client_weights,
                client_p=client_p,
                hidden=hidden_dim,
                num_B=args.num_B,
                num_tasks=len(GLUE_TASKS),
                lora_r=flora_r,
            )

        # Create a new global model with potentially expanded rank
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
        # Transfer non-LoRA parameters from the old model to the new model
        transfer_non_lora_params(global_model, new_global_model, round_num=round_idx + 1)
        # Update the new model with the aggregated LoRA weights
        print(f"[DEBUG] FL round {round_idx + 1}: Updating global model with aggregated LoRA weights")
        update_global_model(new_global_model, avg_weights)
        # Reapply trainability settings
        set_trainable_params(new_global_model, train_bias=train_bias, train_layernorm=train_ln)
        cast_trainable_params_to_fp32(new_global_model)
        # Replace the global model reference
        global_model = new_global_model
        print(f"[INFO] Completed FL round {round_idx + 1}/{args.num_fl_rounds}")

    # -------------------------------------------------------------------------
    # Save final global model and adapter/head weights
    #
    # After completing all FL rounds, `global_model` holds the aggregated LoRA
    # weights with an expanded rank equal to `flora_r`.  By default the
    # training loop only saves checkpoints during local fine-tuning, which
    # means the final aggregated state is not persisted.  To enable true
    # evaluation of the global model, we explicitly save both a full
    # checkpoint (for potential resumption) and the lightweight adapter/head
    # weights.  These files are named with a "_global_final" suffix so
    # `resolve_load_paths()` can prefer them when loading via --load_dir.
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_ckpt_path = os.path.join(ckpt_dir, "ckpt_global_final.pt")
    # Save a full model checkpoint (optimizer and scheduler are None here)
    save_checkpoint(
        ckpt_path=final_ckpt_path,
        model=global_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=args.epochs,
        update_step=0,
        args=args,
    )
    # Save adapter and head states; this writes adapter_state.pt,
    # heads_state.pt, and also adapter_state_last.pt/heads_state_last.pt.
    save_adapter_and_heads(args.output_dir, global_model)
    # Copy to *_final.pt so resolve_load_paths() picks these when present.
    import shutil

    adapter_src = os.path.join(args.output_dir, "adapter_state.pt")
    heads_src = os.path.join(args.output_dir, "heads_state.pt")
    if os.path.exists(adapter_src):
        shutil.copy(adapter_src, os.path.join(args.output_dir, "adapter_state_final.pt"))
    if os.path.exists(heads_src):
        shutil.copy(heads_src, os.path.join(args.output_dir, "heads_state_final.pt"))

    # Save run configuration
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()