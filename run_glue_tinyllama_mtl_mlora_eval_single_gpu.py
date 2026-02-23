#!/usr/bin/env python3
"""Eval‑only runner for TinyLlama + (MTL-)mLoRA on GLUE (single GPU).

This script loads a fine‑tuned TinyLlama mLoRA model and evaluates it on
all GLUE tasks.  It can load either a full checkpoint
(`ckpt_*.pt`) or separate adapter/head state files.  The mLoRA
hyperparameters passed here must match those used during training.

This version includes a fix for the Softmax normalisation of the mLoRA B
weights: after loading the model, the `block_size` attribute is set on
each mLoRALinear layer so that the softmax is computed separately within
each client's block of B adapters. Without this fix, the softmax would
normalise over all concatenated B adapters and produce incorrect
logits on aggregated models.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import torch

from src.roberta_glue_mtl_mlora.checkpoint import (
    load_adapter_and_heads,
    load_from_checkpoint,
    resolve_load_paths,
)
from src.roberta_glue_mtl_mlora.data import build_dataloaders
from src.roberta_glue_mtl_mlora.eval_loop import evaluate
from src.roberta_glue_mtl_mlora.hf_utils import default_hf_home, get_hf_token
from src.roberta_glue_mtl_mlora.utils import set_seed

from tinyllama_glue_mtl_mlora.factory import create_model, create_tokenizer

# Import the mLoRALinear class so we can set the block_size on each layer.
# This ensures that the softmax over the B weight vector is computed
# within each client's block of adapters rather than over all B adapters.
try:
    # type: ignore because the module may not declare typing information
    from src.adapter.mlora import mLoRALinear as _mLoRALinear  # type: ignore
except Exception:
    _mLoRALinear = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TinyLlama + mLoRA on GLUE (single GPU)")

    # Model / output
    p.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Pretrained TinyLlama model name",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If unset, defaults to <load_dir>/eval_only or ./eval_only",
    )
    p.add_argument("--seed", type=int, default=42)

    # Loading
    p.add_argument("--load_dir", type=str, default=None, help="Directory containing adapter_state.pt and heads_state.pt")
    p.add_argument("--load_adapter", type=str, default=None)
    p.add_argument("--load_heads", type=str, default=None)
    p.add_argument("--load_ckpt", type=str, default=None)

    # Data / eval
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    # mLoRA hyperparams (must match training)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # ``num_B`` is the number of B matrices *per client* (local block size).
    # For aggregated models this may differ from the global number of B matrices.
    p.add_argument("--num_B", type=int, default=2, help="Local number of B matrices per client")
    p.add_argument("--temperature", type=float, default=0.1)
    # ``global_num_B`` allows overriding the number of B matrices when creating
    # the model, useful for evaluating aggregated models.  If unset, the
    # model uses ``--num_B`` as both global and local counts.
    p.add_argument(
        "--global_num_B",
        type=int,
        default=None,
        help="Global number of B matrices for the aggregated model; overrides num_B",
    )
    # ``block_size`` explicitly sets the block size for softmax normalisation over
    # the B weight vector.  If unset, defaults to ``--num_B`` (local block size).
    p.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for mLoRA softmax; defaults to num_B if unset",
    )

    # Mixed precision
    p.add_argument("--fp16", action="store_true")

    # HF / dataset cache
    p.add_argument("--offline", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--glue_disk_cache_dir", type=str, default=None)
    p.add_argument("--hf_datasets_cache_dir", type=str, default=None)

    # Eval details
    p.add_argument("--save_eval_details", action="store_true")
    p.add_argument("--eval_details_max_examples", type=int, default=200)
    p.add_argument("--eval_details_only_errors", action="store_true")
    p.add_argument("--eval_details_topk", type=int, default=2)
    p.add_argument("--stsb_abs_err_threshold", type=float, default=0.5)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not (args.load_ckpt or args.load_dir or (args.load_adapter and args.load_heads)):
        raise ValueError("Provide --load_ckpt OR --load_dir OR (--load_adapter and --load_heads).")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    # HF configuration
    hf_token = get_hf_token(args.hf_token)
    hf_home = default_hf_home()
    glue_disk_cache_dir = args.glue_disk_cache_dir or os.path.join(hf_home, "glue_disk_cache")
    hf_datasets_cache_dir = args.hf_datasets_cache_dir or os.environ.get("HF_DATASETS_CACHE") or None

    # Determine output directory
    if args.output_dir is None:
        if args.load_dir:
            args.output_dir = os.path.join(args.load_dir, "eval_only")
        else:
            args.output_dir = "./eval_only"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] device={device} use_amp={use_amp}")
    print(f"[INFO] output_dir={args.output_dir}")

    tokenizer = create_tokenizer(args.model_name, offline=args.offline)
    # Determine the number of B matrices to use when constructing the model.  For
    # aggregated models this may differ from the local block size passed via
    # ``--num_B``.  If ``--global_num_B`` is provided, it overrides the
    # ``num_B`` argument when creating the model.
    num_B_model = args.global_num_B if args.global_num_B is not None else args.num_B
    model = create_model(
        model_name=args.model_name,
        offline=args.offline,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_B=num_B_model,
        temperature=args.temperature,
    )

    # Load weights (either from full checkpoint or from adapter/head state files)
    if args.load_ckpt:
        print(f"[CKPT] Loading full checkpoint from {args.load_ckpt}")
        load_from_checkpoint(args.load_ckpt, model=model, optimizer=None, scheduler=None, scaler=None, strict_heads=True)
    else:
        if args.load_dir:
            adapter_path, heads_path = resolve_load_paths(args.load_dir)
        else:
            adapter_path, heads_path = args.load_adapter, args.load_heads
        if adapter_path is None or heads_path is None:
            raise FileNotFoundError("Could not resolve adapter/heads paths.")
        print(f"[LOAD] adapter={adapter_path}")
        print(f"[LOAD] heads={heads_path}")
        load_adapter_and_heads(model, adapter_path=adapter_path, heads_path=heads_path, strict_heads=True)

    # Set block_size on each mLoRALinear layer so softmax is applied per client block
    if _mLoRALinear is not None:
        for module in model.modules():
            if isinstance(module, _mLoRALinear):
                # If --block_size is provided, use it; otherwise default to args.num_B.
                # This ensures that for aggregated models the softmax normalisation
                # operates over the original local block size rather than the global
                # number of B matrices.
                bs = args.block_size if args.block_size is not None else args.num_B
                module.block_size = bs

    # Disable gradients for evaluation
    for p in model.parameters():
        p.requires_grad = False

    # Build dataloaders. Note that build_dataloaders always returns a list of per‑client task dictionaries,
    # even when num_clients=1. Unwrap the single element for evaluation so that `task_data` is indexed by task name.
    task_data = build_dataloaders(
        tokenizer,
        max_length=args.max_length,
        train_batch_size=1,  # Unused in eval
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        hf_datasets_cache_dir=hf_datasets_cache_dir,
        glue_disk_cache_dir=glue_disk_cache_dir,
        hf_token=hf_token,
        offline=args.offline,
        save_eval_details=args.save_eval_details,
    )

    # Unwrap the list if there is exactly one element (single‑client setup).
    if isinstance(task_data, list):
        if len(task_data) != 1:
            raise ValueError(
                f"Expected a single client in eval; got {len(task_data)} clients."
            )
        task_data = task_data[0]

    # Run evaluation
    results = evaluate(
        model=model,
        task_data=task_data,
        device=device,
        use_amp=use_amp,
        output_dir=args.output_dir,
        tag="eval_only",
        save_details=args.save_eval_details,
        details_max_examples=args.eval_details_max_examples,
        details_only_errors=args.eval_details_only_errors,
        details_topk=args.eval_details_topk,
        stsb_abs_err_threshold=args.stsb_abs_err_threshold,
    )

    print("[eval_only] results:")
    print(json.dumps(results, indent=2))
    with open(os.path.join(args.output_dir, "eval_only_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()