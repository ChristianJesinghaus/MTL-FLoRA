#!/usr/bin/env python3
"""Eval-only runner for RoBERTa + (MTL-)mLoRA on GLUE (SINGLE GPU).

Loads:
- Either a full training checkpoint (ckpt_*.pt)
- Or adapter_state*.pt + heads_state*.pt (saved by the training script)

Then runs evaluation on all GLUE tasks, and optionally writes per-example JSONL.

NOTE: adapter_state contains ALL trainable encoder params from training
(LoRA + biases + LayerNorm, unless you froze them).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import torch

from src.roberta_glue_mtl_mlora.checkpoint import load_adapter_and_heads, load_from_checkpoint, resolve_load_paths
from src.roberta_glue_mtl_mlora.data import build_dataloaders
from src.roberta_glue_mtl_mlora.eval_loop import evaluate
from src.roberta_glue_mtl_mlora.factory import create_model, create_tokenizer
from src.roberta_glue_mtl_mlora.hf_utils import default_hf_home, get_hf_token
from src.roberta_glue_mtl_mlora.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Model / output
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--output_dir", type=str, default=None, help="If unset, defaults to <load_dir>/eval_<timestamp>.")
    p.add_argument("--seed", type=int, default=42)

    # Loading
    p.add_argument("--load_dir", type=str, default=None)
    p.add_argument("--load_adapter", type=str, default=None)
    p.add_argument("--load_heads", type=str, default=None)
    p.add_argument("--load_ckpt", type=str, default=None)
    p.add_argument("--load_global_model", type=str, default=None, help="Path to global_model_final.pt from FL training")

    # Data / eval
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    # mLoRA hyperparams (MUST match training)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_B", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)

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

    if not (args.load_ckpt or args.load_dir or (args.load_adapter and args.load_heads) or args.load_global_model):
        raise ValueError("Provide --load_ckpt OR --load_dir OR (--load_adapter and --load_heads) OR --load_global_model.")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    # HF config
    hf_token = get_hf_token(args.hf_token)
    hf_home = default_hf_home()
    glue_disk_cache_dir = args.glue_disk_cache_dir or os.path.join(hf_home, "glue_disk_cache")
    hf_datasets_cache_dir = args.hf_datasets_cache_dir or os.environ.get("HF_DATASETS_CACHE") or None

    # Output dir
    if args.output_dir is None:
        if args.load_dir:
            args.output_dir = os.path.join(args.load_dir, "eval_only")
        else:
            args.output_dir = "./eval_only"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] device={device} use_amp={use_amp}")
    print(f"[INFO] output_dir={args.output_dir}")

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

    # Load weights
    if args.load_global_model:
        print(f"[LOAD] Loading full global model from: {args.load_global_model}")
        global_state_dict = torch.load(args.load_global_model, map_location="cpu")
        model.load_state_dict(global_state_dict, strict=False)
        print(f"[LOAD] Successfully loaded global model")
    elif args.load_ckpt:
        print(f"[CKPT] Loading from full checkpoint: {args.load_ckpt}")
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
        
        # Load adapter
        from src.roberta_glue_mtl_mlora.model import load_trainable_encoder_state
        adapter_state = torch.load(adapter_path, map_location="cpu")
        load_trainable_encoder_state(model, adapter_state)
        
        # Load heads, stripping "heads." prefix if present
        heads_state = torch.load(heads_path, map_location="cpu")
        corrected_heads_state = {}
        for k, v in heads_state.items():
            if k.startswith("heads."):
                corrected_heads_state[k.replace("heads.", "", 1)] = v
            else:
                corrected_heads_state[k] = v
        model.heads.load_state_dict(corrected_heads_state, strict=True)  # type: ignore[attr-defined]

    # Disable grads
    for p in model.parameters():
        p.requires_grad = False

    # Data
    task_data = build_dataloaders(
        tokenizer,
        max_length=args.max_length,
        train_batch_size=1,  # unused
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        hf_datasets_cache_dir=hf_datasets_cache_dir,
        glue_disk_cache_dir=glue_disk_cache_dir,
        hf_token=hf_token,
        offline=args.offline,
        save_eval_details=args.save_eval_details,
    )[0]

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
