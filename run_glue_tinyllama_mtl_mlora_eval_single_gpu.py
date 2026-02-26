#!/usr/bin/env python3
"""Eval‑only runner for TinyLlama + (MTL-)mLoRA on GLUE (single GPU).

This script loads a fine‑tuned TinyLlama mLoRA model and evaluates it on
all GLUE tasks.

Key features (important for federated/FLoRA runs):
- You can point `--load_dir` at the *training output directory* (e.g.
  `outputs/tinyllama_train_...`). The script will automatically prefer
  the final aggregated *global* model files:
    1) `checkpoints/ckpt_global_final.pt` (if present)
    2) `adapter_state_final.pt` + `heads_state_final.pt` (if present)
    3) fallback: whatever `resolve_load_paths()` finds in the directory

- If a `run_config.json` exists next to the trained model, missing CLI
  hyperparameters are filled from it (model_name, lora_*, num_B,
  temperature, max_length, fp16, offline, ...).

- For FLoRA-style stacking (anything except `centralized` or `fedit`),
  the final global LoRA rank is computed to match the training script:
      global_r = local_r * (num_clients ** num_fl_rounds)

The goal is that evaluation can run fully automated (e.g. from a Slurm
batch script) without hand-maintaining per-run global hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

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


# -----------------------------------------------------------------------------
# Helpers


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_run_dir_from_path(path: str) -> str:
    """Infer the training run directory from a given path.

    Accepts:
    - run dir itself
    - run dir/checkpoints
    - run dir/checkpoints/<ckpt>.pt
    - arbitrary file inside run dir
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)

    # If they pass ".../checkpoints", go one up.
    if os.path.basename(path) == "checkpoints":
        path = os.path.dirname(path)

    return path


def _find_run_config(run_dir: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Find and load run_config.json in run_dir or its parent."""
    cand1 = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cand1):
        return cand1, _read_json(cand1)

    parent = os.path.dirname(run_dir)
    cand2 = os.path.join(parent, "run_config.json")
    if os.path.exists(cand2):
        return cand2, _read_json(cand2)

    return None, None


def _compute_final_global_lora_r(local_r: int, num_clients: int, num_fl_rounds: int, strat: str) -> int:
    """Match the TinyLlama training script's LoRA-rank evolution.

    Training script behaviour:
      - strat == "fedit" or "centralized": rank stays constant (local_r)
      - else (FLoRA / "federated"): rank is multiplied by num_clients each FL round
    """
    strat_l = (strat or "").lower()

    if strat_l in {"fedit", "centralized"}:
        return int(local_r)

    # Default / "federated" / "flora": iterative stacking
    # After R rounds: local_r * (num_clients ** R)
    return int(local_r) * (int(num_clients) ** int(num_fl_rounds))


def _maybe_fill_from_run_config(args: argparse.Namespace, run_cfg: Optional[Dict[str, Any]]) -> None:
    """Fill missing CLI args from run_config.json (in-place)."""
    if not run_cfg:
        return

    # Model
    if args.model_name is None and "model_name" in run_cfg:
        args.model_name = run_cfg["model_name"]

    # Core eval/tokenization
    if args.max_length is None and "max_length" in run_cfg:
        args.max_length = int(run_cfg["max_length"])

    # LoRA/mLoRA hparams
    if args.lora_alpha is None and "lora_alpha" in run_cfg:
        args.lora_alpha = int(run_cfg["lora_alpha"])
    if args.lora_dropout is None and "lora_dropout" in run_cfg:
        args.lora_dropout = float(run_cfg["lora_dropout"])
    if args.num_B is None and "num_B" in run_cfg:
        args.num_B = int(run_cfg["num_B"])
    if args.temperature is None and "temperature" in run_cfg:
        args.temperature = float(run_cfg["temperature"])

    # fp16/offline are tri-state (None means: not specified on CLI)
    if args.fp16 is None and "fp16" in run_cfg:
        args.fp16 = bool(run_cfg["fp16"])
    if args.offline is None and "offline" in run_cfg:
        args.offline = bool(run_cfg["offline"])

    # Derive final *global* rank (important!)
    if args.lora_r is None:
        local_r = int(run_cfg.get("lora_r", 4))
        num_clients = int(run_cfg.get("num_clients", 1))
        num_fl_rounds = int(run_cfg.get("num_fl_rounds", 1))
        strat = str(run_cfg.get("strat", "centralized"))
        args.lora_r = _compute_final_global_lora_r(local_r, num_clients, num_fl_rounds, strat)


def _finalize_defaults(args: argparse.Namespace) -> None:
    """Ensure args have sensible defaults after config-resolution."""
    if args.model_name is None:
        args.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    if args.max_length is None:
        args.max_length = 256

    if args.lora_r is None:
        args.lora_r = 4
    if args.lora_alpha is None:
        args.lora_alpha = 16
    if args.lora_dropout is None:
        args.lora_dropout = 0.05
    if args.num_B is None:
        args.num_B = 2
    if args.temperature is None:
        args.temperature = 0.1

    if args.fp16 is None:
        args.fp16 = False
    if args.offline is None:
        args.offline = False


def _resolve_global_weights_from_dir(load_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Resolve which weights to load from a training run directory.

    Returns:
      (ckpt_path, adapter_path, heads_path, resolved_from_dir)

    One of ckpt_path OR (adapter_path and heads_path) will be set.
    """
    run_dir = _infer_run_dir_from_path(load_dir)

    # 1) Prefer explicit final global checkpoint
    ckpt_global_final = os.path.join(run_dir, "checkpoints", "ckpt_global_final.pt")
    if os.path.exists(ckpt_global_final):
        return ckpt_global_final, None, None, run_dir

    # 2) Prefer explicit final adapter/head state
    adapter_final = os.path.join(run_dir, "adapter_state_final.pt")
    heads_final = os.path.join(run_dir, "heads_state_final.pt")
    if os.path.exists(adapter_final) and os.path.exists(heads_final):
        return None, adapter_final, heads_final, run_dir

    # 3) Fallback: resolve_load_paths (may pick *_last or plain *.pt)
    adapter_path, heads_path = resolve_load_paths(run_dir)
    if adapter_path is None or heads_path is None:
        raise FileNotFoundError(
            f"Could not resolve adapter/heads in '{run_dir}'. "
            f"Expected ckpt_global_final.pt OR adapter_state_final.pt+heads_state_final.pt "
            f"OR adapter_state*.pt+heads_state*.pt."
        )
    return None, adapter_path, heads_path, run_dir


# -----------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TinyLlama + mLoRA on GLUE (single GPU)")

    # Model / output
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Pretrained TinyLlama model name. If omitted and run_config.json exists, it is taken from there.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If unset, defaults to <run_dir>/eval_only (where run_dir is derived from --load_dir/--load_ckpt).",
    )
    p.add_argument("--seed", type=int, default=42)

    # Loading
    p.add_argument(
        "--load_dir",
        type=str,
        default=None,
        help=(
            "Directory containing the trained model. Recommended: the training output directory "
            "(e.g. outputs/tinyllama_train_...). The script will prefer the final global files."
        ),
    )
    p.add_argument("--load_adapter", type=str, default=None)
    p.add_argument("--load_heads", type=str, default=None)
    p.add_argument("--load_ckpt", type=str, default=None)

    # Data / eval
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Tokenization max length. If omitted and run_config.json exists, it is taken from there.",
    )
    p.add_argument("--num_workers", type=int, default=2)

    # mLoRA hyperparams (must match training). Defaults are None so they can be auto-filled from run_config.json.
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument("--num_B", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)

    # Mixed precision (tri-state)
    fp16_group = p.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16", dest="fp16", action="store_true", help="Enable CUDA AMP (fp16)")
    fp16_group.add_argument("--no_fp16", dest="fp16", action="store_false", help="Disable fp16 AMP")
    p.set_defaults(fp16=None)

    # Offline mode (tri-state)
    offline_group = p.add_mutually_exclusive_group()
    offline_group.add_argument("--offline", dest="offline", action="store_true", help="Use offline (local) HF caches only")
    offline_group.add_argument("--online", dest="offline", action="store_false", help="Allow online HF access")
    p.set_defaults(offline=None)

    # HF / dataset cache
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


# -----------------------------------------------------------------------------
# Main


def main() -> None:
    args = parse_args()

    if not (args.load_ckpt or args.load_dir or (args.load_adapter and args.load_heads)):
        raise ValueError("Provide --load_ckpt OR --load_dir OR (--load_adapter and --load_heads).")

    # Determine run_dir for config + default output placement
    run_dir: Optional[str] = None
    if args.load_dir:
        run_dir = _infer_run_dir_from_path(args.load_dir)
    elif args.load_ckpt:
        run_dir = _infer_run_dir_from_path(args.load_ckpt)
    elif args.load_adapter:
        run_dir = _infer_run_dir_from_path(args.load_adapter)

    # Load run_config.json if present
    run_cfg_path, run_cfg = (None, None)
    if run_dir:
        run_cfg_path, run_cfg = _find_run_config(run_dir)

    # Fill missing args from config and finalize defaults
    _maybe_fill_from_run_config(args, run_cfg)
    _finalize_defaults(args)

    # Determine output directory
    if args.output_dir is None:
        if run_dir:
            args.output_dir = os.path.join(run_dir, "eval_only")
        else:
            args.output_dir = "./eval_only"
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed for reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    # HF configuration
    hf_token = get_hf_token(args.hf_token)
    hf_home = default_hf_home()
    glue_disk_cache_dir = args.glue_disk_cache_dir or os.path.join(hf_home, "glue_disk_cache")
    hf_datasets_cache_dir = args.hf_datasets_cache_dir or os.environ.get("HF_DATASETS_CACHE") or None

    # Pretty info
    print(f"[INFO] device={device} use_amp={use_amp}")
    print(f"[INFO] output_dir={args.output_dir}")
    if args.load_dir:
        print(f"[INFO] load_dir={args.load_dir}")
    if args.load_ckpt:
        print(f"[INFO] load_ckpt={args.load_ckpt}")
    if run_cfg_path:
        print(f"[INFO] run_config={run_cfg_path}")
    print(
        "[INFO] model/hparams: "
        f"model_name={args.model_name} lora_r={args.lora_r} lora_alpha={args.lora_alpha} "
        f"lora_dropout={args.lora_dropout} num_B={args.num_B} temperature={args.temperature} "
        f"max_length={args.max_length} fp16={args.fp16} offline={args.offline}"
    )

    # Create tokenizer/model
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

    # Resolve and load weights (prefer final global artifacts)
    if args.load_ckpt:
        ckpt_path = args.load_ckpt
        print(f"[CKPT] Loading full checkpoint from {ckpt_path}")
        load_from_checkpoint(ckpt_path, model=model, optimizer=None, scheduler=None, scaler=None, strict_heads=True)
    elif args.load_adapter and args.load_heads:
        print(f"[LOAD] adapter={args.load_adapter}")
        print(f"[LOAD] heads={args.load_heads}")
        load_adapter_and_heads(model, adapter_path=args.load_adapter, heads_path=args.load_heads, strict_heads=True)
    else:
        ckpt_path, adapter_path, heads_path, resolved_dir = _resolve_global_weights_from_dir(args.load_dir)
        if ckpt_path:
            print(f"[CKPT] Using final global checkpoint: {ckpt_path}")
            load_from_checkpoint(ckpt_path, model=model, optimizer=None, scheduler=None, scaler=None, strict_heads=True)
        else:
            print(f"[LOAD] Using adapter/heads from {resolved_dir}")
            print(f"[LOAD] adapter={adapter_path}")
            print(f"[LOAD] heads={heads_path}")
            load_adapter_and_heads(model, adapter_path=adapter_path, heads_path=heads_path, strict_heads=True)

    # Disable gradients
    for p in model.parameters():
        p.requires_grad = False

    # Build dataloaders (single-client eval; we want full task dict, not per-client splits)
    task_data = build_dataloaders(
        tokenizer,
        max_length=args.max_length,
        train_batch_size=1,  # unused in eval
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        hf_datasets_cache_dir=hf_datasets_cache_dir,
        glue_disk_cache_dir=glue_disk_cache_dir,
        hf_token=hf_token,
        offline=args.offline,
        save_eval_details=args.save_eval_details,
        # IMPORTANT: do NOT pass num_clients>1 here; eval expects a single task dict.
    )

    # Unwrap the list if there is exactly one element (single-client setup).
    if isinstance(task_data, list):
        if len(task_data) != 1:
            raise ValueError(f"Expected a single client in eval; got {len(task_data)} clients.")
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

    # Persist the (resolved) eval configuration for reproducibility
    with open(os.path.join(args.output_dir, "eval_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()