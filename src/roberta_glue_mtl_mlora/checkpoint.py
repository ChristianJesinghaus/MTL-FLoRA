from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .model import get_trainable_encoder_state, load_trainable_encoder_state


def save_checkpoint(
    *,
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    update_step: int,
    args: Any,
) -> None:
    """Save a full training checkpoint (for resume)."""

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    adapter_state = get_trainable_encoder_state(model)
    heads_state = {k: v.detach().cpu() for k, v in model.heads.state_dict().items()}  # type: ignore[attr-defined]

    state = {
        "epoch": int(epoch),
        "update_step": int(update_step),
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
        "adapter_state": adapter_state,
        "heads_state": heads_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "scaler": scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None,
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    torch.save(state, ckpt_path)
    print(f"[CKPT] saved {ckpt_path}", flush=True)


def maybe_rotate_checkpoints(ckpt_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return

    p = Path(ckpt_dir)
    if not p.exists():
        return

    ckpts = sorted([x for x in p.glob("ckpt_*.pt")], key=lambda x: x.stat().st_mtime)
    while len(ckpts) > keep_last:
        old = ckpts.pop(0)
        try:
            old.unlink()
            print(f"[CKPT] removed old checkpoint: {old.name}", flush=True)
        except Exception as e:
            print(f"[WARN] could not remove old checkpoint {old}: {e}", flush=True)
            break


def load_from_checkpoint(
    ckpt_path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict_heads: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    adapter_state = ckpt.get("adapter_state", {})
    heads_state = ckpt.get("heads_state", {})

    if adapter_state:
        load_trainable_encoder_state(model, adapter_state)

    if heads_state:
        model.heads.load_state_dict(heads_state, strict=strict_heads)  # type: ignore[attr-defined]

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    return ckpt


def save_adapter_and_heads(output_dir: str, model: torch.nn.Module) -> None:
    """Save trainable encoder params (adapter_state) and heads weights.

    This is the lightweight artifact you typically want for evaluation.
    """

    os.makedirs(output_dir, exist_ok=True)

    adapter_state = get_trainable_encoder_state(model)
    heads_state = {k: v.detach().cpu() for k, v in model.heads.state_dict().items()}  # type: ignore[attr-defined]

    torch.save(adapter_state, os.path.join(output_dir, "adapter_state.pt"))
    torch.save(heads_state, os.path.join(output_dir, "heads_state.pt"))

    # convenience "last" copies
    torch.save(adapter_state, os.path.join(output_dir, "adapter_state_last.pt"))
    torch.save(heads_state, os.path.join(output_dir, "heads_state_last.pt"))


def resolve_load_paths(load_dir: str) -> Tuple[Optional[str], Optional[str]]:
    d = Path(load_dir)
    # Prioritize final versions from FL training, then fall back to regular versions
    cand_adapter = [d / "adapter_state_final.pt", d / "adapter_state_last.pt", d / "adapter_state.pt"]
    cand_heads = [d / "heads_state_final.pt", d / "heads_state_last.pt", d / "heads_state.pt"]

    ap = next((str(p) for p in cand_adapter if p.exists()), None)
    hp = next((str(p) for p in cand_heads if p.exists()), None)
    return ap, hp


def load_adapter_and_heads(
    model: torch.nn.Module,
    *,
    adapter_path: str,
    heads_path: str,
    strict_heads: bool = True,
) -> None:
    adapter_state = torch.load(adapter_path, map_location="cpu")
    load_trainable_encoder_state(model, adapter_state)

    heads_state = torch.load(heads_path, map_location="cpu")
    model.heads.load_state_dict(heads_state, strict=strict_heads)  # type: ignore[attr-defined]
