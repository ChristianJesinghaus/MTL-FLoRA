from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .constants import GLUE_TASKS, TASK_NUM_LABELS, TASK_TO_ID

# Repo adapter (mLoRA)
from src.adapter.mlora import mLoRALinear  # noqa: E402

# Context manager to set global task id for mLoRA (required for RoBERTa/BERT-style forwards)
try:
    from src.adapter.mlora import mlora_set_lambda_index  # noqa: E402
except Exception:
    mlora_set_lambda_index = None


def replace_roberta_linears_with_mlora(
    model: nn.Module,
    *,
    target_substrings: Tuple[str, ...],
    r: int,
    alpha: int,
    dropout: float,
    num_B: int,
    lambda_num: int,
    temperature: float,
) -> int:
    """Replace attention Linear layers with mLoRALinear."""

    replaced = 0
    module_names = [name for name, _ in model.named_modules()]

    for name in module_names:
        if not any(s in name for s in target_substrings):
            continue

        try:
            target = model.get_submodule(name)
        except AttributeError:
            continue

        if not isinstance(target, nn.Linear):
            continue
        if isinstance(target, mLoRALinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        new_layer = mLoRALinear(
            in_features=target.in_features,
            out_features=target.out_features,
            B_num=num_B,
            lambda_num=lambda_num,
            diagonal_format=False,
            B_scale=temperature,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=(target.bias is not None),
        )
        new_layer.to(device=target.weight.device, dtype=target.weight.dtype)
        new_layer.load_state_dict(target.state_dict(), strict=False)

        setattr(parent, child_name, new_layer)
        replaced += 1

    return replaced


class MultiTaskRobertaMLoRA(nn.Module):
    """Shared encoder + per-task classification/regression heads."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({task: nn.Linear(hidden_size, TASK_NUM_LABELS[task]) for task in GLUE_TASKS})

    def forward(self, task: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if mlora_set_lambda_index is None:
            raise RuntimeError(
                "mlora_set_lambda_index is not available, but your mLoRA implementation requires a task id.\n"
                "Please ensure src/adapter/mlora.py provides mlora_set_lambda_index."
            )

        task_id = TASK_TO_ID[task]
        with mlora_set_lambda_index(task_id):
            outputs = self.encoder(**batch)

        cls = outputs.last_hidden_state[:, 0]  # [batch, hidden]
        logits = self.heads[task](cls)
        return logits


def set_trainable_params(
    model: nn.Module,
    *,
    train_lora: bool = True,
    train_heads: bool = True,
    train_bias: bool = True,
    train_layernorm: bool = True,
) -> None:
    """Freeze everything, then unfreeze requested parameter groups.

    - LoRA params: anything with "lora_" in the name
    - Heads: anything under "heads."
    - Bias: any parameter ending with ".bias"
    - LayerNorm: parameters in nn.LayerNorm modules

    NOTE: Bias/LayerNorm belong to the base encoder, so if you train them you must
    also SAVE them (we do this by saving all trainable encoder params).
    """

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze heads
    if train_heads:
        for name, p in model.named_parameters():
            if name.startswith("heads."):
                p.requires_grad = True

    # Unfreeze LoRA
    if train_lora:
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    # Unfreeze bias
    if train_bias:
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True

    # Unfreeze LayerNorm (more robust than name matching)
    if train_layernorm:
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True


def cast_trainable_params_to_fp32(model: nn.Module) -> List[str]:
    """Ensure all trainable params are fp32 (helps stability + AMP scaler)."""

    casted: List[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            casted.append(name)
    return casted


def count_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def summarize_trainable_params(model: nn.Module) -> Dict[str, int]:
    """Return counts per group (heuristic by name/module)."""

    counts = {"lora": 0, "heads": 0, "bias": 0, "layernorm": 0, "other": 0}

    # Identify LN params by id
    ln_param_ids = set()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters(recurse=False):
                ln_param_ids.add(id(p))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("heads."):
            counts["heads"] += p.numel()
        elif "lora_" in name:
            counts["lora"] += p.numel()
        elif id(p) in ln_param_ids:
            counts["layernorm"] += p.numel()
        elif name.endswith(".bias"):
            counts["bias"] += p.numel()
        else:
            counts["other"] += p.numel()

    return counts


def get_trainable_encoder_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect *only* the trainable encoder parameters.

    This is what we save as "adapter_state" (LoRA + optional bias/LN), so we don't
    have to save the full RoBERTa weights.
    """

    # We expect model to be MultiTaskRobertaMLoRA; but keep it generic.
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError("Model has no attribute 'encoder'; expected MultiTaskRobertaMLoRA")

    state: Dict[str, torch.Tensor] = {}
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu()
    return state


def load_trainable_encoder_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError("Model has no attribute 'encoder'; expected MultiTaskRobertaMLoRA")

    # strict=False: because we only provide a subset of weights
    encoder.load_state_dict(state, strict=False)
