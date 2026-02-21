"""Model definitions for TinyLlama + mLoRA on GLUE.

This file contains helper functions to inject mLoRA into a TinyLlama
encoder and a wrapper class that adds per‑task classification heads.
The overall design mirrors the RoBERTa implementation in
``src/roberta_glue_mtl_mlora/model.py``, but adapts the logic for
decoder‑only Llama models.  The heads operate on the final token of
the hidden state sequence rather than a ``[CLS]`` token.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .constants import GLUE_TASKS, TASK_NUM_LABELS, TASK_TO_ID

# Import the mLoRA layer and context manager from the shared adapter
try:
    from src.adapter.mlora import mLoRALinear  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import mLoRALinear from src.adapter.mlora. "
        "Ensure that the parent repository includes the adapter implementation."
    ) from exc

try:
    from src.adapter.mlora import mlora_set_lambda_index  # type: ignore
except Exception:
    mlora_set_lambda_index = None  # type: ignore


def replace_llama_linears_with_mlora(
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
    """Replace Llama attention Linear layers with ``mLoRALinear``.

    This function traverses the provided model, finds all modules
    whose fully qualified name contains any of the substrings in
    ``target_substrings`` and which are instances of
    ``torch.nn.Linear``.  Those layers are replaced by
    ``mLoRALinear`` layers with the specified hyperparameters.  The
    new layers are initialised from the original weights and placed on
    the same device/dtype.  The number of replaced layers is
    returned.
    """
    replaced = 0
    # Precompute module names because modifying the module tree
    # invalidates the iterator returned by ``named_modules``.
    module_names: List[str] = [name for name, _ in model.named_modules()]
    for name in module_names:
        if not any(s in name for s in target_substrings):
            continue
        try:
            target = model.get_submodule(name)
        except AttributeError:
            continue
        # Only replace plain Linear layers
        if not isinstance(target, nn.Linear):
            continue
        # Avoid double replacement
        if isinstance(target, mLoRALinear):
            continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        # Construct a new mLoRALinear with the same dimensions
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
        # Place on same device/dtype
        new_layer.to(device=target.weight.device, dtype=target.weight.dtype)
        # Load original weights into the new layer
        new_layer.load_state_dict(target.state_dict(), strict=False)
        # Replace on parent
        setattr(parent, child_name, new_layer)
        replaced += 1
    return replaced


class MultiTaskLlamaMLoRA(nn.Module):
    """Shared Llama encoder plus per‑task classification heads.

    The encoder is assumed to have been wrapped with mLoRA adapters via
    ``replace_llama_linears_with_mlora``.  For each GLUE task a
    separate linear head maps the final hidden state (last token) to
    logits.  During the forward pass the appropriate LoRA lambda index
    is set via ``mlora_set_lambda_index`` based on the task id.  This
    ensures that the correct task‑specific B matrices are used.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        # One head per task; note that heads are stored in a ModuleDict for
        # easy access and separate parameter groups
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, TASK_NUM_LABELS[task]) for task in GLUE_TASKS}
        )

    def forward(self, task: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute logits for a single task.

        Args:
            task: The GLUE task name.  Must be present in ``TASK_TO_ID``.
            batch: A dictionary of input tensors as produced by a
                HuggingFace ``DataCollatorWithPadding`` (e.g. contains
                ``input_ids`` and ``attention_mask``).  Additional
                keys are passed through to the underlying encoder.

        Returns:
            A tensor of shape ``(batch_size, num_labels)`` containing
            logits for the specified task.
        """
        if mlora_set_lambda_index is None:
            raise RuntimeError(
                "mlora_set_lambda_index is not available. Ensure that "
                "src/adapter/mlora.py defines this context manager so that "
                "task IDs can be propagated to mLoRALinear layers."
            )
        if task not in TASK_TO_ID:
            raise ValueError(f"Unknown task '{task}'. Valid tasks: {list(TASK_TO_ID.keys())}")
        task_id = TASK_TO_ID[task]
        # Set the global lambda index for mLoRA so that the adapter uses
        # the correct slice of B and Lambda matrices.
        with mlora_set_lambda_index(task_id):
            outputs = self.encoder(**batch)
        # For decoder models the [CLS] token does not exist; use the
        # representation of the final token instead.  ``last_hidden_state``
        # has shape (batch_size, seq_len, hidden_size).
        final_state = outputs.last_hidden_state[:, -1]  # shape (batch_size, hidden)
        logits = self.heads[task](final_state)
        return logits


def set_trainable_params(
    model: nn.Module,
    *,
    train_lora: bool = True,
    train_heads: bool = True,
    train_bias: bool = True,
    train_layernorm: bool = True,
) -> None:
    """Freeze all parameters then unfreeze selected groups.

    This helper follows the logic of the RoBERTa implementation: all
    parameters are initially frozen; then LoRA parameters (those whose
    names contain ``"lora_"``), head parameters (those in
    ``heads.``), bias terms and layernorm parameters are selectively
    unfrozen based on the supplied boolean flags.

    If you choose to train bias or LayerNorm parameters you must
    remember to save them along with the LoRA adapter weights.
    """
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze heads
    if train_heads:
        for name, p in model.named_parameters():
            if name.startswith("heads."):
                p.requires_grad = True
    # Unfreeze LoRA parameters
    if train_lora:
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
    # Unfreeze biases
    if train_bias:
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
    # Unfreeze LayerNorm parameters
    if train_layernorm:
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True


def cast_trainable_params_to_fp32(model: nn.Module) -> List[str]:
    """Ensure all trainable parameters are in float32.

    When using mixed precision it is beneficial for the trainable
    parameters (LoRA, heads, bias, LayerNorm) to remain in FP32 for
    numerical stability.  This function walks through all
    trainable parameters and casts them to ``float32``.

    Returns:
        A list of parameter names that were cast.
    """
    casted: List[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            casted.append(name)
    return casted
