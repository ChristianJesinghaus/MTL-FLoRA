"""Factory functions for TinyLlama + mLoRA models and tokenizers.

This module provides convenience functions to load a pretrained
TinyLlama model, inject mLoRA adapters into its attention layers,
construct per‑task classification heads and return a complete
``MultiTaskLlamaMLoRA`` instance ready for training or evaluation.
``create_tokenizer`` handles padding token configuration for
decoder‑only models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .constants import GLUE_TASKS, LLAMA_TARGET_SUBSTRINGS
from .model import MultiTaskLlamaMLoRA, replace_llama_linears_with_mlora


def create_tokenizer(
    model_name: str,
    *,
    offline: bool,
) -> AutoTokenizer:
    """Load a HuggingFace tokenizer for TinyLlama.

    For decoder‑only models like Llama there is typically no padding token
    defined; therefore this function sets the pad token to the EOS token
    if it is missing and ensures that padding occurs on the right.

    Args:
        model_name: Name of the pretrained Llama model on HuggingFace.
        offline: If ``True`` the model will only be loaded from the
            local cache.

    Returns:
        A configured ``AutoTokenizer`` instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=offline)
    # Ensure a pad token exists for batch padding.  Many Llama models
    # define ``pad_token_id`` as None, in which case we reuse the EOS token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Llama models expect right‑padding
    tokenizer.padding_side = "right"
    return tokenizer


def create_encoder(
    model_name: str,
    *,
    offline: bool,
    device: torch.device,
) -> torch.nn.Module:
    """Load the TinyLlama base model (without heads) in float32.

    Args:
        model_name: Name of the pretrained Llama model.
        offline: If ``True`` only look locally.
        device: Device to move the model to.

    Returns:
        The base Llama model as a ``torch.nn.Module``.
    """
    # ``add_pooling_layer=False`` is only valid for BERT/RoBERTa; avoid
    # passing it here.  We rely on the AutoModel class to choose the
    # appropriate architecture (causal LM).
    encoder = AutoModel.from_pretrained(
        model_name,
        local_files_only=offline,
    )
    encoder.to(device)
    return encoder


def create_model(
    *,
    model_name: str,
    offline: bool,
    device: torch.device,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    num_B: int,
    temperature: float,
) -> MultiTaskLlamaMLoRA:
    """Construct a ``MultiTaskLlamaMLoRA`` model with injected adapters.

    The returned model uses a shared TinyLlama encoder with mLoRA
    adapters applied to its attention projections (``q_proj``,
    ``k_proj``, ``v_proj``, ``o_proj``) and a classification head per
    GLUE task.  The number of LoRA lambdas is set to ``len(GLUE_TASKS)``.

    Args:
        model_name: Pretrained Llama model name.
        offline: Load only from local cache if ``True``.
        device: Device to place the model on.
        lora_r: LoRA rank (per task or aggregated across tasks).
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout to apply to the LoRA activations.
        num_B: Number of LoRA B matrices per layer (for mLoRA).
        temperature: Scaling applied to task weights in mLoRA.

    Returns:
        A ready‑to‑train ``MultiTaskLlamaMLoRA`` instance.
    """
    encoder = create_encoder(model_name, offline=offline, device=device)
    # Insert mLoRA adapters into the attention projections
    replaced = replace_llama_linears_with_mlora(
        encoder,
        target_substrings=LLAMA_TARGET_SUBSTRINGS,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        num_B=num_B,
        lambda_num=len(GLUE_TASKS),
        temperature=temperature,
    )
    print(f"[INFO] Replaced {replaced} Linear layers with mLoRALinear in TinyLlama encoder")
    model = MultiTaskLlamaMLoRA(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
    ).to(device)
    return model
