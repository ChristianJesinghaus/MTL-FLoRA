from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .constants import GLUE_TASKS
from .model import MultiTaskRobertaMLoRA, replace_roberta_linears_with_mlora


def create_tokenizer(
    model_name: str,
    *,
    offline: bool,
) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=offline)


def create_encoder(
    model_name: str,
    *,
    offline: bool,
    device: torch.device,
) -> torch.nn.Module:
    # Load encoder in FP32
    try:
        encoder = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            local_files_only=offline,
        )
    except TypeError:
        encoder = AutoModel.from_pretrained(model_name, local_files_only=offline)

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
) -> MultiTaskRobertaMLoRA:
    encoder = create_encoder(model_name, offline=offline, device=device)

    # Inject mLoRA into attention linears
    target_substrings = (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    )
    replaced = replace_roberta_linears_with_mlora(
        encoder,
        target_substrings=target_substrings,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        num_B=num_B,
        lambda_num=len(GLUE_TASKS),
        temperature=temperature,
    )
    print(f"[INFO] Replaced {replaced} Linear layers with mLoRALinear")

    model = MultiTaskRobertaMLoRA(encoder=encoder, hidden_size=encoder.config.hidden_size).to(device)
    return model
