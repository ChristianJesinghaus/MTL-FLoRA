#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List, Optional

from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn.functional as F
from torch import nn

from .base import LoRALayer, should_gather


# --------------------------------------------------------------------------------------
# Compatibility helper:
# HuggingFace encoder models (RoBERTa/BERT) call Linear layers without extra args.
# For single-task batches we route the mLoRA task-id via a global context.
#
# Usage:
#   from src.adapter.mlora import mlora_set_lambda_index
#   with mlora_set_lambda_index(task_id):
#       outputs = encoder(**batch)
# --------------------------------------------------------------------------------------

_CURRENT_LAMBDA_INDEX: ContextVar[Optional[int]] = ContextVar("mlora_lambda_index", default=None)


@contextmanager
def mlora_set_lambda_index(task_id: int):
    token = _CURRENT_LAMBDA_INDEX.set(int(task_id))
    try:
        yield
    finally:
        _CURRENT_LAMBDA_INDEX.reset(token)


def _resolve_lambda_index(x: torch.Tensor, lambda_index: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Resolve lambda_index for the current batch.

    - If `lambda_index` is provided: ensure it's a 1D LongTensor on the correct device.
    - If `lambda_index` is None: read from the global context set by `mlora_set_lambda_index`.
    """
    batch_size = int(x.shape[0])

    if lambda_index is None:
        task_id = _CURRENT_LAMBDA_INDEX.get()
        if task_id is None:
            raise RuntimeError(
                "mLoRA layer was called without `lambda_index` and no global task id is set. "
                "Wrap the encoder forward pass in `with mlora_set_lambda_index(task_id): ...` "
                "or pass `lambda_index` explicitly."
            )
        return torch.full((batch_size,), int(task_id), device=x.device, dtype=torch.long)

    if isinstance(lambda_index, torch.Tensor):
        li = lambda_index
    else:
        li = torch.tensor(lambda_index, device=x.device)

    if li.dim() == 0:
        li = li.view(1)
    if li.numel() == 1 and batch_size > 1:
        li = li.repeat(batch_size)
    if li.numel() != batch_size:
        raise ValueError(
            f"lambda_index must have shape (batch_size,) or be a scalar; got {tuple(li.shape)} for batch_size={batch_size}"
        )
    if li.device != x.device:
        li = li.to(x.device)
    if li.dtype != torch.long:
        li = li.long()
    return li


class mLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        B_num: int,
        lambda_num: int,
        diagonal_format: bool = True,
        B_scale: float = 0.0,
        dec_param: str = "Q.K.V.O",
        lora_param: str = "Q.K.V.O",
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        tunable_scaler: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            tunable_scaler=tunable_scaler,
        )
        self.fan_in_fan_out = fan_in_fan_out
        self.B_num = B_num
        self.lambda_num = lambda_num
        self.diagonal_format = diagonal_format
        self.B_scale = B_scale

        self.dec_param = dec_param
        self.lora_param = lora_param

        # NEW: block_size determines the size of each softmax block for lora_B_w.
        # When set, softmax will be computed independently within each block rather than globally.
        # For local models the block_size should equal num_B; for aggregated models it remains the per-client B.
        self.block_size: Optional[int] = None

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((1, r, in_features)))  # task-agnostic

            if not diagonal_format:
                self.lora_lambdas = nn.Parameter(self.weight.new_zeros((lambda_num, r, r)))  # task-specific
            else:
                self.lora_lambdas = nn.Parameter(self.weight.new_zeros((lambda_num, r)))  # task-specific

            self.lora_B = nn.Parameter(self.weight.new_zeros((B_num, out_features, r)))  # task-agnostic
            self.lora_B_w = nn.Parameter(self.weight.new_zeros((lambda_num, B_num)))    # task-specific
            self.scaling = lora_alpha / r

            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            for i in range(self.lora_A.size(0)):
                nn.init.kaiming_uniform_(self.lora_A[i, ...], a=math.sqrt(5))
            if not self.diagonal_format:
                for i in range(self.lora_lambdas.size(0)):
                    nn.init.eye_(self.lora_lambdas[i, ...])
            else:
                nn.init.ones_(self.lora_lambdas)
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_B_w, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor, lambda_index: Optional[torch.Tensor] = None, statistics=None):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        lambda_index = _resolve_lambda_index(x, lambda_index)

        # one task-id per sample (shape: [batch])
        zero_index = torch.zeros_like(lambda_index)

        lora_A = torch.index_select(self.lora_A, 0, zero_index)                 # [B, r, in]
        lora_lambdas = torch.index_select(self.lora_lambdas, 0, lambda_index)   # [B, r, r] or [B, r]
        if self.diagonal_format:
            lora_lambdas = torch.diag_embed(lora_lambdas)                       # [B, r, r]

        # ----- IMPORTANT FIX -----
        # Previous code used: task_B = lora_B_w @ self.lora_B.view((B_num, -1))
        # This triggers cuBLAS GEMM with tiny k (=B_num), which can crash on some GPU/software stacks.
        # We compute the weighted sum without GEMM:
        #
        # task_B[b] = sum_i lora_B_w[b,i] * lora_B[i]
        #
        # Shapes:
        #   lora_B_w: [B, B_num]
        #   lora_B:   [B_num, out, r]
        #   task_B:   [B, out, r]
        lora_B_w = torch.index_select(self.lora_B_w, 0, lambda_index)  # [B, B_num]
        if self.B_scale > 0:
            # Apply softmax globally or blockwise depending on block_size.
            if self.block_size is not None and self.block_size > 0 and (lora_B_w.shape[-1] % self.block_size) == 0:
                num_blocks = lora_B_w.shape[-1] // self.block_size
                w = lora_B_w.view(lora_B_w.shape[0], num_blocks, self.block_size)
                w = F.softmax(w / self.B_scale, dim=-1, dtype=torch.float32)
                lora_B_w = w.view_as(lora_B_w)
            else:
                lora_B_w = F.softmax(lora_B_w / self.B_scale, dim=-1, dtype=torch.float32)
        # ensure dtype matches B
        lora_B_w = lora_B_w.to(self.lora_B.dtype)

        # Broadcast multiply + reduce over B_num
        task_B = (lora_B_w[:, :, None, None] * self.lora_B[None, :, :, :]).sum(dim=1)  # [B, out, r]
        lora_B = task_B

        # base projection
        result = F.linear(x, T(self.weight), bias=self.bias)

        if self.r > 0:
            dropout_x = self.lora_dropout(x)
            after_A = torch.bmm(dropout_x, lora_A.transpose(-2, -1))                 # [B, seq, r]
            after_A = torch.bmm(after_A, lora_lambdas.transpose(-2, -1))             # [B, seq, r]
            after_B = torch.bmm(after_A, lora_B.transpose(-2, -1))                   # [B, seq, out]
            result += after_B * self.scaling * self.compute_tunable_scale(requires_grad=False)

        if statistics is not None:
            statistics["after_A"] = after_A
            statistics["after_B"] = after_B

        return result


class mLoRAMergedLinear(nn.Linear, LoRALayer):
    # (unused in TinyLlama setup)
    def __init__(
        self,
        in_features: int,
        out_features: int,
        B_num: int,
        lambda_num: int,
        diagonal_format: bool = True,
        B_scale: float = 0.0,
        dec_param: str = "Q.K.V.O",
        lora_param: str = "Q.K.V.O",
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        tunable_scaler: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            tunable_scaler=tunable_scaler,
        )
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.B_num = B_num
        self.lambda_num = lambda_num
        self.diagonal_format = diagonal_format
        self.B_scale = B_scale

        self.dec_param = dec_param
        self.lora_param = lora_param

        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((len(enable_lora), r, in_features)))
            if not diagonal_format:
                self.lora_lambdas = nn.Parameter(self.weight.new_zeros((lambda_num, r, r)))
            else:
                self.lora_lambdas = nn.Parameter(self.weight.new_zeros((lambda_num, r)))

            self.lora_B = nn.Parameter(
                self.weight.new_zeros((B_num, out_features // len(enable_lora) * sum(enable_lora), r))
            )
            self.lora_B_w = nn.Parameter(self.weight.new_zeros((lambda_num, B_num)))
            self.scaling = lora_alpha / r

            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            for i in range(self.lora_A.size(0)):
                nn.init.kaiming_uniform_(self.lora_A[i, ...], a=math.sqrt(5))
            if not self.diagonal_format:
                for i in range(self.lora_lambdas.size(0)):
                    nn.init.eye_(self.lora_lambdas[i, ...])
            else:
                nn.init.ones_(self.lora_lambdas)
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_B_w, a=math.sqrt(5))

    # (forward of merged layer unchanged)