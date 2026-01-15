"""Shared LoRA base utilities.

This repo originally relied on DeepSpeed ZeRO-3 to gather partitioned parameters.
On small single-GPU setups (e.g., GTX 1080 Ti) DeepSpeed is typically not used.
To keep the adapter code importable without DeepSpeed, we make DeepSpeed optional.

If DeepSpeed is installed and ZeRO is enabled, the original behaviour is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# DeepSpeed is optional (we don't use it for the 1080 Ti GLUE reproduction).
try:
    import deepspeed  # type: ignore
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus  # type: ignore

    _DEEPSPEED_AVAILABLE = True
except Exception:  # pragma: no cover
    deepspeed = None  # type: ignore

    class ZeroParamStatus:  # type: ignore
        """Fallback stub for deepspeed.runtime.zero.partition_parameters.ZeroParamStatus."""

        NOT_AVAILABLE = "NOT_AVAILABLE"

    _DEEPSPEED_AVAILABLE = False


class EmptyContext:
    def __enter__(self):
        return self

    def __exit__(self, *exec):
        return False


def should_gather(param):
    """Context manager that gathers a (possibly ZeRO-partitioned) parameter.

    If DeepSpeed is not available, or ZeRO is not being used, this becomes a no-op.
    """
    if (not _DEEPSPEED_AVAILABLE) or (param is None):
        return EmptyContext()

    should = hasattr(param, "ds_id") and getattr(param, "ds_status", None) == ZeroParamStatus.NOT_AVAILABLE
    # deepspeed.zero.GatheredParameters can take either a single param or a list.
    return deepspeed.zero.GatheredParameters(param, modifier_rank=0, enabled=should)  # type: ignore


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        tunable_scaler: bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        if tunable_scaler:
            # nn.Parameter needs an existing tensor on the correct dtype/device
            self.lora_scaler = nn.Parameter(self.weight.new_zeros(()))
        else:
            self.lora_scaler = None

    def compute_tunable_scale(self, requires_grad: bool = False):
        if self.lora_scaler is None:
            return 1.0
        elif requires_grad:
            return torch.sigmoid(self.lora_scaler)
        else:
            return torch.sigmoid(self.lora_scaler.data).item()
