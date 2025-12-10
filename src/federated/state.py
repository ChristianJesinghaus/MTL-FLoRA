"""Utilities for serializing adapter weights during Flower rounds."""
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.utils import get_lora_param_maybe_zero_3


def adapter_state_to_numpy(named_params: Iterable[Tuple[str, torch.Tensor]]) -> Tuple[List[str], List[np.ndarray]]:
    """Collect trainable adapter parameters and convert them to numpy arrays.

    The parameter order is deterministic (sorted by name) so that Flower can
    broadcast and aggregate weights consistently across clients.
    """

    adapter_only = get_lora_param_maybe_zero_3(named_params, valid_keys=["lora"])
    keys = sorted(adapter_only.keys())
    weights = [adapter_only[k].cpu().numpy() for k in keys]
    return keys, weights


def apply_adapter_state(model: torch.nn.Module, keys: Sequence[str], weights: Sequence[np.ndarray]) -> None:
    """Load aggregated adapter weights back into the model."""

    if not keys:
        return

    state_dict = model.state_dict()
    for key, weight in zip(keys, weights):
        tensor = torch.as_tensor(weight)
        if key in state_dict:
            state_dict[key] = tensor
    model.load_state_dict(state_dict, strict=False)
