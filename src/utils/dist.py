"""Small torch.distributed helpers.

The original repo assumed distributed training (DeepSpeed/torchrun) and used
dist.is_available() only. On single-GPU runs without initializing the process
group, dist.get_rank()/get_world_size() raise an exception.

This version is safe for both distributed and non-distributed execution.
"""

from __future__ import annotations

import torch.distributed as dist


def get_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_global_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1
