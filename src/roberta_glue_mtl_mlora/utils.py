from __future__ import annotations

import os
import random
import sys
from typing import Any

import torch
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    """Best-effort reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tqdm_main(*args: Any, **kwargs: Any):
    """tqdm configured for Slurm log files."""
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 5.0)
    kwargs.setdefault("leave", True)
    return tqdm(*args, **kwargs)
