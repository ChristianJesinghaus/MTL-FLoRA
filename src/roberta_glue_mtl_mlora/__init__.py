"""Utilities for running RoBERTa + (MTL-)mLoRA on GLUE (single GPU).

This package is a refactor of the monolithic script into smaller modules:
- constants: task metadata
- hf_utils: HF token + disk cache loading
- data: dataloaders
- model: model + mLoRA injection + trainable parameter selection
- train_loop: training loop
- eval_loop: evaluation loop + optional per-example dumps
- checkpoint: save/load checkpoints + adapter/head weights

The goal is clarity and reusability, not an API guarantee.
"""

__all__ = [
    "constants",
    "hf_utils",
    "data",
    "model",
    "metrics",
    "prompts",
    "checkpoint",
    "train_loop",
    "eval_loop",
    "utils",
]
