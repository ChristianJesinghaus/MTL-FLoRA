"""
Modified GLUE dataloader for TinyLlama/RoBERTa mLoRA.

This version extends the original `data.py` loader to support evaluation on
the GLUE *test* split.  The original implementation only provided
train/validation loaders; here we add a parameter ``eval_split`` so that
evaluation can optionally be run on the test data instead of the
validation data.  Note that the GLUE test splits generally do not
include labels (the GLUE leaderboard withholds them), so metric
computation will yield meaningless numbers.  When ``eval_split`` is
"test", the returned dataloaders will include the test split for each
task.

Usage:

    from .data_eval_test import build_dataloaders

    # Build loaders with eval_split="test"
    task_data = build_dataloaders(
        tokenizer,
        max_length=256,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=2,
        seed=42,
        hf_datasets_cache_dir=cache_dir,
        glue_disk_cache_dir=glue_cache,
        hf_token=token,
        offline=False,
        save_eval_details=False,
        num_clients=1,
        dirichlet_alpha=1.0,
        test_mode=False,
        eval_split="test",
    )

The returned data structure matches the original `build_dataloaders`,
but ``val_loaders`` for each task will reference the test split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .constants import (
    FALLBACK_LABEL_NAMES,
    GLUE_TASKS,
    TASK_TO_KEYS,
    TEST_SAMPLE_SIZE,
)
from .hf_utils import load_glue_dataset_with_disk_cache


@dataclass
class TaskData:
    train_loader: DataLoader
    val_loaders: List[Tuple[str, DataLoader]]
    label_names: Optional[List[str]]


def extract_label_names_from_dataset(ds_dict: DatasetDict, task: str) -> Optional[List[str]]:
    if task == "stsb":
        return None
    try:
        feat = ds_dict["train"].features.get("label", None)
        names = getattr(feat, "names", None)
        if isinstance(names, list) and len(names) > 0:
            return [str(x) for x in names]
    except Exception:
        pass
    fb = FALLBACK_LABEL_NAMES.get(task)
    if fb:
        max_id = max(fb.keys())
        return [fb.get(i, str(i)) for i in range(max_id + 1)]
    return None


def partition_indices_dirichlet(
    num_samples: int,
    num_clients: int,
    num_classes: int,
    labels: np.ndarray,
    alpha: float,
    seed: int,
) -> List[np.ndarray]:
    """Partition dataset indices using non-IID Dirichlet distribution.

    Args:
        num_samples: Total number of samples
        num_clients: Number of clients
        num_classes: Number of classes
        labels: Array of labels for all samples
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility

    Returns:
        List of index arrays, one per client
    """
    rng = np.random.RandomState(seed)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices: List[np.ndarray] = [[] for _ in range(num_clients)]
    for class_idx, indices in enumerate(class_indices):
        proportions = rng.dirichlet([alpha] * num_clients)
        rng.shuffle(indices)
        splits = (np.cumsum(proportions) * len(indices)).astype(int)
        class_splits = np.split(indices, splits[:-1])
        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split)
    return [np.array(client_idx) for client_idx in client_indices]


def build_dataloaders(
    tokenizer,
    *,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    hf_datasets_cache_dir: Optional[str],
    glue_disk_cache_dir: str,
    hf_token: Optional[str],
    offline: bool,
    save_eval_details: bool,
    num_clients: int = 1,
    dirichlet_alpha: float = 1.0,
    test_mode: bool = False,
    eval_split: str = "validation",
) -> List[Dict[str, TaskData]]:
    """Create per-task dataloaders with optional federated partitioning.

    This version adds the ``eval_split`` argument.  When set to
    ``"validation"`` (default) the returned ``val_loaders`` contain
    validation splits.  When set to ``"test"`` they contain test splits.
    Note that GLUE test splits may not contain labels, so metric
    computation is not meaningful.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length for tokenisation
        train_batch_size: Mini-batch size for training
        eval_batch_size: Mini-batch size for evaluation
        num_workers: Number of DataLoader workers
        seed: Random seed
        hf_datasets_cache_dir: HuggingFace datasets cache directory
        glue_disk_cache_dir: On-disk cache for GLUE datasets
        hf_token: HuggingFace authentication token
        offline: If True, operate in offline mode
        save_eval_details: If True, include raw texts in evaluation details
        num_clients: Number of federated clients
        dirichlet_alpha: Dirichlet concentration for non-IID partitioning
        test_mode: If True, sample a small subset of training data per task
        eval_split: "validation" or "test"; which split to use for evaluation loaders

    Returns:
        A list of dicts: one per client (length == num_clients), mapping
        task names to TaskData.
    """
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def tokenize_fn(task: str):
        sent1_key, sent2_key = TASK_TO_KEYS[task]
        def _fn(examples):
            if sent2_key is None:
                return tokenizer(examples[sent1_key], truncation=True, max_length=max_length)
            return tokenizer(examples[sent1_key], examples[sent2_key], truncation=True, max_length=max_length)
        return _fn

    def make_collate_fn(task: str, include_meta: bool):
        is_regression = task == "stsb"
        sent1_key, sent2_key = TASK_TO_KEYS[task]
        def _collate(features: List[Dict[str, Any]]):
            labels = [f.get("label", -1) for f in features]
            model_keys = ["input_ids", "attention_mask"]
            if "token_type_ids" in features[0]:
                model_keys.append("token_type_ids")
            input_feats = [{k: f[k] for k in model_keys if k in f} for f in features]
            batch = collator(input_feats)
            # For test splits the label column may be missing; set dummy values
            if is_regression:
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
                # Use -1 as dummy label when no gold is available
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
            if include_meta:
                metas: List[Dict[str, Any]] = []
                for f in features:
                    metas.append(
                        {
                            "text1": f.get(sent1_key),
                            "text2": f.get(sent2_key) if sent2_key else None,
                            "idx": f.get("idx", None),
                        }
                    )
                batch["meta"] = metas
            return batch
        return _collate

    # Determine which evaluation splits to use per task
    eval_split = eval_split.lower().strip()
    if eval_split not in ("validation", "test"):
        raise ValueError(f"Unsupported eval_split: {eval_split}. Use 'validation' or 'test'.")

    # Initialise per-client task data
    client_task_data: List[Dict[str, TaskData]] = [{} for _ in range(num_clients)]

    for task in GLUE_TASKS:
        ds_dict = load_glue_dataset_with_disk_cache(
            task,
            hf_datasets_cache_dir=hf_datasets_cache_dir,
            glue_disk_cache_dir=glue_disk_cache_dir,
            hf_token=hf_token,
            offline=offline,
        )
        label_names = extract_label_names_from_dataset(ds_dict, task)
        # Training dataset
        train_ds = ds_dict["train"].shuffle(seed=seed).map(tokenize_fn(task), batched=True)
        keep_cols = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in train_ds.column_names:
            keep_cols.append("token_type_ids")
        train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
        # Partition training data for clients
        if num_clients > 1:
            train_labels = np.array(train_ds["label"])
            num_classes = len(set(train_labels))
            client_indices = partition_indices_dirichlet(
                num_samples=len(train_ds),
                num_clients=num_clients,
                num_classes=num_classes,
                labels=train_labels,
                alpha=dirichlet_alpha,
                seed=seed,
            )
            for client_id in range(num_clients):
                client_train_indices = client_indices[client_id]
                if test_mode and len(client_train_indices) > TEST_SAMPLE_SIZE:
                    client_train_indices = client_train_indices[:TEST_SAMPLE_SIZE]
                client_train_ds = train_ds.select(client_train_indices)
                train_loader = DataLoader(
                    client_train_ds,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=make_collate_fn(task, include_meta=False),
                )
                # Build evaluation loaders
                val_loaders: List[Tuple[str, DataLoader]] = []
                # Determine split names
                if eval_split == "validation":
                    splits = ["validation_matched", "validation_mismatched"] if task == "mnli" else ["validation"]
                else:
                    splits = ["test_matched", "test_mismatched"] if task == "mnli" else ["test"]
                def make_val_loader(split_name: str):
                    val_ds = ds_dict[split_name].map(tokenize_fn(task), batched=True)
                    sent1_key, sent2_key = TASK_TO_KEYS[task]
                    keep_cols_v = ["input_ids", "attention_mask", "label"]
                    if "token_type_ids" in val_ds.column_names:
                        keep_cols_v.append("token_type_ids")
                    if save_eval_details:
                        if sent1_key in val_ds.column_names:
                            keep_cols_v.append(sent1_key)
                        if sent2_key and sent2_key in val_ds.column_names:
                            keep_cols_v.append(sent2_key)
                        if "idx" in val_ds.column_names:
                            keep_cols_v.append("idx")
                    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols_v])
                    return DataLoader(
                        val_ds,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=make_collate_fn(task, include_meta=save_eval_details),
                    )
                for split_name in splits:
                    val_loaders.append((split_name, make_val_loader(split_name)))
                client_task_data[client_id][task] = TaskData(
                    train_loader=train_loader,
                    val_loaders=val_loaders,
                    label_names=label_names,
                )
        else:
            # Single client
            train_ds_for_loader = train_ds
            if test_mode and len(train_ds) > TEST_SAMPLE_SIZE:
                train_ds_for_loader = train_ds.select(range(TEST_SAMPLE_SIZE))
            train_loader = DataLoader(
                train_ds_for_loader,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=make_collate_fn(task, include_meta=False),
            )
            val_loaders: List[Tuple[str, DataLoader]] = []
            if eval_split == "validation":
                splits = ["validation_matched", "validation_mismatched"] if task == "mnli" else ["validation"]
            else:
                splits = ["test_matched", "test_mismatched"] if task == "mnli" else ["test"]
            def make_val_loader(split_name: str):
                val_ds = ds_dict[split_name].map(tokenize_fn(task), batched=True)
                sent1_key, sent2_key = TASK_TO_KEYS[task]
                keep_cols_v = ["input_ids", "attention_mask", "label"]
                if "token_type_ids" in val_ds.column_names:
                    keep_cols_v.append("token_type_ids")
                if save_eval_details:
                    if sent1_key in val_ds.column_names:
                        keep_cols_v.append(sent1_key)
                    if sent2_key and sent2_key in val_ds.column_names:
                        keep_cols_v.append(sent2_key)
                    if "idx" in val_ds.column_names:
                        keep_cols_v.append("idx")
                val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols_v])
                return DataLoader(
                    val_ds,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=make_collate_fn(task, include_meta=save_eval_details),
                )
            if task == "mnli" and len(splits) == 2:
                for split_name in splits:
                    val_loaders.append((split_name, make_val_loader(split_name)))
            else:
                for split_name in splits:
                    val_loaders.append((split_name, make_val_loader(split_name)))
            client_task_data[0][task] = TaskData(
                train_loader=train_loader,
                val_loaders=val_loaders,
                label_names=label_names,
            )
    return client_task_data