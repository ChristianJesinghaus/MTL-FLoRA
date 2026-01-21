from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .constants import FALLBACK_LABEL_NAMES, GLUE_TASKS, TASK_TO_KEYS, TEST_SAMPLE_SIZE
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
    
    # Get indices grouped by class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    
    # Initialize client index lists
    client_indices: List[np.ndarray] = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples using Dirichlet
    for class_idx, indices in enumerate(class_indices):
        # Sample from Dirichlet to get proportion for each client
        proportions = rng.dirichlet([alpha] * num_clients)
        
        # Shuffle indices for this class
        rng.shuffle(indices)
        
        # Split indices according to proportions
        splits = (np.cumsum(proportions) * len(indices)).astype(int)
        class_splits = np.split(indices, splits[:-1])
        
        # Assign to clients
        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split)
    
    # Convert to numpy arrays
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
) -> List[Dict[str, TaskData]]:
    """Create per-task dataloaders, optionally split across federated clients.

    This is single-GPU only: no DistributedSampler / sharding.
    
    Args:
        num_clients: Number of federated clients. If > 1, uses Dirichlet partitioning.
        dirichlet_alpha: Dirichlet concentration parameter (lower = more non-IID).
                        Only used when num_clients > 1.
        test_mode: If True, limit training data to 50 samples per client per task.
    
    Returns:
        If num_clients == 1: List with single dict mapping task names to TaskData
        If num_clients > 1: List of dicts, one per client, each mapping task names to TaskData
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
            labels = [f["label"] for f in features]

            model_keys = ["input_ids", "attention_mask"]
            if "token_type_ids" in features[0]:
                model_keys.append("token_type_ids")

            input_feats = [{k: f[k] for k in model_keys if k in f} for f in features]
            batch = collator(input_feats)

            if is_regression:
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
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

    # Initialize list of client task data dicts
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

        # -----------------
        # Train
        # -----------------
        train_ds = ds_dict["train"].shuffle(seed=seed).map(tokenize_fn(task), batched=True)

        keep_cols = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in train_ds.column_names:
            keep_cols.append("token_type_ids")
        train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])

        # Partition training data if using multiple clients
        if num_clients > 1:
            # Get labels for Dirichlet partitioning
            train_labels = np.array(train_ds["label"])
            num_classes = len(set(train_labels))
            
            # Partition indices using Dirichlet distribution
            client_indices = partition_indices_dirichlet(
                num_samples=len(train_ds),
                num_clients=num_clients,
                num_classes=num_classes,
                labels=train_labels,
                alpha=dirichlet_alpha,
                seed=seed,
            )
            
            # Create per-client train loaders
            for client_id in range(num_clients):
                client_train_indices = client_indices[client_id]
                
                # In test mode, limit to TEST_SAMPLE_SIZE samples per client per task
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
                
                # Create per-client validation loaders
                val_loaders: List[Tuple[str, DataLoader]] = []

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

                if task == "mnli":
                    for split_name in ["validation_matched", "validation_mismatched"]:
                        val_loaders.append((split_name, make_val_loader(split_name)))
                else:
                    val_loaders.append(("validation", make_val_loader("validation")))

                client_task_data[client_id][task] = TaskData(
                    train_loader=train_loader, val_loaders=val_loaders, label_names=label_names
                )
        else:
            # Single client case: shared dataloaders for all tasks
            train_ds_for_loader = train_ds
            
            # In test mode, limit to TEST_SAMPLE_SIZE samples per client per task
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

            # -----------------
            # Validation
            # -----------------
            val_loaders: List[Tuple[str, DataLoader]] = []

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

            if task == "mnli":
                for split_name in ["validation_matched", "validation_mismatched"]:
                    val_loaders.append((split_name, make_val_loader(split_name)))
            else:
                val_loaders.append(("validation", make_val_loader("validation")))

            client_task_data[0][task] = TaskData(train_loader=train_loader, val_loaders=val_loaders, label_names=label_names)

    return client_task_data
