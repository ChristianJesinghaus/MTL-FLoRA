from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .constants import FALLBACK_LABEL_NAMES, GLUE_TASKS, TASK_TO_KEYS
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
) -> Dict[str, TaskData]:
    """Create per-task dataloaders.

    This is single-GPU only: no DistributedSampler / sharding.
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

    task_data: Dict[str, TaskData] = {}

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

        train_loader = DataLoader(
            train_ds,
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

        task_data[task] = TaskData(train_loader=train_loader, val_loaders=val_loaders, label_names=label_names)

    return task_data
