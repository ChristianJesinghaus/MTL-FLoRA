"""Lightweight training/evaluation loops for Flower clients."""
from typing import Dict, Iterable

import torch
import transformers
from torch.utils.data import DataLoader


def build_dataloader(dataset, tokenizer, batch_size: int, cutoff_len: int):
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True, max_length=cutoff_len
    )

    def _collate(features: Iterable[Dict]):
        lambda_indices = None
        if "lambda_index" in features[0]:
            lambda_indices = torch.tensor(
                [f.get("lambda_index", 0) for f in features], dtype=torch.long
            )
        stripped = [
            {k: v for k, v in feature.items() if k != "lambda_index"}
            for feature in features
        ]
        batch = collator(stripped)
        if lambda_indices is not None:
            batch["lambda_index"] = lambda_indices
        return batch

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)


def train_one_round(model, dataloader: DataLoader, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        lambda_index = batch.pop("lambda_index", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        if lambda_index is not None:
            lambda_index = lambda_index.to(device)
        outputs = model(**batch, lambda_index=lambda_index)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            lambda_index = batch.pop("lambda_index", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if lambda_index is not None:
                lambda_index = lambda_index.to(device)
            outputs = model(**batch, lambda_index=lambda_index)
            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
            total_examples += batch["input_ids"].size(0)
    if total_examples == 0:
        return float("nan")
    return total_loss / total_examples
