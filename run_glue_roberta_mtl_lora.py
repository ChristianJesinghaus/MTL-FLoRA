#!/usr/bin/env python3
"""
Multi-task GLUE reproduction on GTX 1080 Ti with RoBERTa + MTL-LoRA.


"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset, load_from_disk

try:
    from huggingface_hub.utils import HfHubHTTPError  
except Exception:  # 
    from huggingface_hub.utils._errors import HfHubHTTPError  

try:
    from huggingface_hub import HfFolder  
except Exception:  
    HfFolder = None  

from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from tqdm.auto import tqdm

# Repo adapter (mLoRA)
from src.adapter.mlora import mLoRALinear  # noqa: E402

# Context manager to set global task id for mLoRA (required for RoBERTa/BERT-style forwards)
try:
    from src.adapter.mlora import mlora_set_lambda_index  # noqa: E402
except Exception:
    mlora_set_lambda_index = None  



def _dist_is_init() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if _dist_is_init() else 0


def get_world_size() -> int:
    return dist.get_world_size() if _dist_is_init() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if _dist_is_init():
        dist.barrier()


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize torch.distributed if launched with torchrun.
    Returns: (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed() -> None:
    if _dist_is_init():
        dist.destroy_process_group()


def tqdm_main(*args, **kwargs):
    # tqdm in Slurm logs: write to stdout and reduce update rate
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 5.0)
    kwargs.setdefault("leave", True)
    kwargs.setdefault("disable", not is_main_process())
    return tqdm(*args, **kwargs)


# GLUE task metadata

GLUE_TASKS: List[str] = [
    "cola",
    "sst2",
    "mrpc",
    "qqp",
    "mnli",
    "qnli",
    "rte",
    "stsb",
]
TASK_TO_ID: Dict[str, int] = {task: i for i, task in enumerate(GLUE_TASKS)}

TASK_TO_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

TASK_NUM_LABELS: Dict[str, int] = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "stsb": 1,  # regression
}

FALLBACK_LABEL_NAMES: Dict[str, Dict[int, str]] = {
    "cola": {0: "unacceptable", 1: "acceptable"},
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "rte": {0: "entailment", 1: "not_entailment"},
}


# Repro helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_float(x: float, ndigits: int = 6) -> float:
    try:
        return float(round(float(x), ndigits))
    except Exception:
        return float(x)


# Metrics
def matthews_corrcoef_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denom


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


def pearson_from_sums(n: float, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    # r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2)*(n*sum_y2 - sum_y^2))
    num = n * sum_xy - sum_x * sum_y
    den_x = n * sum_x2 - sum_x * sum_x
    den_y = n * sum_y2 - sum_y * sum_y
    denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
    if denom == 0.0:
        return 0.0
    return num / denom


# HF Hub / Datasets caching
def _default_hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _get_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        t = explicit.strip()
        return t or None
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    if HfFolder is not None:
        try:
            tok = HfFolder.get_token()
            if tok:
                return tok
        except Exception:
            pass
    return None


def _is_hf_429(err: Exception) -> bool:
    s = str(err)
    if "429" in s and ("Too Many Requests" in s or "Client Error" in s):
        return True
    resp = getattr(err, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code == 429:
            return True
    return False


def load_glue_dataset_with_disk_cache(
    task: str,
    *,
    hf_datasets_cache_dir: Optional[str],
    glue_disk_cache_dir: str,
    hf_token: Optional[str],
    offline: bool,
    max_retries_with_token: int = 6,
):
    """
    First tries to load GLUE/<task> from disk cache (save_to_disk).
    If missing and not offline, downloads (with token if provided) and saves to disk cache.
    """
    os.makedirs(glue_disk_cache_dir, exist_ok=True)
    disk_path = os.path.join(glue_disk_cache_dir, f"glue_{task}")

    if os.path.isdir(disk_path):
        if is_main_process():
            print(f"[INFO] GLUE/{task}: loading from disk cache: {disk_path}")
        return load_from_disk(disk_path)

    if offline:
        raise RuntimeError(
            f"[OFFLINE] GLUE/{task} is not available in disk cache ({disk_path}). "
            "Run once with internet + HF token to populate the cache, then re-run with --offline."
        )

    token = hf_token
    attempt = 0
    while True:
        try:
            # datasets>=2.14 prefers token=...
            try:
                ds = load_dataset("glue", task, cache_dir=hf_datasets_cache_dir, token=token)
            except TypeError:
                ds = load_dataset("glue", task, cache_dir=hf_datasets_cache_dir, use_auth_token=token)

            if is_main_process():
                print(f"[INFO] GLUE/{task}: saving to disk cache: {disk_path}")
            ds.save_to_disk(disk_path)
            return ds

        except HfHubHTTPError as e:
            if _is_hf_429(e):
                if not token:
                    raise RuntimeError(
                        "HuggingFace Hub returned HTTP 429 (Too Many Requests) for the shared cluster IP.\n\n"
                        "Fix:\n"
                        "  1) Create a HF account + access token (read is enough)\n"
                        "  2) `huggingface-cli login` once OR export HF_TOKEN in sbatch\n"
                        "  3) Re-run the job\n"
                    ) from e

                attempt += 1
                if attempt > max_retries_with_token:
                    raise RuntimeError(
                        f"Still getting HTTP 429 after {max_retries_with_token} retries with token. "
                        "Try again later or reduce concurrent downloads."
                    ) from e

                wait_s = min(15 * (2 ** (attempt - 1)), 300)
                if is_main_process():
                    print(f"[WARN] HF 429 for GLUE/{task}. Retry {attempt}/{max_retries_with_token} in {wait_s}s...")
                time.sleep(wait_s)
                continue

            raise


# Dataloading helpers
class EvalShardSampler(Sampler[int]):
    """Shard indices across ranks without padding (each example exactly once globally)."""

    def __init__(self, dataset_len: int, rank: int, world_size: int):
        self.dataset_len = int(dataset_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._indices = list(range(self.rank, self.dataset_len, self.world_size))

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


@dataclass
class TaskData:
    train_loader: DataLoader
    val_loaders: List[Tuple[str, DataLoader]]
    label_names: Optional[List[str]]  


def _extract_label_names_from_dataset(ds_dict, task: str) -> Optional[List[str]]:
    if task == "stsb":
        return None
    try:
        feat = ds_dict["train"].features.get("label", None)
        names = getattr(feat, "names", None)
        if isinstance(names, list) and len(names) > 0:
            return [str(x) for x in names]
    except Exception:
        pass
    # fallback
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
    rank: int,
    world_size: int,
    save_eval_details: bool,
) -> Dict[str, TaskData]:
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

        label_names = _extract_label_names_from_dataset(ds_dict, task)

        train_ds = ds_dict["train"].shuffle(seed=seed).map(tokenize_fn(task), batched=True)

        # Keep only tensors for training 
        keep_cols = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in train_ds.column_names:
            keep_cols.append("token_type_ids")
        train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])

        if world_size > 1:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(task, include_meta=False),
        )

        #  Validation split
        val_loaders: List[Tuple[str, DataLoader]] = []

        def make_val_loader(split_name: str):
            val_ds = ds_dict[split_name].map(tokenize_fn(task), batched=True)

            # Keep meta columns for evaluation-dump if requested
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

            if world_size > 1:
                # No-padding sharding to avoid duplicates in metrics & dumps
                val_sampler = EvalShardSampler(len(val_ds), rank=rank, world_size=world_size)
            else:
                val_sampler = None

            val_loader = DataLoader(
                val_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=make_collate_fn(task, include_meta=save_eval_details),
            )
            return val_loader

        if task == "mnli":
            for split_name in ["validation_matched", "validation_mismatched"]:
                val_loaders.append((split_name, make_val_loader(split_name)))
        else:
            val_loaders.append(("validation", make_val_loader("validation")))

        task_data[task] = TaskData(train_loader=train_loader, val_loaders=val_loaders, label_names=label_names)

    return task_data



# Model
def replace_roberta_linears_with_mlora(
    model: nn.Module,
    *,
    target_substrings: Tuple[str, ...],
    r: int,
    alpha: int,
    dropout: float,
    num_B: int,
    lambda_num: int,
    temperature: float,
) -> int:
    replaced = 0
    module_names = [name for name, _ in model.named_modules()]

    for name in module_names:
        if not any(s in name for s in target_substrings):
            continue

        try:
            target = model.get_submodule(name)
        except AttributeError:
            continue

        if not isinstance(target, nn.Linear):
            continue
        if isinstance(target, mLoRALinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        new_layer = mLoRALinear(
            in_features=target.in_features,
            out_features=target.out_features,
            B_num=num_B,
            lambda_num=lambda_num,
            diagonal_format=False,
            B_scale=temperature,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=(target.bias is not None),
        )
        new_layer.to(device=target.weight.device, dtype=target.weight.dtype)
        new_layer.load_state_dict(target.state_dict(), strict=False)

        setattr(parent, child_name, new_layer)
        replaced += 1

    return replaced


class MultiTaskRobertaMLoRA(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, TASK_NUM_LABELS[task]) for task in GLUE_TASKS}
        )

    def forward(self, task: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if mlora_set_lambda_index is None:
            raise RuntimeError(
                "mlora_set_lambda_index is not available, but your mLoRA implementation requires a task id.\n"
                "Please ensure src/adapter/mlora.py provides mlora_set_lambda_index."
            )

        task_id = TASK_TO_ID[task]
        with mlora_set_lambda_index(task_id):
            outputs = self.encoder(**batch)

        cls = outputs.last_hidden_state[:, 0]  # [batch, hidden]
        logits = self.heads[task](cls)
        return logits


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


# Eval prompt/dump formatting
def build_eval_prompt(task: str, text1: str, text2: Optional[str], label_names: Optional[List[str]]) -> str:
    """
    Human-readable "prompt" string for analysis.
    Note: The model is NOT trained with prompts; this is only for interpretability.
    """
    task_upper = task.upper()
    if task == "sst2":
        choices = ", ".join(label_names or ["0", "1"])
        return f"[{task_upper}] Sentiment classification\nSentence: {text1}\nChoices: {choices}\nAnswer:"
    if task == "cola":
        choices = ", ".join(label_names or ["0", "1"])
        return f"[{task_upper}] Grammatical acceptability\nSentence: {text1}\nChoices: {choices}\nAnswer:"
    if task in ("mrpc", "qqp"):
        choices = ", ".join(label_names or ["0", "1"])
        return (
            f"[{task_upper}] Paraphrase / duplicate question detection\n"
            f"Text A: {text1}\nText B: {text2}\nChoices: {choices}\nAnswer:"
        )
    if task == "mnli":
        choices = ", ".join(label_names or ["0", "1", "2"])
        return (
            f"[{task_upper}] Natural language inference\n"
            f"Premise: {text1}\nHypothesis: {text2}\nChoices: {choices}\nAnswer:"
        )
    if task in ("qnli", "rte"):
        choices = ", ".join(label_names or ["0", "1"])
        a = "Question" if task == "qnli" else "Premise"
        b = "Sentence" if task == "qnli" else "Hypothesis"
        return f"[{task_upper}] NLI\n{a}: {text1}\n{b}: {text2}\nChoices: {choices}\nAnswer:"
    if task == "stsb":
        return f"[{task_upper}] Semantic textual similarity (0..5)\nText A: {text1}\nText B: {text2}\nAnswer:"
    # fallback
    return f"[{task_upper}] Input: {text1} {text2 or ''}".strip()


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return s.replace("\r", " ").replace("\n", " ").strip()


# Checkpointing
def save_checkpoint(
    *,
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    update_step: int,
    args: argparse.Namespace,
) -> None:
    if not is_main_process():
        return

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    state = {
        "epoch": int(epoch),
        "update_step": int(update_step),
        "args": vars(args),
        "adapter_state": {k: v.detach().cpu() for k, v in unwrap_model(model).encoder.state_dict().items() if "lora_" in k},
        "heads_state": {k: v.detach().cpu() for k, v in unwrap_model(model).heads.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "scaler": scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None,
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, ckpt_path)
    print(f"[CKPT] saving {ckpt_path}", flush=True)


def maybe_rotate_checkpoints(ckpt_dir: str, keep_last: int) -> None:
    if not is_main_process():
        return
    if keep_last <= 0:
        return
    p = Path(ckpt_dir)
    if not p.exists():
        return
    ckpts = sorted([x for x in p.glob("ckpt_*.pt")], key=lambda x: x.stat().st_mtime)
    while len(ckpts) > keep_last:
        old = ckpts.pop(0)
        try:
            old.unlink()
            print(f"[CKPT] removed old checkpoint: {old.name}", flush=True)
        except Exception as e:
            print(f"[WARN] could not remove old checkpoint {old}: {e}", flush=True)
            break


def load_from_checkpoint(
    ckpt_path: str,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict_heads: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    adapter_state = ckpt.get("adapter_state", {})
    heads_state = ckpt.get("heads_state", {})

    missing, unexpected = unwrap_model(model).encoder.load_state_dict(adapter_state, strict=False)
    if is_main_process():
        if missing:
            print(f"[CKPT] adapter missing keys (ignored): {len(missing)}")
        if unexpected:
            print(f"[CKPT] adapter unexpected keys (ignored): {len(unexpected)}")

    unwrap_model(model).heads.load_state_dict(heads_state, strict=strict_heads)

    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    return ckpt


# Evaluation (metrics + optional per-example dumps)
@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    task_data: Dict[str, TaskData],
    device: torch.device,
    use_amp: bool,
    output_dir: str,
    tag: str,
    save_details: bool,
    details_max_examples: int,
    details_only_errors: bool,
    details_topk: int,
    stsb_abs_err_threshold: float,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    rank = get_rank()
    world_size = get_world_size()

    details_dir = os.path.join(output_dir, "eval_details")
    if save_details:
        os.makedirs(details_dir, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}

    for task in GLUE_TASKS:
        td = task_data[task]
        label_names = td.label_names

        task_results: Dict[str, float] = {}

        for split_name, loader in td.val_loaders:
            #  Metrics accumulators 
            if task == "stsb":
                n = torch.zeros((1,), device=device, dtype=torch.float64)
                sum_x = torch.zeros((1,), device=device, dtype=torch.float64)
                sum_y = torch.zeros((1,), device=device, dtype=torch.float64)
                sum_x2 = torch.zeros((1,), device=device, dtype=torch.float64)
                sum_y2 = torch.zeros((1,), device=device, dtype=torch.float64)
                sum_xy = torch.zeros((1,), device=device, dtype=torch.float64)
            elif task in ("mrpc", "qqp"):
                correct = torch.zeros((1,), device=device, dtype=torch.int64)
                total = torch.zeros((1,), device=device, dtype=torch.int64)
                tp = torch.zeros((1,), device=device, dtype=torch.int64)
                fp = torch.zeros((1,), device=device, dtype=torch.int64)
                fn = torch.zeros((1,), device=device, dtype=torch.int64)
            elif task == "cola":
                tp = torch.zeros((1,), device=device, dtype=torch.int64)
                tn = torch.zeros((1,), device=device, dtype=torch.int64)
                fp = torch.zeros((1,), device=device, dtype=torch.int64)
                fn = torch.zeros((1,), device=device, dtype=torch.int64)
            else:
                correct = torch.zeros((1,), device=device, dtype=torch.int64)
                total = torch.zeros((1,), device=device, dtype=torch.int64)

            #  Optional details file 
            details_fh = None
            wrote = 0
            if save_details:
                # interpret max_examples as GLOBAL cap; split across ranks
                if details_max_examples < 0:
                    per_rank_cap = -1
                else:
                    per_rank_cap = int(math.ceil(details_max_examples / max(1, world_size)))

                details_path_rank = os.path.join(details_dir, f"{tag}_{task}_{split_name}.rank{rank}.jsonl")
                details_fh = open(details_path_rank, "w", encoding="utf-8")
                header = {
                    "_meta": {
                        "task": task,
                        "split": split_name,
                        "tag": tag,
                        "rank": rank,
                        "world_size": world_size,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "format": "jsonl",
                    }
                }
                details_fh.write(json.dumps(header) + "\n")
                details_fh.flush()

            try:
                pbar = tqdm_main(loader, desc=f"eval {task}:{split_name}")

                for batch in pbar:
                    labels = batch.pop("labels").to(device)

                    meta_list = batch.pop("meta", None)  # list[dict] or None
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(task=task, batch=batch)

                    if task == "stsb":
                        preds = logits.squeeze(-1).float()

                        # Pearson stats
                        x = preds.to(torch.float64)
                        y = labels.to(torch.float64)
                        n += torch.tensor([x.numel()], device=device, dtype=torch.float64)
                        sum_x += x.sum()
                        sum_y += y.sum()
                        sum_x2 += (x * x).sum()
                        sum_y2 += (y * y).sum()
                        sum_xy += (x * y).sum()

                        # details
                        if details_fh is not None and meta_list is not None:
                            for i in range(len(meta_list)):
                                if per_rank_cap >= 0 and wrote >= per_rank_cap:
                                    break

                                text1 = _safe_text(meta_list[i].get("text1"))
                                text2 = _safe_text(meta_list[i].get("text2"))
                                idx_val = meta_list[i].get("idx")

                                gold = float(labels[i].detach().cpu().item())
                                pred = float(preds[i].detach().cpu().item())
                                abs_err = abs(pred - gold)
                                if details_only_errors and abs_err < stsb_abs_err_threshold:
                                    continue

                                record = {
                                    "task": task,
                                    "split": split_name,
                                    "idx": idx_val,
                                    "text1": text1,
                                    "text2": text2,
                                    "prompt": build_eval_prompt(task, text1, text2, label_names),
                                    "gold": gold,
                                    "pred": pred,
                                    "abs_error": _as_float(abs_err, 6),
                                }
                                details_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                                wrote += 1

                            if wrote and (wrote % 50 == 0):
                                details_fh.flush()

                    else:
                        preds = torch.argmax(logits, dim=-1)

                        # metrics
                        if task in ("mrpc", "qqp"):
                            correct += (preds == labels).sum().to(torch.int64)
                            total += torch.tensor([labels.numel()], device=device, dtype=torch.int64)
                            tp += ((preds == 1) & (labels == 1)).sum().to(torch.int64)
                            fp += ((preds == 1) & (labels == 0)).sum().to(torch.int64)
                            fn += ((preds == 0) & (labels == 1)).sum().to(torch.int64)
                        elif task == "cola":
                            tp += ((preds == 1) & (labels == 1)).sum().to(torch.int64)
                            tn += ((preds == 0) & (labels == 0)).sum().to(torch.int64)
                            fp += ((preds == 1) & (labels == 0)).sum().to(torch.int64)
                            fn += ((preds == 0) & (labels == 1)).sum().to(torch.int64)
                        else:
                            correct += (preds == labels).sum().to(torch.int64)
                            total += torch.tensor([labels.numel()], device=device, dtype=torch.int64)

                        # details
                        if details_fh is not None and meta_list is not None:
                            probs = torch.softmax(logits.float(), dim=-1)
                            topk = min(int(details_topk), probs.shape[-1])

                            top_vals, top_idx = torch.topk(probs, k=topk, dim=-1)
                            top_vals = top_vals.detach().cpu()
                            top_idx = top_idx.detach().cpu()
                            probs_cpu = probs.detach().cpu()

                            for i in range(len(meta_list)):
                                if per_rank_cap >= 0 and wrote >= per_rank_cap:
                                    break

                                text1 = _safe_text(meta_list[i].get("text1"))
                                text2 = _safe_text(meta_list[i].get("text2"))
                                idx_val = meta_list[i].get("idx")

                                gold_id = int(labels[i].detach().cpu().item())
                                pred_id = int(preds[i].detach().cpu().item())
                                correct_flag = bool(pred_id == gold_id)
                                if details_only_errors and correct_flag:
                                    continue

                                gold_name = None
                                pred_name = None
                                if label_names is not None:
                                    if 0 <= gold_id < len(label_names):
                                        gold_name = label_names[gold_id]
                                    if 0 <= pred_id < len(label_names):
                                        pred_name = label_names[pred_id]

                                prob_list = [_as_float(x, 6) for x in probs_cpu[i].tolist()]
                                top_list = []
                                for k in range(topk):
                                    lid = int(top_idx[i, k].item())
                                    lname = (
                                        label_names[lid]
                                        if (label_names is not None and 0 <= lid < len(label_names))
                                        else str(lid)
                                    )
                                    top_list.append({"label_id": lid, "label_name": lname, "p": _as_float(top_vals[i, k].item(), 6)})

                                record = {
                                    "task": task,
                                    "split": split_name,
                                    "idx": idx_val,
                                    "text1": text1,
                                    "text2": text2,
                                    "prompt": build_eval_prompt(task, text1, text2, label_names),
                                    "gold": {"id": gold_id, "name": gold_name},
                                    "pred": {"id": pred_id, "name": pred_name},
                                    "correct": correct_flag,
                                    "probs": prob_list,
                                    "topk": top_list,
                                }
                                details_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                                wrote += 1

                            if wrote and (wrote % 50 == 0):
                                details_fh.flush()

            finally:
                if details_fh is not None:
                    details_fh.flush()
                    details_fh.close()

            #  Reduce metrics across ranks 
            if _dist_is_init():
                if task == "stsb":
                    for t in (n, sum_x, sum_y, sum_x2, sum_y2, sum_xy):
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                elif task in ("mrpc", "qqp"):
                    for t in (correct, total, tp, fp, fn):
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                elif task == "cola":
                    for t in (tp, tn, fp, fn):
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                else:
                    for t in (correct, total):
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)

            #  Compute split metrics on rank0 
            if is_main_process():
                if task == "cola":
                    mcc = matthews_corrcoef_from_counts(int(tp.item()), int(tn.item()), int(fp.item()), int(fn.item()))
                    task_results[f"{split_name}_mcc"] = float(mcc)
                elif task == "stsb":
                    r = pearson_from_sums(
                        float(n.item()),
                        float(sum_x.item()),
                        float(sum_y.item()),
                        float(sum_x2.item()),
                        float(sum_y2.item()),
                        float(sum_xy.item()),
                    )
                    task_results[f"{split_name}_pearson"] = float(r)
                elif task in ("mrpc", "qqp"):
                    acc = float(correct.item()) / max(1.0, float(total.item()))
                    f1 = f1_from_counts(int(tp.item()), int(fp.item()), int(fn.item()))
                    task_results[f"{split_name}_acc"] = float(acc)
                    task_results[f"{split_name}_f1"] = float(f1)
                else:
                    acc = float(correct.item()) / max(1.0, float(total.item()))
                    task_results[f"{split_name}_acc"] = float(acc)

        # MNLI average
        if is_main_process() and task == "mnli":
            m = task_results.get("validation_matched_acc", None)
            mm = task_results.get("validation_mismatched_acc", None)
            if m is not None and mm is not None:
                task_results["validation_avg_acc"] = 0.5 * (float(m) + float(mm))

        results[task] = task_results

    # GLUE avg
    if is_main_process():
        avg_scores: List[float] = []
        for task in GLUE_TASKS:
            tr = results.get(task, {})
            if task == "mnli":
                avg_scores.append(float(tr.get("validation_avg_acc", tr.get("validation_matched_acc", 0.0))))
            elif task == "stsb":
                avg_scores.append(float(tr.get("validation_pearson", 0.0)))
            elif task == "cola":
                avg_scores.append(float(tr.get("validation_mcc", 0.0)))
            else:
                avg_scores.append(float(tr.get("validation_acc", 0.0)))

        results["glue_avg"] = {"avg": float(sum(avg_scores) / len(avg_scores))}

    # Merge details shards (rank0) after all ranks finished writing.
    if save_details and world_size > 1:
        barrier()
        if is_main_process():
            for task in GLUE_TASKS:
                for split_name, _ in task_data[task].val_loaders:
                    merged_path = os.path.join(details_dir, f"{tag}_{task}_{split_name}.jsonl")
                    with open(merged_path, "w", encoding="utf-8") as out_f:
                        for r in range(world_size):
                            part_path = os.path.join(details_dir, f"{tag}_{task}_{split_name}.rank{r}.jsonl")
                            if not os.path.exists(part_path):
                                continue
                            with open(part_path, "r", encoding="utf-8") as in_f:
                                for line_idx, line in enumerate(in_f):
                                    if r != 0 and line_idx == 0:
                                        continue
                                    out_f.write(line)
        barrier()

    model.train()
    return results


# Main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Model / output
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--output_dir", type=str, default="./outputs_roberta_glue_mlora")
    p.add_argument("--seed", type=int, default=42)

    # Modes
    p.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation.")
    p.add_argument("--load_dir", type=str, default=None, help="Directory containing adapter_state*.pt + heads_state*.pt to load.")
    p.add_argument("--load_adapter", type=str, default=None, help="Path to adapter_state*.pt")
    p.add_argument("--load_heads", type=str, default=None, help="Path to heads_state*.pt")
    p.add_argument("--load_ckpt", type=str, default=None, help="Path to a full training checkpoint ckpt_*.pt (optional).")

    # Training hyperparams (defaults tuned for 1080 Ti)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    # Mixed precision
    p.add_argument("--fp32", action="store_true", help="Enable FP32 autocast (recommended on 1080 Ti).")

    # LoRA / mLoRA hyperparams
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_B", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)

    # Checkpointing
    p.add_argument("--save_steps", type=int, default=2500, help="Save training checkpoint every N update steps (rank0 only).")
    p.add_argument("--save_total_limit", type=int, default=2, help="Keep only the last N checkpoints in output_dir/checkpoints.")
    p.add_argument("--save_pre_eval_ckpt", action="store_true", help="Always save a checkpoint right before evaluation.")
    p.add_argument("--resume_from_ckpt", type=str, default=None, help="Resume training from a checkpoint ckpt_*.pt")

    # HF / dataset cache
    p.add_argument("--offline", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--glue_disk_cache_dir", type=str, default=None)
    p.add_argument("--hf_datasets_cache_dir", type=str, default=None)

    # Eval details dump
    p.add_argument("--save_eval_details", action="store_true", help="Write per-example JSONL with prompt + prediction for analysis.")
    p.add_argument("--eval_details_max_examples", type=int, default=200, help="Max examples per split to save (global). Use -1 for all.")
    p.add_argument("--eval_details_only_errors", action="store_true", help="Save only wrong predictions (classification) / large error (stsb).")
    p.add_argument("--eval_details_topk", type=int, default=2, help="Top-k predictions to store for classification tasks.")
    p.add_argument("--stsb_abs_err_threshold", type=float, default=0.5, help="When --eval_details_only_errors, keep STSB examples with abs error >= threshold.")

    return p.parse_args()


def resolve_load_paths(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve (adapter_path, heads_path) from args.load_*.
    """
    adapter_path = args.load_adapter
    heads_path = args.load_heads

    if (adapter_path and not heads_path) or (heads_path and not adapter_path):
        raise ValueError("Please provide both --load_adapter and --load_heads, or use --load_dir.")

    if adapter_path and heads_path:
        return adapter_path, heads_path

    if args.load_dir:
        d = Path(args.load_dir)
        cand_adapter = [d / "adapter_state_last.pt", d / "adapter_state.pt"]
        cand_heads = [d / "heads_state_last.pt", d / "heads_state.pt"]
        ap = next((str(p) for p in cand_adapter if p.exists()), None)
        hp = next((str(p) for p in cand_heads if p.exists()), None)
        if ap is None or hp is None:
            raise FileNotFoundError(
                f"Could not find adapter/head weights in {args.load_dir}. Expected one of: "
                f"{[str(x) for x in cand_adapter]} and {[str(x) for x in cand_heads]}"
            )
        return ap, hp

    return None, None


def main() -> None:
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()

    if is_main_process():
        print(f"[INFO] rank={rank} world_size={world_size} local_rank={local_rank}")
        print(f"[INFO] output_dir={args.output_dir}")
        print(f"[INFO] eval_only={args.eval_only}")

    # Create output dir on rank0, then sync
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    barrier()

    # Make randomness deterministic but different per-rank
    set_seed(args.seed + rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    use_amp = bool(args.fp32 and device.type == "cuda")

    # HF config
    hf_token = _get_hf_token(args.hf_token)
    hf_home = _default_hf_home()
    glue_disk_cache_dir = args.glue_disk_cache_dir or os.path.join(hf_home, "glue_disk_cache")
    hf_datasets_cache_dir = args.hf_datasets_cache_dir or os.environ.get("HF_DATASETS_CACHE") or None

    if is_main_process():
        print(f"[INFO] device={device} use_amp={use_amp}")
        print(f"[INFO] HF_HOME={hf_home}")
        print(f"[INFO] glue_disk_cache_dir={glue_disk_cache_dir}")
        print(f"[INFO] hf_datasets_cache_dir={hf_datasets_cache_dir}")
        print(f"[INFO] hf_token_present={'yes' if hf_token else 'no'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, local_files_only=args.offline)

    # Load encoder in FP32
    try:
        encoder = AutoModel.from_pretrained(
            args.model_name,
            add_pooling_layer=False,
            local_files_only=args.offline,
        )
    except TypeError:
        encoder = AutoModel.from_pretrained(args.model_name, local_files_only=args.offline)

    encoder.to(device)

    # Inject mLoRA into attention linears
    target_substrings = (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    )
    replaced = replace_roberta_linears_with_mlora(
        encoder,
        target_substrings=target_substrings,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        num_B=args.num_B,
        lambda_num=len(GLUE_TASKS),
        temperature=args.temperature,
    )
    if is_main_process():
        print(f"[INFO] Replaced {replaced} Linear layers with mLoRALinear")

    model = MultiTaskRobertaMLoRA(encoder=encoder, hidden_size=encoder.config.hidden_size).to(device)

    # Freeze base params, train only LoRA + heads
    for name, param in model.named_parameters():
        if ("lora_" in name) or name.startswith("heads."):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Ensure trainable params are FP32 so GradScaler works.
    for name, p_ in model.named_parameters():
        if p_.requires_grad and p_.dtype != torch.float32:
            if is_main_process():
                print(f"[WARN] Casting trainable param to fp32: {name} ({p_.dtype} -> fp32)")
            p_.data = p_.data.float()

    trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    total = sum(p_.numel() for p_ in model.parameters())
    if is_main_process():
        dtypes_trainable = sorted({str(p_.dtype) for p_ in model.parameters() if p_.requires_grad})
        print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
        print(f"[INFO] Trainable dtypes: {dtypes_trainable}")

    # DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Data
    task_data = build_dataloaders(
        tokenizer,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        hf_datasets_cache_dir=hf_datasets_cache_dir,
        glue_disk_cache_dir=glue_disk_cache_dir,
        hf_token=hf_token,
        offline=args.offline,
        rank=rank,
        world_size=world_size,
        save_eval_details=args.save_eval_details,
    )

    # Optimizer / scheduler
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0)

    steps_per_epoch = sum(len(td.train_loader) for td in task_data.values())
    total_updates = int(math.ceil((steps_per_epoch * args.epochs) / max(1, args.grad_accum_steps)))
    warmup_steps = int(total_updates * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    # GradScaler enabled only if AMP and trainable params are not fp16
    trainable_has_fp32 = any((p_.requires_grad and p_.dtype == torch.float16) for p_ in model.parameters())
    use_scaler = bool(use_amp and (not trainable_has_fp32))
    if is_main_process() and use_amp and not use_scaler:
        print("[WARN] AMP requested but trainable params are fp32 -> disabling GradScaler to avoid crash.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if device.type == "cuda" else None

    # Resume / load for eval
    start_epoch = 0
    global_update = 0

    if args.resume_from_ckpt:
        if is_main_process():
            print(f"[CKPT] Resuming training from {args.resume_from_ckpt}")
        barrier()
        ckpt = load_from_checkpoint(args.resume_from_ckpt, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = int(ckpt.get("epoch", 0))
        global_update = int(ckpt.get("update_step", 0))

    if args.eval_only:
        # Load weights (adapter+heads or full checkpoint)
        if args.load_ckpt:
            if is_main_process():
                print(f"[CKPT] Loading model weights from checkpoint {args.load_ckpt} for eval_only")
            barrier()
            load_from_checkpoint(args.load_ckpt, model=model, optimizer=None, scheduler=None, scaler=None, strict_heads=True)
        else:
            adapter_path, heads_path = resolve_load_paths(args)
            if adapter_path is None or heads_path is None:
                raise ValueError("For --eval_only you must provide --load_dir or --load_adapter+--load_heads (or --load_ckpt).")

            if is_main_process():
                print(f"[LOAD] adapter={adapter_path}")
                print(f"[LOAD] heads={heads_path}")

            adapter_state = torch.load(adapter_path, map_location="cpu")
            missing, unexpected = unwrap_model(model).encoder.load_state_dict(adapter_state, strict=False)
            if is_main_process():
                if missing:
                    print(f"[LOAD] adapter missing keys (ignored): {len(missing)}")
                if unexpected:
                    print(f"[LOAD] adapter unexpected keys (ignored): {len(unexpected)}")
            heads_state = torch.load(heads_path, map_location="cpu")
            unwrap_model(model).heads.load_state_dict(heads_state, strict=True)

        for p_ in model.parameters():
            p_.requires_grad = False

        barrier()
        results = evaluate(
            model=model,
            task_data=task_data,
            device=device,
            use_amp=use_amp,
            output_dir=args.output_dir,
            tag="eval_only",
            save_details=args.save_eval_details,
            details_max_examples=args.eval_details_max_examples,
            details_only_errors=args.eval_details_only_errors,
            details_topk=args.eval_details_topk,
            stsb_abs_err_threshold=args.stsb_abs_err_threshold,
        )
        if is_main_process():
            print("[eval_only] results:")
            print(json.dumps(results, indent=2))
            with open(os.path.join(args.output_dir, "eval_only_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            # optional convenience file
            with open(os.path.join(args.output_dir, "eval_latest.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        cleanup_distributed()
        return

    # ---------------- Training loop ----------------
    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Ensure DistributedSamplers shuffle deterministically per epoch
        for td in task_data.values():
            sampler = td.train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(args.seed + epoch)

        task_iters = {task: iter(task_data[task].train_loader) for task in GLUE_TASKS}

        # Build and shuffle schedule deterministically (same across ranks)
        schedule: List[str] = []
        for task in GLUE_TASKS:
            schedule.extend([task] * len(task_data[task].train_loader))
        rng = random.Random(args.seed + epoch)
        rng.shuffle(schedule)

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm_main(schedule, desc=f"train epoch {epoch+1}/{args.epochs}")
        micro_step = 0

        for task in pbar:
            micro_step += 1
            batch = next(task_iters[task])
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            do_step = (micro_step % max(1, args.grad_accum_steps) == 0)
            sync_ctx = nullcontext()
            if world_size > 1 and hasattr(model, "no_sync") and not do_step:
                sync_ctx = model.no_sync()  # type: ignore

            with sync_ctx:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(task=task, batch=batch)
                    if task == "stsb":
                        loss = F.mse_loss(logits.squeeze(-1).float(), labels.float())
                    else:
                        loss = F.cross_entropy(logits.float(), labels)
                    loss = loss / max(1, args.grad_accum_steps)

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if do_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_update += 1

                if is_main_process():
                    lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.learning_rate
                    elapsed_min = (time.time() - start_time) / 60.0
                    pbar.set_postfix(
                        upd=f"{global_update}/{total_updates}",
                        task=task,
                        lr=f"{lr:.2e}",
                        loss=f"{loss.item()*max(1,args.grad_accum_steps):.4f}",
                        min=f"{elapsed_min:.1f}",
                    )

                if args.save_steps > 0 and (global_update % args.save_steps == 0):
                    ckpt_path = os.path.join(args.output_dir, "checkpoints", f"ckpt_train_step{global_update}.pt")
                    save_checkpoint(
                        ckpt_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        update_step=global_update,
                        args=args,
                    )
                    maybe_rotate_checkpoints(os.path.join(args.output_dir, "checkpoints"), keep_last=args.save_total_limit)

        # Save pre-eval checkpoint
        if args.save_pre_eval_ckpt:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"ckpt_epoch{epoch+1}_pre_eval_step{global_update}.pt")
            save_checkpoint(
                ckpt_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                update_step=global_update,
                args=args,
            )
            maybe_rotate_checkpoints(os.path.join(args.output_dir, "checkpoints"), keep_last=args.save_total_limit)

        barrier()

        # Evaluation
        results = evaluate(
            model=model,
            task_data=task_data,
            device=device,
            use_amp=use_amp,
            output_dir=args.output_dir,
            tag=f"epoch{epoch+1}",
            save_details=args.save_eval_details,
            details_max_examples=args.eval_details_max_examples,
            details_only_errors=args.eval_details_only_errors,
            details_topk=args.eval_details_topk,
            stsb_abs_err_threshold=args.stsb_abs_err_threshold,
        )

        if is_main_process():
            print(f"[eval] epoch={epoch+1}/{args.epochs}")
            print(json.dumps(results, indent=2))

            with open(os.path.join(args.output_dir, f"eval_epoch_{epoch+1}.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            with open(os.path.join(args.output_dir, "eval_latest.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            adapter_state = {k: v.detach().cpu() for k, v in unwrap_model(model).encoder.state_dict().items() if "lora_" in k}
            heads_state = {k: v.detach().cpu() for k, v in unwrap_model(model).heads.state_dict().items()}

            torch.save(adapter_state, os.path.join(args.output_dir, "adapter_state.pt"))
            torch.save(heads_state, os.path.join(args.output_dir, "heads_state.pt"))
            torch.save(adapter_state, os.path.join(args.output_dir, "adapter_state_last.pt"))
            torch.save(heads_state, os.path.join(args.output_dir, "heads_state_last.pt"))

            with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)

            print(f"[INFO] Saved adapter/head weights to {args.output_dir}", flush=True)

        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
