#!/usr/bin/env python3
"""
Train TinyLlama + (MTL-)mLoRA on multi-task GLUE (single GPU).

This version keeps lora_alpha FIXED (no dynamic alpha scaling), and adds ONE optional feature:
    - linear_freeze growth for FLoRA stacking across multiple FL rounds

Why:
- The default FLoRA-style stacking in this repo copies the *entire* global adapter to each client
  every round. With multiple rounds this leads to "exponential" growth (rank and B count multiply
  by num_clients each round).
- With --flora_growth linear_freeze, clients keep the aggregated (old) blocks frozen and only
  train a newly appended block each round. The server then appends/stacks only these new blocks.
  This produces linear growth in rank/B over rounds.

Important:
- lora_alpha stays constant (as in your original code). Because mLoRA uses scaling = lora_alpha / r,
  the effective adapter strength still decreases as r grows, but in linear_freeze it shrinks ~1/round
  rather than ~1/(K^rounds).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List

import logging
import torch

# Suppress tokenization warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.roberta_glue_mtl_mlora.data import build_dataloaders
from src.roberta_glue_mtl_mlora.hf_utils import default_hf_home, get_hf_token
from src.roberta_glue_mtl_mlora.train_loop import train
from src.roberta_glue_mtl_mlora.checkpoint import save_checkpoint, save_adapter_and_heads
from src.roberta_glue_mtl_mlora.utils import set_seed

from tinyllama_glue_mtl_mlora.constants import GLUE_TASKS
from tinyllama_glue_mtl_mlora.factory import create_model, create_tokenizer
from tinyllama_glue_mtl_mlora.model import cast_trainable_params_to_fp32, set_trainable_params

# Import the mLoRA layer so we can set block_size later
try:
    from src.adapter.mlora import mLoRALinear as _mLoRALinear  # type: ignore
except Exception:
    _mLoRALinear = None  # type: ignore


# Federated Averaging helpers

def fed_avg(client_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average LoRA weight dictionaries across clients (simple mean)."""
    if not client_weights:
        raise ValueError("fed_avg: No client weights provided")
    num_clients = len(client_weights)
    avg = {k: v.clone() for k, v in client_weights[0].items()}
    for key in avg.keys():
        for i in range(1, num_clients):
            avg[key] += client_weights[i][key]
        avg[key] = avg[key] / num_clients
    return avg


def fed_avg_heads(client_heads: List[Dict[str, torch.Tensor]], client_p: List[float]) -> Dict[str, torch.Tensor]:
    """
    Weighted aggregation of classification head parameters across clients.

    NOTE:
    - This function performs a weighted SUM (not dividing by sum(client_p)).
    - If you want a true weighted average, provide client_p already normalized to sum to 1.
    """
    if not client_heads:
        raise ValueError("fed_avg_heads: No client head weights provided")
    num_clients = len(client_heads)
    if len(client_p) != num_clients:
        raise ValueError("fed_avg_heads: client_p length must match number of clients")
    avg: Dict[str, torch.Tensor] = {}
    for name in client_heads[0]:
        tensor_shape = client_heads[0][name].shape
        dtype = client_heads[0][name].dtype
        weighted_sum = torch.zeros(tensor_shape, dtype=dtype)
        for i in range(num_clients):
            weighted_sum += client_p[i] * client_heads[i][name].to(dtype=dtype)
        avg[name] = weighted_sum
    return avg



# Existing (exponential) FLoRA stacking aggregation functions

def stack_A(client_A: List[Dict[str, torch.Tensor]], client_p: List[float], hidden: int, lora_r: int) -> Dict[str, torch.Tensor]:
    """Stack A matrices from clients along the LoRA rank dimension with weighting."""
    device = next(iter(client_A[0].values())).device
    num_clients = len(client_A)
    stacked: Dict[str, torch.Tensor] = {}
    for layer in client_A[0]:
        stacked[layer] = torch.cat([client_p[i] * client_A[i][layer] for i in range(num_clients)], dim=1).to(device)
    return stacked


def stack_B(client_B: List[Dict[str, torch.Tensor]], num_B: int, hidden: int, lora_r: int) -> Dict[str, torch.Tensor]:
    """
    Aggregate lora_B matrices from clients along both the B and rank dimensions.

    client_B[i][layer] has shape (num_B_local, hidden, r_i)
    output has shape (sum_B, hidden, lora_r), embedding each local B into its own rank slice.
    """
    num_clients = len(client_B)
    device = next(iter(client_B[0].values())).device
    dtype = next(iter(client_B[0].values())).dtype
    stacked: Dict[str, torch.Tensor] = {}
    for layer in client_B[0]:
        local_hidden = client_B[0][layer].shape[1]
        total_B = sum(client_B[i][layer].shape[0] for i in range(num_clients))
        aggregated = torch.zeros(total_B, local_hidden, lora_r, device=device, dtype=dtype)
        b_idx = 0
        r_offset = 0
        for i in range(num_clients):
            local_B = client_B[i][layer]
            r_i = local_B.shape[2]
            for b_local in range(local_B.shape[0]):
                aggregated[b_idx, :, r_offset : r_offset + r_i] = local_B[b_local]
                b_idx += 1
            r_offset += r_i
        stacked[layer] = aggregated
    return stacked


def stack_lambdas(client_lambdas: List[Dict[str, torch.Tensor]], num_tasks: int, lora_r: int) -> Dict[str, torch.Tensor]:
    """Stack Lambda matrices from clients into a block-diagonal tensor."""
    device = next(iter(client_lambdas[0].values())).device
    dtype = next(iter(client_lambdas[0].values())).dtype
    num_clients = len(client_lambdas)
    stacked: Dict[str, torch.Tensor] = {
        key: torch.zeros((num_tasks, lora_r, lora_r), dtype=dtype, device=device)
        for key in client_lambdas[0]
    }
    for layer in client_lambdas[0]:
        lambdas = [client_lambdas[i][layer] for i in range(num_clients)]
        sizes = [l.shape[1] for l in lambdas]
        offset = 0
        for lam, r in zip(lambdas, sizes):
            stacked[layer][:, offset : offset + r, offset : offset + r] = lam.to(device)
            offset += r
    return stacked


def stack_B_w(client_B_w: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate lora_B_w across clients along the B dimension."""
    num_clients = len(client_B_w)
    stacked: Dict[str, torch.Tensor] = {}
    for layer in client_B_w[0]:
        stacked[layer] = torch.cat([client_B_w[i][layer] for i in range(num_clients)], dim=1)
    return stacked


def aggregate_mtl_weights(
    client_weights: List[Dict[str, torch.Tensor]],
    client_p: List[float],
    hidden: int,
    num_B: int,
    num_tasks: int,
    lora_r: int,
) -> Dict[str, torch.Tensor]:
    """Aggregate per-client LoRA weights for FLoRA by stacking and concatenation (exponential growth behaviour)."""
    client_A: List[Dict[str, torch.Tensor]] = []
    client_B: List[Dict[str, torch.Tensor]] = []
    client_lambdas: List[Dict[str, torch.Tensor]] = []
    client_B_w: List[Dict[str, torch.Tensor]] = []
    for weights in client_weights:
        client_A.append({k: v for k, v in weights.items() if k.endswith("lora_A")})
        client_B.append({k: v for k, v in weights.items() if "lora_B" in k and not k.endswith("lora_B_w")})
        client_lambdas.append({k: v for k, v in weights.items() if k.endswith("lora_lambdas")})
        client_B_w.append({k: v for k, v in weights.items() if k.endswith("lora_B_w")})
    a_stacked = stack_A(client_A, client_p, hidden, lora_r)
    b_stacked = stack_B(client_B, num_B, hidden, lora_r)
    lambdas_stacked = stack_lambdas(client_lambdas, num_tasks, lora_r)
    b_w_stacked = stack_B_w(client_B_w)
    return {**a_stacked, **b_stacked, **lambdas_stacked, **b_w_stacked}



# New: linear_freeze helpers (freeze prefix + append-only growth)


def _set_block_size(model: torch.nn.Module, block_size: int) -> None:
    """Set block_size on all mLoRALinear layers (enables blockwise softmax per block)."""
    if _mLoRALinear is None:
        return
    for module in model.modules():
        if isinstance(module, _mLoRALinear):
            module.block_size = int(block_size)


def _extract_lora_weights(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract all parameters whose name contains 'lora' as CPU tensors."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if "lora" in name}


def _copy_lora_prefix(old_model: torch.nn.Module, new_model: torch.nn.Module, *, old_r: int, old_B: int) -> None:
    """Copy prefix slices of LoRA params from old_model into an expanded new_model."""
    old_params = dict(old_model.named_parameters())
    new_params = dict(new_model.named_parameters())
    with torch.no_grad():
        for name, old_p in old_params.items():
            if "lora" not in name:
                continue
            if name not in new_params:
                continue
            new_p = new_params[name]
            src = old_p.data.to(device=new_p.device, dtype=new_p.dtype)

            if name.endswith("lora_A"):
                # (1, r, in)
                new_p.data[:, :old_r, :].copy_(src)
            elif name.endswith("lora_lambdas"):
                # (tasks, r, r)
                new_p.data[:, :old_r, :old_r].copy_(src)
            elif name.endswith("lora_B_w"):
                # (tasks, B)
                new_p.data[:, :old_B].copy_(src)
            elif "lora_B" in name and not name.endswith("lora_B_w"):
                # (B, out, r)
                new_p.data[:old_B, :, :old_r].copy_(src)


def _register_freeze_prefix_hooks(model: torch.nn.Module, *, old_r: int, old_B: int) -> None:
    """
    Register gradient hooks to freeze the old (prefix) blocks:
      - lora_A: freeze rank [:old_r]
      - lora_lambdas: freeze rows/cols [:old_r]
      - lora_B_w: freeze [:old_B]
      - lora_B: freeze B indices [:old_B], and for new B indices freeze old rank cols [:old_r]
    """
    for name, p in model.named_parameters():
        if "lora" not in name or not p.requires_grad:
            continue

        if name.endswith("lora_A"):
            def hook_A(grad, old_r=old_r):
                if grad is None:
                    return None
                g = grad.clone()
                g[:, :old_r, :] = 0
                return g
            p.register_hook(hook_A)

        elif name.endswith("lora_lambdas"):
            def hook_L(grad, old_r=old_r):
                if grad is None:
                    return None
                g = grad.clone()
                g[:, :old_r, :] = 0
                g[:, :, :old_r] = 0
                return g
            p.register_hook(hook_L)

        elif name.endswith("lora_B_w"):
            def hook_Bw(grad, old_B=old_B):
                if grad is None:
                    return None
                g = grad.clone()
                g[:, :old_B] = 0
                return g
            p.register_hook(hook_Bw)

        elif "lora_B" in name and not name.endswith("lora_B_w"):
            def hook_B(grad, old_B=old_B, old_r=old_r):
                if grad is None:
                    return None
                g = grad.clone()
                # Freeze all old B matrices completely
                g[:old_B, :, :] = 0
                # For newly appended B matrices, freeze old rank columns
                g[old_B:, :, :old_r] = 0
                return g
            p.register_hook(hook_B)


def aggregate_mtl_weights_linear_freeze(
    *,
    global_lora: Dict[str, torch.Tensor],
    client_weights: List[Dict[str, torch.Tensor]],
    client_p: List[float],
    num_tasks: int,
    old_r: int,
    add_r: int,
    old_B: int,
    add_B: int,
) -> Dict[str, torch.Tensor]:
    """
    Append-only aggregation:
      - keep global prefix blocks unchanged
      - append each client's NEW block into disjoint rank/B slices
    """
    K = len(client_weights)
    new_r = old_r + K * add_r
    new_B = old_B + K * add_B

    out: Dict[str, torch.Tensor] = {}

    for name, g_old in global_lora.items():
        if name.endswith("lora_A"):
            in_dim = int(g_old.shape[2])
            A_new = torch.zeros((1, new_r, in_dim), dtype=g_old.dtype)
            A_new[:, :old_r, :] = g_old
            for c in range(K):
                A_c = client_weights[c][name]
                block = A_c[:, old_r:, :]
                A_new[:, old_r + c * add_r : old_r + (c + 1) * add_r, :] = client_p[c] * block
            out[name] = A_new

        elif name.endswith("lora_lambdas"):
            L_new = torch.zeros((num_tasks, new_r, new_r), dtype=g_old.dtype)
            L_new[:, :old_r, :old_r] = g_old
            for c in range(K):
                L_c = client_weights[c][name]
                block = L_c[:, old_r:, old_r:]
                s = old_r + c * add_r
                L_new[:, s : s + add_r, s : s + add_r] = block
            out[name] = L_new

        elif name.endswith("lora_B_w"):
            Bw_new = torch.zeros((num_tasks, new_B), dtype=g_old.dtype)
            Bw_new[:, :old_B] = g_old
            for c in range(K):
                Bw_c = client_weights[c][name]
                block = Bw_c[:, old_B:]
                s = old_B + c * add_B
                Bw_new[:, s : s + add_B] = block
            out[name] = Bw_new

        elif "lora_B" in name and not name.endswith("lora_B_w"):
            out_dim = int(g_old.shape[1])
            B_new = torch.zeros((new_B, out_dim, new_r), dtype=g_old.dtype)
            B_new[:old_B, :, :old_r] = g_old
            for c in range(K):
                B_c = client_weights[c][name]
                block = B_c[old_B:, :, old_r:]  # (add_B, out, add_r)
                b0 = old_B + c * add_B
                r0 = old_r + c * add_r
                B_new[b0 : b0 + add_B, :, r0 : r0 + add_r] = block
            out[name] = B_new

    return out



# Model update utilities


def update_global_model(global_model: torch.nn.Module, avg_weights: Dict[str, torch.Tensor]) -> None:
    """Copy aggregated LoRA weights into the global model (in-place)."""
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in avg_weights:
                w = avg_weights[name].to(device=param.device, dtype=param.dtype)
                if param.shape == w.shape:
                    param.copy_(w)
                else:
                    print(f"[WARNING] Shape mismatch for {name}: model={param.shape}, weights={w.shape}")


def update_head_params(global_model: torch.nn.Module, avg_heads: Dict[str, torch.Tensor]) -> None:
    """Copy aggregated head weights into the global model (in-place)."""
    if not avg_heads:
        return
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in avg_heads:
                w = avg_heads[name].to(device=param.device, dtype=param.dtype)
                if param.shape == w.shape:
                    param.copy_(w)
                else:
                    print(f"[WARNING] Shape mismatch for head {name}: model={param.shape}, weights={w.shape}")


def transfer_non_lora_params(old_model: torch.nn.Module, new_model: torch.nn.Module, round_num: int | None = None) -> int:
    """Transfer non-LoRA parameters from old_model to new_model."""
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    params_transferred = 0
    for name, param in old_state_dict.items():
        if "lora" not in name and name in new_state_dict:
            if new_state_dict[name].shape == param.shape:
                new_state_dict[name].copy_(param)
                params_transferred += 1
            else:
                round_str = f" (FL round {round_num})" if round_num is not None else ""
                print(f"[WARNING] Shape mismatch for {name}{round_str}: old={param.shape}, new={new_state_dict[name].shape}")
    new_model.load_state_dict(new_state_dict)
    return params_transferred



# Args / parsing

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TinyLlama + mLoRA on multi-task GLUE (single GPU)")

    # Model / output
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--output_dir", type=str, default="./outputs_tinyllama_glue_mlora_sgpu")
    p.add_argument("--seed", type=int, default=42)

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    # Mixed precision
    p.add_argument("--fp16", action="store_true")

    # LoRA / mLoRA hyperparams (alpha FIXED)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_B", type=int, default=2, help="Local number of B matrices per client-block")
    p.add_argument("--temperature", type=float, default=0.1)

    # Which additional params to train
    p.add_argument("--freeze_bias", action="store_true")
    p.add_argument("--freeze_layernorm", action="store_true")

    # Checkpointing
    p.add_argument("--save_steps", type=int, default=2500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--save_pre_eval_ckpt", action="store_true")
    p.add_argument("--resume_from_ckpt", type=str, default=None)

    # HF / dataset cache
    p.add_argument("--offline", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--glue_disk_cache_dir", type=str, default=None)
    p.add_argument("--hf_datasets_cache_dir", type=str, default=None)

    # Eval details dump
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--save_eval_details", action="store_true")
    p.add_argument("--eval_details_max_examples", type=int, default=200)
    p.add_argument("--eval_details_only_errors", action="store_true")
    p.add_argument("--eval_details_topk", type=int, default=2)
    p.add_argument("--stsb_abs_err_threshold", type=float, default=0.5)

    # Federated learning settings
    p.add_argument("--num_fl_rounds", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=1)
    p.add_argument("--dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--strat", type=str, default="FLoRA", help="Aggregation strategy: 'FLoRA'/'federated', 'fedit', or 'centralized'")

    # New: growth mode toggle (ONLY new feature)
    p.add_argument(
        "--flora_growth",
        type=str,
        choices=["exponential", "linear_freeze"],
        default="exponential",
        help="exponential: old behaviour (rank/B multiply each round). linear_freeze: freeze old blocks; per round clients train only a new block; server appends blocks (linear growth).",
    )
    p.add_argument(
        "--linear_add_r",
        type=int,
        default=None,
        help="In linear_freeze mode: rank of the NEW block each client trains per round (default: lora_r).",
    )
    p.add_argument(
        "--linear_add_B",
        type=int,
        default=None,
        help="In linear_freeze mode: number of NEW B matrices each client adds per round (default: num_B).",
    )

    # Override client weights in federated averaging
    p.add_argument(
        "--client_p",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Override client weights: provide one value (replicated) or one per client; "
            "NO normalisation is applied (values are used as-is)."
        ),
    )

    # Test mode
    p.add_argument("--test", action="store_true")

    return p.parse_args()



# Client fine-tuning

def fine_tune_client(
    model: torch.nn.Module,
    client_data: dict,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool = False,
) -> Dict[str, torch.Tensor]:
    """Fine-tune a model on a single client's data and return LoRA weights."""
    train(
        model=model,
        task_data=client_data,
        device=device,
        use_amp=use_amp,
        output_dir=args.output_dir,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_pre_eval_ckpt=args.save_pre_eval_ckpt,
        eval_every_epoch=(not args.skip_eval),
        save_eval_details=args.save_eval_details,
        eval_details_max_examples=args.eval_details_max_examples,
        eval_details_only_errors=args.eval_details_only_errors,
        eval_details_topk=args.eval_details_topk,
        stsb_abs_err_threshold=args.stsb_abs_err_threshold,
        resume_from_ckpt=args.resume_from_ckpt,
        args_for_ckpt=args,
    )
    return _extract_lora_weights(model)



def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    hf_token = get_hf_token(args.hf_token)
    hf_home = default_hf_home()
    glue_disk_cache_dir = args.glue_disk_cache_dir or os.path.join(hf_home, "glue_disk_cache")
    hf_datasets_cache_dir = args.hf_datasets_cache_dir or os.environ.get("HF_DATASETS_CACHE") or None

    print(f"[INFO] device={device} use_amp={use_amp}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] strat={args.strat} flora_growth={args.flora_growth}")

    tokenizer = create_tokenizer(args.model_name, offline=args.offline)

    # Create initial global model (fixed alpha)
    global_model = create_model(
        model_name=args.model_name,
        offline=args.offline,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_B=args.num_B,
        temperature=args.temperature,
    )
    _set_block_size(global_model, args.num_B)

    # Trainable params
    train_bias = not args.freeze_bias
    train_ln = not args.freeze_layernorm
    set_trainable_params(global_model, train_bias=train_bias, train_layernorm=train_ln)
    cast_trainable_params_to_fp32(global_model)

    # Data loaders
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
        save_eval_details=args.save_eval_details,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        test_mode=args.test,
    )

    # Track global shapes
    flora_r = int(args.lora_r)
    flora_B = int(args.num_B)

    # Linear-freeze block sizes per round
    add_r = int(args.linear_add_r) if args.linear_add_r is not None else int(args.lora_r)
    add_B = int(args.linear_add_B) if args.linear_add_B is not None else int(args.num_B)

    if args.flora_growth == "linear_freeze":
        if add_r <= 0:
            raise ValueError("--linear_add_r must be > 0 in linear_freeze mode")
        if add_B <= 0:
            raise ValueError("--linear_add_B must be > 0 in linear_freeze mode")
        # To keep blockwise softmax valid, ensure global B stays divisible by local num_B.
        if (add_B % args.num_B) != 0:
            raise ValueError(
                f"In linear_freeze mode, require linear_add_B ({add_B}) to be a multiple of local num_B ({args.num_B}) "
                "so B_num remains divisible by block_size."
            )

    for round_idx in range(args.num_fl_rounds):
        print(f"[INFO] Starting FL round {round_idx + 1}/{args.num_fl_rounds} (global_r={flora_r}, global_B={flora_B})")

        # Snapshot global lora weights (needed for linear_freeze aggregation)
        global_lora_cpu = _extract_lora_weights(global_model)

        client_weights: List[Dict[str, torch.Tensor]] = []
        client_heads: List[Dict[str, torch.Tensor]] = []

        # Determine whether linear_freeze is active for this round (only in FLoRA mode and from round 2 onwards)
        is_flora = (args.strat != "fedit" and args.strat != "centralized")
        do_linear_freeze = (args.flora_growth == "linear_freeze" and is_flora and round_idx > 0)

        for client_id in range(args.num_clients):
            print(f"[INFO] Fine-tuning client {client_id + 1}/{args.num_clients}")

            if do_linear_freeze:
                # Expand model by one new block for this client: (old + add)
                client_r = flora_r + add_r
                client_B = flora_B + add_B

                if (client_B % args.num_B) != 0:
                    raise ValueError(
                        f"[linear_freeze] client_B={client_B} must be divisible by local num_B={args.num_B} "
                        "so blockwise softmax remains well-defined."
                    )

                client_model = create_model(
                    model_name=args.model_name,
                    offline=args.offline,
                    device=device,
                    lora_r=client_r,
                    lora_alpha=args.lora_alpha,  # fixed
                    lora_dropout=args.lora_dropout,
                    num_B=client_B,
                    temperature=args.temperature,
                )
                _set_block_size(client_model, args.num_B)

                # Copy non-LoRA params
                transfer_non_lora_params(global_model, client_model, round_num=round_idx + 1)

                # Copy LoRA prefix blocks into expanded model
                _copy_lora_prefix(global_model, client_model, old_r=flora_r, old_B=flora_B)

                # Set trainability and freeze prefix via gradient hooks
                set_trainable_params(client_model, train_bias=train_bias, train_layernorm=train_ln)
                cast_trainable_params_to_fp32(client_model)
                _register_freeze_prefix_hooks(client_model, old_r=flora_r, old_B=flora_B)

            else:
                # Baseline behaviour: clients train the full current global adapter
                client_model = copy.deepcopy(global_model)

            client_data_dict = task_data[client_id]
            cw = fine_tune_client(client_model, client_data_dict, device, args, use_amp)
            client_weights.append(cw)

            ch = {
                name: param.detach().cpu().clone()
                for name, param in client_model.named_parameters()
                if name.startswith("heads.")
            }
            client_heads.append(ch)

        # Compute client_p from data sizes (default)
        client_sizes: List[int] = []
        for client_data_dict in task_data:
            size = 0
            for _, data_obj in client_data_dict.items():
                train_loader = data_obj.train_loader
                try:
                    bs = getattr(train_loader, "batch_size", 1)
                    size += len(train_loader) * bs
                except Exception:
                    pass
            client_sizes.append(size)

        total_size = sum(client_sizes)
        if total_size > 0:
            client_p = [s / total_size for s in client_sizes]
        else:
            client_p = [1.0 / len(client_sizes)] * len(client_sizes)

        # Override client_p if provided (NO NORMALISATION)
        if args.client_p is not None:
            override = list(args.client_p)
            if len(override) == 1:
                override = override * len(client_sizes)
            if len(override) != len(client_sizes):
                raise ValueError(
                    f"--client_p length ({len(override)}) must match number of clients "
                    f"({len(client_sizes)}) or be a single value"
                )
            # Use values as-is (no division by sum)
            client_p = [float(x) for x in override]

        print(f"[DEBUG] FL round {round_idx + 1}: client_p={client_p}")
        print(f"[DEBUG] FL round {round_idx + 1}: Aggregating weights from {len(client_weights)} clients")

        avg_weights: Dict[str, torch.Tensor]
        avg_heads: Dict[str, torch.Tensor]
        next_num_B: int

        if args.strat == "fedit":
            avg_weights = fed_avg(client_weights)
            next_num_B = args.num_B
            avg_heads = fed_avg_heads(client_heads, client_p)

        elif args.strat == "centralized":
            avg_weights = client_weights[0]
            next_num_B = args.num_B
            avg_heads = client_heads[0]

        else:
            # FLoRA mode
            if do_linear_freeze:
                # Linear growth: append only the new blocks
                new_flora_r = flora_r + len(client_weights) * add_r
                global_num_B = flora_B + len(client_weights) * add_B
                if (global_num_B % args.num_B) != 0:
                    raise ValueError(
                        f"[linear_freeze] global_num_B={global_num_B} must be divisible by local num_B={args.num_B}"
                    )

                avg_weights = aggregate_mtl_weights_linear_freeze(
                    global_lora=global_lora_cpu,
                    client_weights=client_weights,
                    client_p=client_p,
                    num_tasks=len(GLUE_TASKS),
                    old_r=flora_r,
                    add_r=add_r,
                    old_B=flora_B,
                    add_B=add_B,
                )
                flora_r = new_flora_r
                flora_B = global_num_B
                next_num_B = flora_B
                avg_heads = fed_avg_heads(client_heads, client_p)

            else:
                # Exponential stacking (existing behaviour)
                # Determine new LoRA rank as sum of per-client ranks
                client_ranks: List[int] = []
                for cw in client_weights:
                    found = False
                    for name, tensor in cw.items():
                        if name.endswith("lora_A"):
                            client_ranks.append(tensor.shape[1])
                            found = True
                            break
                    if not found:
                        raise ValueError("Could not determine LoRA rank from client weights")
                new_flora_r = sum(client_ranks)

                # B dimension multiplies by K each round
                global_num_B = flora_B * len(client_weights)

                hidden_dim = global_model.encoder.config.hidden_size
                avg_weights = aggregate_mtl_weights(
                    client_weights,
                    client_p=client_p,
                    hidden=hidden_dim,
                    num_B=global_num_B,
                    num_tasks=len(GLUE_TASKS),
                    lora_r=new_flora_r,
                )
                flora_r = new_flora_r
                flora_B = global_num_B
                next_num_B = flora_B
                avg_heads = fed_avg_heads(client_heads, client_p)

        # Create new global model with updated shapes (alpha fixed)
        new_global_model = create_model(
            model_name=args.model_name,
            offline=args.offline,
            device=device,
            lora_r=flora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_B=next_num_B,
            temperature=args.temperature,
        )
        _set_block_size(new_global_model, args.num_B)

        # Transfer non-LoRA parameters
        transfer_non_lora_params(global_model, new_global_model, round_num=round_idx + 1)

        # Update LoRA weights + heads
        update_global_model(new_global_model, avg_weights)
        update_head_params(new_global_model, avg_heads)

        # Reapply trainability settings
        set_trainable_params(new_global_model, train_bias=train_bias, train_layernorm=train_ln)
        cast_trainable_params_to_fp32(new_global_model)

        global_model = new_global_model
        print(f"[INFO] Completed FL round {round_idx + 1}/{args.num_fl_rounds} (global_r={flora_r}, global_B={flora_B})")

    # Save final global model and adapter/head weights
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_ckpt_path = os.path.join(ckpt_dir, "ckpt_global_final.pt")

    save_checkpoint(
        ckpt_path=final_ckpt_path,
        model=global_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=args.epochs,
        update_step=0,
        args=args,
    )

    save_adapter_and_heads(args.output_dir, global_model)

    # Copy to *_final.pt so resolve_load_paths() can prefer these when present
    import shutil

    adapter_src = os.path.join(args.output_dir, "adapter_state.pt")
    heads_src = os.path.join(args.output_dir, "heads_state.pt")
    if os.path.exists(adapter_src):
        shutil.copy(adapter_src, os.path.join(args.output_dir, "adapter_state_final.pt"))
    if os.path.exists(heads_src):
        shutil.copy(heads_src, os.path.join(args.output_dir, "heads_state_final.pt"))

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()