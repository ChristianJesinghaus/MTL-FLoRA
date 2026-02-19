from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from .checkpoint import maybe_rotate_checkpoints, save_adapter_and_heads, save_checkpoint
from .constants import GLUE_TASKS
from .data import TaskData
from .eval_loop import evaluate
from .utils import tqdm_main


def train(
    *,
    model: torch.nn.Module,
    task_data: Dict[str, TaskData],
    device: torch.device,
    use_amp: bool,
    output_dir: str,
    epochs: int,
    grad_accum_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    save_steps: int,
    save_total_limit: int,
    save_pre_eval_ckpt: bool,
    eval_every_epoch: bool,
    save_eval_details: bool,
    eval_details_max_examples: int,
    eval_details_only_errors: bool,
    eval_details_topk: int,
    stsb_abs_err_threshold: float,
    resume_from_ckpt: Optional[str],
    args_for_ckpt: Any,
) -> None:
    """Single-GPU training loop.

    Uses the same multi-task schedule logic as the original script.
    """

    # Optimizer / scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)

    steps_per_epoch = sum(len(td.train_loader) for td in task_data.values())
    total_updates = int(math.ceil((steps_per_epoch * epochs) / max(1, grad_accum_steps)))
    warmup_steps = int(total_updates * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    start_epoch = 0
    global_update = 0

    if resume_from_ckpt:
        print(f"[CKPT] Resuming training from {resume_from_ckpt}")
        ckpt = torch.load(resume_from_ckpt, map_location="cpu")
        # load trainable encoder + heads
        from .checkpoint import load_from_checkpoint

        _ = load_from_checkpoint(
            resume_from_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            strict_heads=True,
        )
        start_epoch = int(ckpt.get("epoch", 0))
        global_update = int(ckpt.get("update_step", 0))

    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # Re-create iterators each epoch
        task_iters = {task: iter(task_data[task].train_loader) for task in GLUE_TASKS}

        # Build and shuffle schedule deterministically (same per epoch)
        schedule = []
        for task in GLUE_TASKS:
            schedule.extend([task] * len(task_data[task].train_loader))

        rng = torch.Generator()
        rng.manual_seed(int(getattr(args_for_ckpt, "seed", 42)) + epoch)
        # Use python random for list shuffle (stable across python versions is not guaranteed,
        # but good enough for this use-case)
        import random

        random.Random(int(getattr(args_for_ckpt, "seed", 42)) + epoch).shuffle(schedule)

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm_main(schedule, desc=f"train epoch {epoch+1}/{epochs}")
        micro_step = 0

        for task in pbar:
            micro_step += 1
            batch = next(task_iters[task])
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            do_step = (micro_step % max(1, grad_accum_steps) == 0)
            sync_ctx = nullcontext()  # kept for symmetry with the old code

            with sync_ctx:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(task=task, batch=batch)  # type: ignore[misc]
                    if task == "stsb":
                        loss = F.mse_loss(logits.squeeze(-1).float(), labels.float())
                    else:
                        loss = F.cross_entropy(logits.float(), labels)
                    loss = loss / max(1, grad_accum_steps)

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

                if save_steps > 0 and (global_update % save_steps == 0):
                    ckpt_path = os.path.join(output_dir, "checkpoints", f"ckpt_train_step{global_update}.pt")
                    save_checkpoint(
                        ckpt_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        update_step=global_update,
                        args=args_for_ckpt,
                    )
                    maybe_rotate_checkpoints(
                        os.path.join(output_dir, "checkpoints"),
                        keep_last=save_total_limit,
                    )

        # Log epoch summary
        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else learning_rate
        elapsed_min = (time.time() - start_time) / 60.0
        print(f"[train] epoch={epoch+1}/{epochs}, step={global_update}/{total_updates}, lr={lr:.2e}, elapsed={elapsed_min:.1f}min")

        # Save pre-eval checkpoint
        if save_pre_eval_ckpt:
            ckpt_path = os.path.join(
                output_dir,
                "checkpoints",
                f"ckpt_epoch{epoch+1}_pre_eval_step{global_update}.pt",
            )
            save_checkpoint(
                ckpt_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                update_step=global_update,
                args=args_for_ckpt,
            )
            maybe_rotate_checkpoints(os.path.join(output_dir, "checkpoints"), keep_last=save_total_limit)

        # Evaluate
        if eval_every_epoch:
            results = evaluate(
                model=model,
                task_data=task_data,
                device=device,
                use_amp=use_amp,
                output_dir=output_dir,
                tag=f"epoch{epoch+1}",
                save_details=save_eval_details,
                details_max_examples=eval_details_max_examples,
                details_only_errors=eval_details_only_errors,
                details_topk=eval_details_topk,
                stsb_abs_err_threshold=stsb_abs_err_threshold,
            )

            print(f"[eval] epoch={epoch+1}/{epochs}")
            print(json.dumps(results, indent=2))

            with open(os.path.join(output_dir, f"eval_epoch_{epoch+1}.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            with open(os.path.join(output_dir, "eval_latest.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            # Save trainable encoder + heads
            save_adapter_and_heads(output_dir=output_dir, model=model)

            with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args_for_ckpt), f, indent=2)

            print(f"[INFO] Saved adapter/head weights to {output_dir}", flush=True)

    # Final save (in case eval was disabled)
    save_adapter_and_heads(output_dir=output_dir, model=model)
