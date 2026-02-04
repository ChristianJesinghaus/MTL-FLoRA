from __future__ import annotations

import json
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F

from .constants import GLUE_TASKS
from .data import TaskData
from .metrics import as_float, f1_from_counts, matthews_corrcoef_from_counts, pearson_from_sums
from .prompts import build_eval_prompt, safe_text
from .utils import tqdm_main


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
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
    """Run evaluation over all GLUE tasks.

    Single-GPU version: no distributed sharding / all_reduce.
    """

    model.eval()

    details_dir = os.path.join(output_dir, "eval_details")
    if save_details:
        os.makedirs(details_dir, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}

    for task in GLUE_TASKS:
        td = task_data[task]
        label_names = td.label_names

        task_results: Dict[str, float] = {}

        for split_name, loader in td.val_loaders:
            # Metrics accumulators
            if task == "stsb":
                n = 0.0
                sum_x = 0.0
                sum_y = 0.0
                sum_x2 = 0.0
                sum_y2 = 0.0
                sum_xy = 0.0
            elif task in ("mrpc", "qqp"):
                correct = 0
                total = 0
                tp = 0
                fp = 0
                fn = 0
            elif task == "cola":
                tp = 0
                tn = 0
                fp = 0
                fn = 0
            else:
                correct = 0
                total = 0

            # Optional details file
            details_fh = None
            wrote = 0
            if save_details:
                details_path = os.path.join(details_dir, f"{tag}_{task}_{split_name}.jsonl")
                details_fh = open(details_path, "w", encoding="utf-8")
                header = {
                    "_meta": {
                        "task": task,
                        "split": split_name,
                        "tag": tag,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "format": "jsonl",
                    }
                }
                details_fh.write(json.dumps(header) + "\n")

            try:
                pbar = tqdm_main(loader, desc=f"eval {task}:{split_name}")
                for batch in pbar:
                    labels = batch.pop("labels").to(device)
                    meta_list = batch.pop("meta", None)
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(task=task, batch=batch)  # type: ignore[misc]

                    if task == "stsb":
                        preds = logits.squeeze(-1).float()

                        x = preds.detach().cpu().double().tolist()
                        y = labels.detach().cpu().double().tolist()

                        # accumulate pearson sums
                        # list -> python sums (fast enough for GLUE)
                        for xi, yi in zip(x, y):
                            n += 1.0
                            sum_x += float(xi)
                            sum_y += float(yi)
                            sum_x2 += float(xi) * float(xi)
                            sum_y2 += float(yi) * float(yi)
                            sum_xy += float(xi) * float(yi)

                        # details
                        if details_fh is not None and meta_list is not None:
                            for i in range(len(meta_list)):
                                if details_max_examples >= 0 and wrote >= details_max_examples:
                                    break

                                text1 = safe_text(meta_list[i].get("text1"))
                                text2 = safe_text(meta_list[i].get("text2"))
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
                                    "abs_error": as_float(abs_err, 6),
                                }
                                details_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                                wrote += 1

                            if wrote and (wrote % 50 == 0):
                                details_fh.flush()

                    else:
                        preds = torch.argmax(logits, dim=-1)

                        if task in ("mrpc", "qqp"):
                            correct += int((preds == labels).sum().item())
                            total += int(labels.numel())
                            tp += int(((preds == 1) & (labels == 1)).sum().item())
                            fp += int(((preds == 1) & (labels == 0)).sum().item())
                            fn += int(((preds == 0) & (labels == 1)).sum().item())
                        elif task == "cola":
                            tp += int(((preds == 1) & (labels == 1)).sum().item())
                            tn += int(((preds == 0) & (labels == 0)).sum().item())
                            fp += int(((preds == 1) & (labels == 0)).sum().item())
                            fn += int(((preds == 0) & (labels == 1)).sum().item())
                        else:
                            correct += int((preds == labels).sum().item())
                            total += int(labels.numel())

                        # details
                        if details_fh is not None and meta_list is not None:
                            probs = torch.softmax(logits.float(), dim=-1)
                            topk = min(int(details_topk), probs.shape[-1])

                            top_vals, top_idx = torch.topk(probs, k=topk, dim=-1)
                            top_vals = top_vals.detach().cpu()
                            top_idx = top_idx.detach().cpu()
                            probs_cpu = probs.detach().cpu()

                            for i in range(len(meta_list)):
                                if details_max_examples >= 0 and wrote >= details_max_examples:
                                    break

                                text1 = safe_text(meta_list[i].get("text1"))
                                text2 = safe_text(meta_list[i].get("text2"))
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

                                prob_list = [as_float(x, 6) for x in probs_cpu[i].tolist()]
                                top_list = []
                                for k in range(topk):
                                    lid = int(top_idx[i, k].item())
                                    lname = (
                                        label_names[lid]
                                        if (label_names is not None and 0 <= lid < len(label_names))
                                        else str(lid)
                                    )
                                    top_list.append(
                                        {"label_id": lid, "label_name": lname, "p": as_float(top_vals[i, k].item(), 6)}
                                    )

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

            # Compute split metrics
            if task == "cola":
                # Totals for sanity checks
                pred_pos = tp + fp
                pred_neg = tn + fn
                true_pos = tp + fn
                true_neg = tn + fp

                denom_zero = (pred_pos * true_pos * true_neg * pred_neg) == 0

                if denom_zero:
                    # If both gold and predicted labels contain both classes, denominator should NOT be zero
                    if pred_pos > 0 and pred_neg > 0 and true_pos > 0 and true_neg > 0:
                        raise AssertionError(
                            f"CoLA MCC denominator==0 but both classes present (tp={tp}, tn={tn}, fp={fp}, fn={fn})."
                        )
                    # Expected degenerate case (e.g., all predictions same or all gold same): warn and set NaN
                    print(f"[WARN] {task}:{split_name} MCC denominator zero (tp={tp}, tn={tn}, fp={fp}, fn={fn}). Setting MCC=NaN.")
                    mcc = float("nan")
                else:
                    mcc = matthews_corrcoef_from_counts(tp, tn, fp, fn)

                task_results[f"{split_name}_mcc"] = float(mcc)
            elif task == "stsb":
                r = pearson_from_sums(n, sum_x, sum_y, sum_x2, sum_y2, sum_xy)
                task_results[f"{split_name}_pearson"] = float(r)
            elif task in ("mrpc", "qqp"):
                acc = float(correct) / max(1.0, float(total))
                f1 = f1_from_counts(tp, fp, fn)
                task_results[f"{split_name}_acc"] = float(acc)
                task_results[f"{split_name}_f1"] = float(f1)
            else:
                acc = float(correct) / max(1.0, float(total))
                task_results[f"{split_name}_acc"] = float(acc)

        # MNLI average
        if task == "mnli":
            m = task_results.get("validation_matched_acc", None)
            mm = task_results.get("validation_mismatched_acc", None)
            if m is not None and mm is not None:
                task_results["validation_avg_acc"] = 0.5 * (float(m) + float(mm))

        results[task] = task_results

    # GLUE average (as in the original script)
    avg_scores = []
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

    model.train()
    return results
