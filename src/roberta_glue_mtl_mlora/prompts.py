from __future__ import annotations

from typing import Any, List, Optional


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return s.replace("\r", " ").replace("\n", " ").strip()


def build_eval_prompt(task: str, text1: str, text2: Optional[str], label_names: Optional[List[str]]) -> str:
    """Human-readable prompt used ONLY for evaluation-dump interpretability.

    Note: The model is NOT trained with prompts; this is only for analysis.
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
    return f"[{task_upper}] Input: {text1} {text2 or ''}".strip()
