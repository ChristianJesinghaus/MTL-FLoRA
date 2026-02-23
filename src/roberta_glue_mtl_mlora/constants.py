from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# -----------------------------
# GLUE task metadata
# -----------------------------

GLUE_TASKS: List[str] = [
    #"cola",
    "sst2",
    "mrpc",
#    "qqp",
#    "mnli",
#    "qnli",
#    "rte",
#    "stsb",
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

# Test mode configuration
TEST_SAMPLE_SIZE: int = 50  # Number of samples per client per task in test mode

# RoBERTa/BERT-style attention linears we adapt with (m)LoRA.
# Note: these are substrings used to match module names.
ROBERTA_TARGET_SUBSTRINGS = (
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
)
