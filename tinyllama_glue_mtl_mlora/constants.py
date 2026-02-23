"""Constants for TinyLlama multi‑task LoRA fine‑tuning on GLUE.

This module defines the set of GLUE tasks, their label mappings and
text field keys as well as the list of linear layer name
substrings used to locate the attention projection layers in the
TinyLlama architecture.  These constants mirror those in
`src/roberta_glue_mtl_mlora/constants.py` but adapt the target layer
names to Llama (``q_proj``, ``k_proj``, ``v_proj`` and
``o_proj``).  The rest of the task metadata remains unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# -----------------------------
# GLUE task metadata
# -----------------------------

# A subset of GLUE tasks used in this project.  Additional tasks can
# be uncommented if desired; ensure that the corresponding datasets
# are available and that the number of label dimensions in
# ``TASK_NUM_LABELS`` below matches your chosen tasks.
GLUE_TASKS: List[str] = [
    #"cola",
    "sst2",
     "mrpc",
     "qqp",
     "mnli",
    # "qnli",
    # "rte",
    # "stsb",
]

# Mapping from task name to integer id.  This is used for
# selecting the correct LoRA lambda slice during forward passes.
TASK_TO_ID: Dict[str, int] = {task: i for i, task in enumerate(GLUE_TASKS)}

# Mapping from task name to the keys used to extract text fields from
# each GLUE example.  For single‑sentence classification tasks the
# second entry can be ``None``.
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

# Number of labels per task.  GLUE is mostly classification except
# ``stsb`` which is regression.
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

# Fallback names for each label id in each task.  These are used
# primarily for human‑readable evaluation outputs.  If you add tasks
# above, be sure to extend this mapping accordingly.
FALLBACK_LABEL_NAMES: Dict[str, Dict[int, str]] = {
    "cola": {0: "unacceptable", 1: "acceptable"},
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "rte": {0: "entailment", 1: "not_entailment"},
}

# In test mode the data loader will sample only this many examples per
# client per task.  This speeds up debugging when developing on
# limited hardware.
TEST_SAMPLE_SIZE: int = 50

# Llama‑style attention projections to be adapted with (m)LoRA.  The
# names here correspond to substrings in the module hierarchy of
# ``transformers`` Llama implementations.  Matching is
# performed by substring containment, so only include unique
# identifiers.
LLAMA_TARGET_SUBSTRINGS: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
