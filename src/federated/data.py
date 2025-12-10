"""Dataset helpers for federated instruction tuning."""
from functools import partial
from datasets import load_dataset, load_from_disk


DEFAULT_TASK_FIELD = "task_id"


def _generate_prompt(data_point):
    if data_point.get("input"):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point['instruction']}

                ### Input:
                {data_point['input']}

                ### Response:
                {data_point['output']}"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point['instruction']}

                ### Response:
                {data_point['output']}"""


def _tokenize_sample(tokenizer, cutoff_len, train_on_inputs, adapter_name, data_point):
    def _tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    full_prompt = _generate_prompt(data_point)
    tokenized_full_prompt = _tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = _generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = _tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]

    if adapter_name.lower() in ["mlora", "moelora"]:
        tokenized_full_prompt["lambda_index"] = data_point.get(DEFAULT_TASK_FIELD, 0)
    return tokenized_full_prompt


def load_instruction_datasets(
    data_path: str,
    tokenizer,
    cutoff_len: int,
    train_on_inputs: bool,
    adapter_name: str,
    cache_dir: str | None = None,
    val_split: float = 0.1,
):
    """Load and tokenize instruction datasets for federated training."""

    if cache_dir and load_from_disk is not None:
        train_cache = f"{cache_dir}/train"
        val_cache = f"{cache_dir}/validation"
        try:
            train_ds = load_from_disk(train_cache)
            val_ds = load_from_disk(val_cache)
            return train_ds, val_ds
        except FileNotFoundError:
            pass

    if data_path.endswith(".json"):
        raw = load_dataset("json", data_files=data_path)
    else:
        raw = load_dataset(data_path)

    if "validation" in raw:
        train_ds = raw["train"]
        val_ds = raw["validation"]
    else:
        split = raw["train"].train_test_split(test_size=val_split, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    mapper = partial(
        _tokenize_sample,
        tokenizer,
        cutoff_len,
        train_on_inputs,
        adapter_name,
    )
    train_ds = train_ds.map(mapper, num_proc=4)
    val_ds = val_ds.map(mapper, num_proc=4)

    if cache_dir:
        train_ds.save_to_disk(f"{cache_dir}/train")
        val_ds.save_to_disk(f"{cache_dir}/validation")

    return train_ds, val_ds
