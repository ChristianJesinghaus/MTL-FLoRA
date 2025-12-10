"""Federated MTL-LoRA training using Flower."""
import argparse
import ast
from typing import Tuple

import flwr as fl
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from src.custom_model import LlamaForCausalLM
from src.federated import (
    FederatedMloraClient,
    build_dataloader,
    load_instruction_datasets,
)
from src.utils import set_no_grad, wrap_model


def build_model_and_tokenizer(
    base_model: str,
    adapter_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules,
    lambda_num: int,
    num_B: int,
    temperature: float,
    lora_num: int,
    expert_num: int,
    task_num: int,
    te_dim: int,
    use_gradient_checkpointing: bool,
    merge_weights: bool,
    Wdecompose: bool,
    dora_simple: bool,
) -> Tuple[torch.nn.Module, transformers.PreTrainedTokenizer]:
    if "llama" in base_model and adapter_name.lower() in ["mlora", "moelora"]:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        if "Llama-3" in base_model:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if adapter_name.lower() == "mlora":
        adapter_config = {
            "type": "mlora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lambda_num": lambda_num,
            "B_num": num_B,
            "B_scale": temperature,
            "diagonal_format": False,
        }
    elif adapter_name.lower() == "multilora":
        adapter_config = {
            "type": "multilora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_num": lora_num,
        }
    elif adapter_name.lower() == "moelora":
        adapter_config = {
            "type": "moelora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "expert_num": expert_num,
            "task_num": task_num,
            "task_embedding_dim": te_dim,
        }
    elif adapter_name.lower() == "dora":
        adapter_config = {
            "type": "dora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "merge_weights": merge_weights,
            "Wdecompose": Wdecompose,
            "dora_simple": dora_simple,
        }
    else:
        raise ValueError(f"Unsupported adapter {adapter_name}")

    model = wrap_model(model, lora_target_modules, adapter_config)
    set_no_grad(model)
    model.config.use_cache = False
    return model, tokenizer


def start_server(server_address: str, num_rounds: int):
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl.server.strategy.FedAvg(),
    )


def start_client(args):
    model, tokenizer = build_model_and_tokenizer(
        base_model=args.base_model,
        adapter_name=args.adapter_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lambda_num=args.lambda_num,
        num_B=args.num_B,
        temperature=args.temperature,
        lora_num=args.lora_num,
        expert_num=args.expert_num,
        task_num=args.task_num,
        te_dim=args.te_dim,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        merge_weights=args.merge_weights,
        Wdecompose=args.Wdecompose,
        dora_simple=args.dora_simple,
    )

    train_ds, val_ds = load_instruction_datasets(
        data_path=args.data_path,
        tokenizer=tokenizer,
        cutoff_len=args.cutoff_len,
        train_on_inputs=args.train_on_inputs,
        adapter_name=args.adapter_name,
        cache_dir=args.cache_dir,
        val_split=args.val_split,
    )

    train_loader = build_dataloader(
        train_ds, tokenizer, batch_size=args.batch_size, cutoff_len=args.cutoff_len
    )
    val_loader = build_dataloader(
        val_ds, tokenizer, batch_size=args.batch_size, cutoff_len=args.cutoff_len
    )

    client = FederatedMloraClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        local_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


def parse_args():
    parser = argparse.ArgumentParser(description="Federated MTL-LoRA with Flower")
    subparsers = parser.add_subparsers(dest="role", required=True)

    server_parser = subparsers.add_parser("server", help="Start Flower server")
    server_parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    server_parser.add_argument("--num_rounds", type=int, default=3)

    client_parser = subparsers.add_parser("client", help="Start a Flower client")
    client_parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    client_parser.add_argument("--base_model", type=str, required=True)
    client_parser.add_argument("--data_path", type=str, required=True)
    client_parser.add_argument("--adapter_name", type=str, default="mlora")
    client_parser.add_argument("--cache_dir", type=str, default=None)
    client_parser.add_argument("--train_on_inputs", action="store_true")
    client_parser.add_argument("--batch_size", type=int, default=2)
    client_parser.add_argument("--num_epochs", type=int, default=1)
    client_parser.add_argument("--learning_rate", type=float, default=5e-5)
    client_parser.add_argument("--cutoff_len", type=int, default=256)
    client_parser.add_argument("--val_split", type=float, default=0.1)
    client_parser.add_argument("--use_gradient_checkpointing", action="store_true")

    client_parser.add_argument("--lora_r", type=int, default=8)
    client_parser.add_argument("--lora_alpha", type=int, default=16)
    client_parser.add_argument("--lora_dropout", type=float, default=0.05)
    client_parser.add_argument(
        "--lora_target_modules",
        type=ast.literal_eval,
        default=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    client_parser.add_argument("--lambda_num", type=int, default=3)
    client_parser.add_argument("--num_B", type=int, default=3)
    client_parser.add_argument("--temperature", type=float, default=1.0)
    client_parser.add_argument("--lora_num", type=int, default=3)
    client_parser.add_argument("--expert_num", type=int, default=3)
    client_parser.add_argument("--task_num", type=int, default=8)
    client_parser.add_argument("--te_dim", type=int, default=64)
    client_parser.add_argument("--merge_weights", action="store_true")
    client_parser.add_argument("--Wdecompose", action="store_true")
    client_parser.add_argument("--dora_simple", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.role == "server":
        start_server(args.server_address, args.num_rounds)
    else:
        start_client(args)
