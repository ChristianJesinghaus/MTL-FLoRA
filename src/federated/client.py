"""Flower NumPyClient implementation for federated MTL-LoRA."""
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from .state import adapter_state_to_numpy, apply_adapter_state
from .training import evaluate, train_one_round


class FederatedMloraClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        local_epochs: int,
        learning_rate: float,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        self.parameter_keys, _ = adapter_state_to_numpy(self.model.state_dict().items())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        _, weights = adapter_state_to_numpy(self.model.state_dict().items())
        return weights

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        apply_adapter_state(self.model, self.parameter_keys, parameters)
        lr = float(config.get("lr", self.learning_rate))
        epochs = int(config.get("local_epochs", self.local_epochs))
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        for _ in range(epochs):
            train_one_round(self.model, self.train_loader, optimizer, self.device)

        _, weights = adapter_state_to_numpy(self.model.state_dict().items())
        return weights, len(self.train_loader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        apply_adapter_state(self.model, self.parameter_keys, parameters)
        loss = evaluate(self.model, self.val_loader, self.device)
        num_examples = len(self.val_loader.dataset)
        return float(loss), num_examples, {"loss": float(loss)}
