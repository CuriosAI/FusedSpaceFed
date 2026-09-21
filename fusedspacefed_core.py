"""Models and federated-learning utilities for FusedSpaceFed."""

from __future__ import annotations

import platform
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset


TensorState = Dict[str, torch.Tensor]


def runtime_versions() -> Dict[str, str]:
    """Return the software/hardware metadata needed to audit a run."""
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> TensorState:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def move_state_dict(state: Mapping[str, torch.Tensor], device: torch.device) -> TensorState:
    return {name: value.to(device, non_blocking=True) for name, value in state.items()}


def weighted_average_states(states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[int]) -> TensorState:
    """Average state dictionaries using positive weights.

    Integer buffers such as BatchNorm ``num_batches_tracked`` are copied from
    the state with the largest weight.
    """
    if not states:
        raise ValueError("Cannot aggregate an empty state list")
    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length")
    if any(weight <= 0 for weight in weights):
        raise ValueError("All aggregation weights must be positive")

    total = float(sum(weights))
    largest = int(np.argmax(np.asarray(weights)))
    averaged: TensorState = {}
    for name in states[0]:
        reference = states[0][name]
        if not (reference.is_floating_point() or reference.is_complex()):
            averaged[name] = states[largest][name].detach().cpu().clone()
            continue
        value = torch.zeros_like(reference, device="cpu")
        for state, weight in zip(states, weights):
            value.add_(state[name].detach().cpu(), alpha=float(weight) / total)
        averaged[name] = value
    return averaged


def dirichlet_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = 1,
    max_attempts: int = 1000,
) -> List[List[int]]:
    """Create a class-wise Dirichlet partition without sample duplication."""
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if len(labels_array) < num_clients * min_client_size:
        raise ValueError("Not enough samples for the requested minimum client size")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels_array)
    for _ in range(max_attempts):
        parts: List[List[int]] = [[] for _ in range(num_clients)]
        for label in classes:
            indices = np.flatnonzero(labels_array == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(indices), proportions)
            boundaries = np.cumsum(counts)[:-1]
            for client_id, split in enumerate(np.split(indices, boundaries)):
                parts[client_id].extend(split.tolist())
        if min(map(len, parts)) >= min_client_size:
            for part in parts:
                rng.shuffle(part)
            flat = [index for part in parts for index in part]
            if len(flat) != len(labels_array) or len(set(flat)) != len(labels_array):
                raise RuntimeError("Partition audit failed: indices are missing or duplicated")
            return parts
    raise RuntimeError("Unable to generate a Dirichlet partition with non-empty clients")


def pathological_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    classes_per_client: int,
    seed: int,
) -> List[List[int]]:
    """Assign every client a fixed number of classes and consume every sample."""
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.unique(labels_array).tolist()
    if classes_per_client < 1 or classes_per_client > len(classes):
        raise ValueError("classes_per_client must be between 1 and the number of classes")
    if num_clients * classes_per_client < len(classes):
        raise ValueError("Not enough client-class slots to cover every class")

    rng = np.random.default_rng(seed)
    shuffled_classes = classes.copy()
    rng.shuffle(shuffled_classes)
    assignments: List[List[int]] = [[] for _ in range(num_clients)]
    cursor = 0
    for client_id in range(num_clients):
        while len(assignments[client_id]) < classes_per_client:
            candidate = shuffled_classes[cursor % len(shuffled_classes)]
            cursor += 1
            if candidate not in assignments[client_id]:
                assignments[client_id].append(candidate)

    parts: List[List[int]] = [[] for _ in range(num_clients)]
    for label in classes:
        holders = [client_id for client_id, owned in enumerate(assignments) if label in owned]
        indices = np.flatnonzero(labels_array == label)
        rng.shuffle(indices)
        for client_id, split in zip(holders, np.array_split(indices, len(holders))):
            parts[client_id].extend(split.tolist())
    for part in parts:
        rng.shuffle(part)
    flat = [index for part in parts for index in part]
    if len(flat) != len(labels_array) or len(set(flat)) != len(labels_array):
        raise RuntimeError("Pathological partition audit failed")
    return parts


def make_client_loaders(
    dataset: Dataset,
    partitions: Sequence[Sequence[int]],
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
) -> List[DataLoader]:
    loaders: List[DataLoader] = []
    for client_id, indices in enumerate(partitions):
        generator = torch.Generator().manual_seed(seed * 10_000 + client_id)
        loaders.append(
            DataLoader(
                Subset(dataset, list(indices)),
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator if shuffle else None,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )
        )
    return loaders


class PreActBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
            if stride != 1 or in_planes != planes
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut


class ResNet20V2(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.layer1 = self._make_layer(16, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, blocks=3, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [PreActBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        layers.extend(PreActBlock(planes, planes, 1) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        return self.fc(self.pool(x).flatten(1))


class UNetSmallAE(nn.Module):
    """Compact U-Net autoencoder with a configurable bottleneck width ``dz``."""

    ENCODER_PREFIXES = ("enc1", "enc2", "bott")
    DECODER_PREFIXES = ("up2", "dec2", "up1", "dec1", "final")

    def __init__(self, in_channels: int, dz: int, base: int = 16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bott = nn.Sequential(
            nn.Conv2d(base * 2, dz, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dz, dz, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(dz, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(base, in_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        z = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(z), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1), z

    def _select_state(self, prefixes: Sequence[str]) -> TensorState:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if name.startswith(tuple(prefixes))
        }

    def encoder_state(self) -> TensorState:
        return self._select_state(self.ENCODER_PREFIXES)

    def decoder_state(self) -> TensorState:
        return self._select_state(self.DECODER_PREFIXES)

    def _load_partial_state(self, partial: Mapping[str, torch.Tensor]) -> None:
        state = self.state_dict()
        unknown = set(partial) - set(state)
        if unknown:
            raise KeyError(f"Unknown autoencoder parameters: {sorted(unknown)}")
        state.update(move_state_dict(partial, next(self.parameters()).device))
        self.load_state_dict(state)

    def load_encoder_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self._load_partial_state(state)

    def load_decoder_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self._load_partial_state(state)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if name.startswith(self.ENCODER_PREFIXES):
                yield parameter

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if name.startswith(self.DECODER_PREFIXES):
                yield parameter


def task_loss(logits: torch.Tensor, targets: torch.Tensor, task: str) -> torch.Tensor:
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    return F.cross_entropy(logits, targets.long().reshape(-1))


def _autocast_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _new_scaler(device: torch.device):
    return torch.amp.GradScaler("cuda") if device.type == "cuda" else None


class FedAvgClient:
    def __init__(
        self,
        client_id: str | int,
        loader: DataLoader,
        num_classes: int,
        in_channels: int,
        task: str,
        device: torch.device,
        lr: float = 0.01,
    ):
        self.client_id = client_id
        self.loader = loader
        self.task = task
        self.device = device
        self.model = ResNet20V2(num_classes, in_channels).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scaler = _new_scaler(device)
        self.lr = lr
        self.local_steps = 0

    @property
    def num_samples(self) -> int:
        return len(self.loader.dataset)

    def set_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.model.load_state_dict(move_state_dict(state, self.device))

    def state(self) -> TensorState:
        return clone_state_dict(self.model.state_dict())

    def train(self, epochs: int) -> float:
        self.model.train()
        losses: List[float] = []
        self.local_steps = 0
        for _ in range(epochs):
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device):
                    loss = task_loss(self.model(inputs), targets, self.task)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.local_steps += 1
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")


class FedProxClient(FedAvgClient):
    def __init__(self, *args, mu: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.global_parameters: Dict[str, torch.Tensor] = {}

    def set_global_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.set_state(state)
        self.global_parameters = {
            name: state[name].to(self.device).detach().clone()
            for name, _ in self.model.named_parameters()
        }

    def train(self, epochs: int) -> float:
        if not self.global_parameters:
            raise RuntimeError("FedProx global state must be set before local training")
        self.model.train()
        losses: List[float] = []
        self.local_steps = 0
        for _ in range(epochs):
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device):
                    loss = task_loss(self.model(inputs), targets, self.task)
                    proximal = torch.zeros((), device=self.device)
                    for name, parameter in self.model.named_parameters():
                        proximal = proximal + (parameter - self.global_parameters[name]).pow(2).sum()
                    loss = loss + 0.5 * self.mu * proximal
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.local_steps += 1
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")


class ScaffoldClient(FedAvgClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_control = {
            name: torch.zeros_like(parameter, device="cpu")
            for name, parameter in self.model.named_parameters()
        }
        self.server_control: TensorState = {}
        self.control_delta: TensorState = {}

    def set_client_control(self, state: Mapping[str, torch.Tensor]) -> None:
        self.client_control = clone_state_dict(state)

    def set_server_control(self, state: Mapping[str, torch.Tensor]) -> None:
        self.server_control = clone_state_dict(state)

    def train(self, epochs: int) -> float:
        if not self.server_control:
            raise RuntimeError("SCAFFOLD server control must be set before training")
        global_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        previous_control = clone_state_dict(self.client_control)
        self.model.train()
        losses: List[float] = []
        self.local_steps = 0
        for _ in range(epochs):
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                loss = task_loss(self.model(inputs), targets, self.task)
                loss.backward()
                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        correction = self.server_control[name].to(self.device) - self.client_control[name].to(self.device)
                        parameter.grad.add_(correction)
                self.optimizer.step()
                self.local_steps += 1
                losses.append(float(loss.detach().cpu()))

        if self.local_steps == 0:
            raise RuntimeError("SCAFFOLD client performed zero local steps")
        scale = 1.0 / (self.local_steps * self.lr)
        new_control: TensorState = {}
        for name, parameter in self.model.named_parameters():
            value = (
                self.client_control[name].to(self.device)
                - self.server_control[name].to(self.device)
                + scale * (global_parameters[name] - parameter.detach())
            )
            new_control[name] = value.cpu().clone()
        self.control_delta = {
            name: new_control[name] - previous_control[name]
            for name in new_control
        }
        self.client_control = new_control
        return float(np.mean(losses)) if losses else float("nan")


class FusedSpaceFedClient:
    def __init__(
        self,
        client_id: str | int,
        loader: DataLoader,
        num_classes: int,
        in_channels: int,
        dz: int,
        task: str,
        device: torch.device,
        classifier_lr: float = 0.01,
        autoencoder_lr: float = 0.001,
    ):
        self.client_id = client_id
        self.loader = loader
        self.task = task
        self.device = device
        self.autoencoder = UNetSmallAE(in_channels, dz).to(device)
        self.classifier = ResNet20V2(num_classes, in_channels).to(device)
        self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=autoencoder_lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=classifier_lr)
        self.scaler = _new_scaler(device)

    @property
    def num_samples(self) -> int:
        return len(self.loader.dataset)

    def classifier_state(self) -> TensorState:
        return clone_state_dict(self.classifier.state_dict())

    def decoder_state(self) -> TensorState:
        return self.autoencoder.decoder_state()

    def encoder_state(self) -> TensorState:
        return self.autoencoder.encoder_state()

    def set_classifier_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.classifier.load_state_dict(move_state_dict(state, self.device))

    def set_decoder_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.autoencoder.load_decoder_state(state)

    def set_encoder_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.autoencoder.load_encoder_state(state)

    def set_full_autoencoder_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.autoencoder.load_state_dict(move_state_dict(state, self.device))

    def _set_trainable(self, encoder: bool, decoder: bool, classifier: bool) -> None:
        for parameter in self.autoencoder.encoder_parameters():
            parameter.requires_grad_(encoder)
        for parameter in self.autoencoder.decoder_parameters():
            parameter.requires_grad_(decoder)
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(classifier)

    def _warmup(self, epochs: int) -> List[float]:
        if epochs <= 0:
            return []
        self.autoencoder.train()
        self.classifier.eval()
        self._set_trainable(encoder=True, decoder=False, classifier=False)
        losses: List[float] = []
        for _ in range(epochs):
            for inputs, _ in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                self.ae_optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device):
                    reconstruction, _ = self.autoencoder(inputs)
                    loss = F.mse_loss(reconstruction, inputs)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.ae_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.ae_optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return losses

    def _joint_train(self, epochs: int) -> List[float]:
        self._set_trainable(encoder=True, decoder=True, classifier=True)
        self.autoencoder.train()
        self.classifier.train()
        losses: List[float] = []
        for _ in range(epochs):
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.ae_optimizer.zero_grad(set_to_none=True)
                self.classifier_optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device):
                    reconstruction, _ = self.autoencoder(inputs)
                    logits = self.classifier(inputs + reconstruction)
                    loss = task_loss(logits, targets, self.task)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.ae_optimizer)
                    self.scaler.step(self.classifier_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.ae_optimizer.step()
                    self.classifier_optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return losses

    def train_round(self, warmup_epochs: int, local_epochs: int) -> Dict[str, float]:
        reconstruction_losses = self._warmup(warmup_epochs)
        classification_losses = self._joint_train(local_epochs)
        return {
            "warmup_reconstruction_loss": float(np.mean(reconstruction_losses)) if reconstruction_losses else float("nan"),
            "classification_loss": float(np.mean(classification_losses)) if classification_losses else float("nan"),
        }


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
        }


@torch.no_grad()
def evaluate_logits(
    forward: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    task: str,
    device: torch.device,
) -> Metrics:
    targets_all: List[np.ndarray] = []
    predictions_all: List[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = forward(inputs)
        if task == "multilabel":
            predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
            targets_np = targets.to(torch.int64).cpu().numpy()
        else:
            predictions = logits.argmax(dim=1)
            targets_np = targets.reshape(-1).cpu().numpy()
        targets_all.append(targets_np)
        predictions_all.append(predictions.cpu().numpy())

    y_true = np.concatenate(targets_all)
    y_pred = np.concatenate(predictions_all)
    if task == "multilabel":
        accuracy = float((y_true == y_pred).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_label = []
        for column in range(y_true.shape[1]):
            per_label.append((y_true[:, column] == y_pred[:, column]).mean())
        balanced = float(np.mean(per_label))
    else:
        accuracy = float((y_true == y_pred).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return Metrics(accuracy, macro_f1, balanced)


def evaluate_model(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> Metrics:
    model.eval()
    return evaluate_logits(model, loader, task, device)


def evaluate_fused(client: FusedSpaceFedClient, loader: DataLoader) -> Metrics:
    client.autoencoder.eval()
    client.classifier.eval()

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        reconstruction, _ = client.autoencoder(inputs)
        return client.classifier(inputs + reconstruction)

    return evaluate_logits(forward, loader, client.task, client.device)


def mean_metrics(metrics: Sequence[Metrics], weights: Sequence[int] | None = None) -> Metrics:
    if not metrics:
        raise ValueError("No metrics to aggregate")
    if weights is None:
        normalized = np.full(len(metrics), 1.0 / len(metrics))
    else:
        values = np.asarray(weights, dtype=np.float64)
        normalized = values / values.sum()
    return Metrics(
        accuracy=float(sum(weight * metric.accuracy for weight, metric in zip(normalized, metrics))),
        macro_f1=float(sum(weight * metric.macro_f1 for weight, metric in zip(normalized, metrics))),
        balanced_accuracy=float(sum(weight * metric.balanced_accuracy for weight, metric in zip(normalized, metrics))),
    )


def classifier_gradient_vector(
    classifier: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Return the mean classifier-parameter gradient on deterministic batches."""
    classifier.eval()
    classifier.zero_grad(set_to_none=True)
    batches = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if transform is not None:
            with torch.no_grad():
                inputs = transform(inputs)
        logits = classifier(inputs)
        loss = task_loss(logits, targets, task)
        batch_examples = inputs.shape[0]
        (loss * batch_examples).backward()
        total_examples += batch_examples
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if total_examples == 0:
        raise RuntimeError("Cannot compute a gradient on an empty loader")
    gradients = []
    for parameter in classifier.parameters():
        if parameter.grad is None:
            gradients.append(torch.zeros_like(parameter).reshape(-1))
        else:
            gradients.append((parameter.grad / total_examples).reshape(-1))
    classifier.zero_grad(set_to_none=True)
    return torch.cat(gradients).detach().cpu()


def gradient_dissimilarity(gradients: Sequence[torch.Tensor]) -> float:
    if len(gradients) < 2:
        raise ValueError("At least two client gradients are required")
    matrix = torch.stack(list(gradients), dim=0)
    deviations = matrix - matrix.mean(dim=0, keepdim=True)
    return float(deviations.pow(2).sum(dim=1).mean())


def initial_server_control(model: nn.Module) -> TensorState:
    return {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in model.named_parameters()
    }


def update_server_control(
    server_control: Mapping[str, torch.Tensor],
    deltas: Sequence[Mapping[str, torch.Tensor]],
    active_clients: int,
    total_clients: int,
) -> TensorState:
    if not deltas:
        return clone_state_dict(server_control)
    participation_scale = active_clients / float(total_clients)
    updated: TensorState = {}
    for name, value in server_control.items():
        mean_delta = torch.stack([delta[name] for delta in deltas], dim=0).mean(dim=0)
        updated[name] = value + participation_scale * mean_delta
    return updated
