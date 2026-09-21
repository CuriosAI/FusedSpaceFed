"""LEAF FEMNIST experiments for FusedSpaceFed and the comparison methods."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from fusedspacefed_core import (
    FedAvgClient,
    FedProxClient,
    FusedSpaceFedClient,
    Metrics,
    ResNet20V2,
    ScaffoldClient,
    UNetSmallAE,
    classifier_gradient_vector,
    clone_state_dict,
    evaluate_model,
    gradient_dissimilarity,
    initial_server_control,
    runtime_versions,
    seed_everything,
    update_server_control,
    weighted_average_states,
)


METHODS = ("fedavg", "fedprox", "scaffold", "fusedspacefed")


class FEMNISTWriterDataset(Dataset):
    def __init__(self, examples: Mapping):
        self.images = examples["x"]
        self.labels = examples["y"]
        if len(self.images) != len(self.labels):
            raise ValueError("FEMNIST x/y lengths do not match")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = torch.as_tensor(self.images[index], dtype=torch.float32).reshape(1, 28, 28)
        image = F.interpolate(
            image.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze(0)
        return image, int(self.labels[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-root", type=Path, required=True,
                        help="Directory containing data/train/*.json and data/test/*.json")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-clients", type=int, default=3400)
    parser.add_argument("--clients-per-round", type=int, default=340,
                        help="The manuscript samples a subset; 340 is 10%% of 3400 writers")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--classifier-lr", type=float, default=0.01)
    parser.add_argument("--autoencoder-lr", type=float, default=0.001)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--dz", type=int, default=64)
    parser.add_argument("--aggregation", choices=("uniform", "samples"), default="uniform")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--client-selection-seed", type=int, default=41)
    parser.add_argument("--gamma-clients", type=int, default=50)
    parser.add_argument("--gamma-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/femnist"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_leaf_split(directory: Path) -> Dict[str, FEMNISTWriterDataset]:
    if not directory.is_dir():
        raise FileNotFoundError(f"LEAF split directory not found: {directory}")
    user_data: Dict[str, Mapping] = {}
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON shards found in: {directory}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for user in payload["users"]:
            if user in user_data:
                raise ValueError(f"Duplicate LEAF user {user!r} in {directory}")
            user_data[user] = payload["user_data"][user]
    return {user: FEMNISTWriterDataset(examples) for user, examples in user_data.items()}


def make_loader(dataset: Dataset, args: argparse.Namespace, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def aggregate(states, sample_counts: Sequence[int], args: argparse.Namespace):
    weights = [1] * len(states) if args.aggregation == "uniform" else list(sample_counts)
    return weighted_average_states(states, weights)


def new_baseline_client(method, user, loader, args, device):
    common = dict(
        client_id=user,
        loader=loader,
        num_classes=62,
        in_channels=1,
        task="multiclass",
        device=device,
        lr=args.classifier_lr,
    )
    if method == "fedprox":
        return FedProxClient(**common, mu=args.fedprox_mu)
    if method == "scaffold":
        return ScaffoldClient(**common)
    return FedAvgClient(**common)


@torch.no_grad()
def evaluate_fused_by_writer(
    users: Sequence[str],
    train_sets: Mapping[str, Dataset],
    test_sets: Mapping[str, Dataset],
    encoder_states: Mapping[str, Mapping[str, torch.Tensor]],
    initial_autoencoder: Mapping[str, torch.Tensor],
    global_decoder: Mapping[str, torch.Tensor],
    global_classifier: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Metrics:
    probe_loader = make_loader(train_sets[users[0]], args, 0, False)
    probe = FusedSpaceFedClient(
        users[0], probe_loader, 62, 1, args.dz, "multiclass", device,
        args.classifier_lr, args.autoencoder_lr,
    )
    probe.set_full_autoencoder_state(initial_autoencoder)
    probe.set_decoder_state(global_decoder)
    probe.set_classifier_state(global_classifier)
    probe.autoencoder.eval()
    probe.classifier.eval()
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    for user_index, user in enumerate(users):
        if user not in test_sets or len(test_sets[user]) == 0:
            continue
        probe.set_encoder_state(encoder_states.get(user, {
            name: value for name, value in initial_autoencoder.items()
            if name.startswith(UNetSmallAE.ENCODER_PREFIXES)
        }))
        loader = make_loader(test_sets[user], args, user_index, False)
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            reconstruction, _ = probe.autoencoder(inputs)
            predictions = probe.classifier(inputs + reconstruction).argmax(dim=1)
            all_targets.append(targets.numpy())
            all_predictions.append(predictions.cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    return Metrics(
        accuracy=float((y_true == y_pred).mean()),
        macro_f1=float(f1_score(y_true, y_pred, labels=np.arange(62), average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )


def fused_gamma(
    users: Sequence[str],
    train_sets: Mapping[str, Dataset],
    encoder_states,
    initial_autoencoder,
    global_decoder,
    global_classifier,
    args,
    device,
    seed,
) -> float:
    rng = np.random.default_rng(seed + 991)
    chosen = rng.choice(users, size=min(args.gamma_clients, len(users)), replace=False).tolist()
    probe_loader = make_loader(train_sets[chosen[0]], args, seed, False)
    probe = FusedSpaceFedClient(
        chosen[0], probe_loader, 62, 1, args.dz, "multiclass", device,
        args.classifier_lr, args.autoencoder_lr,
    )
    probe.set_full_autoencoder_state(initial_autoencoder)
    probe.set_decoder_state(global_decoder)
    probe.set_classifier_state(global_classifier)
    gradients = []
    initial_encoder = {
        name: value for name, value in initial_autoencoder.items()
        if name.startswith(UNetSmallAE.ENCODER_PREFIXES)
    }
    for index, user in enumerate(chosen):
        probe.loader = make_loader(train_sets[user], args, seed * 100_000 + index, False)
        probe.set_encoder_state(encoder_states.get(user, initial_encoder))

        def transform(inputs):
            reconstruction, _ = probe.autoencoder(inputs)
            return inputs + reconstruction

        gradients.append(classifier_gradient_vector(
            probe.classifier, probe.loader, "multiclass", device, transform, args.gamma_batches
        ))
    return gradient_dissimilarity(gradients)


def baseline_gamma(users, train_sets, global_state, args, device, seed) -> float:
    rng = np.random.default_rng(seed + 991)
    chosen = rng.choice(users, size=min(args.gamma_clients, len(users)), replace=False).tolist()
    model = ResNet20V2(62, 1).to(device)
    model.load_state_dict({name: value.to(device) for name, value in global_state.items()})
    gradients = []
    for index, user in enumerate(chosen):
        loader = make_loader(train_sets[user], args, seed * 100_000 + index, False)
        gradients.append(classifier_gradient_vector(
            model, loader, "multiclass", device, max_batches=args.gamma_batches
        ))
    return gradient_dissimilarity(gradients)


def run_method(method, users, train_sets, test_sets, args, seed) -> Dict:
    seed_everything(seed)
    device = torch.device(args.device)
    classifier_template = ResNet20V2(62, 1)
    classifier_initial = clone_state_dict(classifier_template.state_dict())
    seed_everything(seed + 1_000_000)
    ae_template = UNetSmallAE(1, args.dz)
    autoencoder_initial = clone_state_dict(ae_template.state_dict())
    rng = np.random.default_rng(seed)
    history = []

    if method == "fusedspacefed":
        global_classifier = clone_state_dict(classifier_initial)
        global_decoder = ae_template.decoder_state()
        encoder_states: Dict[str, Mapping[str, torch.Tensor]] = {}
        for round_index in range(args.rounds):
            active = rng.choice(users, size=min(args.clients_per_round, len(users)), replace=False).tolist()
            losses = []
            classifier_states = []
            decoder_states = []
            sample_counts = []
            for offset, user in enumerate(active):
                loader = make_loader(train_sets[user], args, seed * 10_000_000 + round_index * 10_000 + offset, True)
                client = FusedSpaceFedClient(
                    user, loader, 62, 1, args.dz, "multiclass", device,
                    args.classifier_lr, args.autoencoder_lr,
                )
                client.set_full_autoencoder_state(autoencoder_initial)
                if user in encoder_states:
                    client.set_encoder_state(encoder_states[user])
                client.set_decoder_state(global_decoder)
                client.set_classifier_state(global_classifier)
                losses.append(client.train_round(args.warmup_epochs, args.local_epochs))
                encoder_states[user] = client.encoder_state()
                classifier_states.append(client.classifier_state())
                decoder_states.append(client.decoder_state())
                sample_counts.append(client.num_samples)
                del client
            global_classifier = aggregate(classifier_states, sample_counts, args)
            global_decoder = aggregate(decoder_states, sample_counts, args)
            history.append({
                "round": round_index + 1,
                "active_clients": active,
                "warmup_reconstruction_loss": float(np.mean([x["warmup_reconstruction_loss"] for x in losses])),
                "classification_loss": float(np.mean([x["classification_loss"] for x in losses])),
            })
            print(f"[{method}] seed={seed} round={round_index + 1}/{args.rounds} "
                  f"loss={history[-1]['classification_loss']:.6f}")
        metrics = evaluate_fused_by_writer(
            users, train_sets, test_sets, encoder_states, autoencoder_initial,
            global_decoder, global_classifier, args, device,
        )
        gamma = fused_gamma(
            users, train_sets, encoder_states, autoencoder_initial, global_decoder,
            global_classifier, args, device, seed,
        )
    else:
        global_state = clone_state_dict(classifier_initial)
        server_control = initial_server_control(classifier_template) if method == "scaffold" else None
        client_controls: Dict[str, Mapping[str, torch.Tensor]] = {}
        for round_index in range(args.rounds):
            active = rng.choice(users, size=min(args.clients_per_round, len(users)), replace=False).tolist()
            losses = []
            deltas = []
            client_states = []
            sample_counts = []
            for offset, user in enumerate(active):
                loader = make_loader(train_sets[user], args, seed * 10_000_000 + round_index * 10_000 + offset, True)
                client = new_baseline_client(method, user, loader, args, device)
                if method == "fedprox":
                    client.set_global_state(global_state)
                else:
                    client.set_state(global_state)
                if method == "scaffold":
                    if user in client_controls:
                        client.set_client_control(client_controls[user])
                    client.set_server_control(server_control)
                losses.append(client.train(args.local_epochs))
                if method == "scaffold":
                    client_controls[user] = clone_state_dict(client.client_control)
                    deltas.append(client.control_delta)
                client_states.append(client.state())
                sample_counts.append(client.num_samples)
                del client
            global_state = aggregate(client_states, sample_counts, args)
            if method == "scaffold":
                server_control = update_server_control(
                    server_control, deltas, len(active), len(users)
                )
            history.append({
                "round": round_index + 1,
                "active_clients": active,
                "classification_loss": float(np.mean(losses)),
            })
            print(f"[{method}] seed={seed} round={round_index + 1}/{args.rounds} "
                  f"loss={history[-1]['classification_loss']:.6f}")
        global_model = ResNet20V2(62, 1).to(device)
        global_model.load_state_dict({name: value.to(device) for name, value in global_state.items()})
        test_data = ConcatDataset([test_sets[user] for user in users if user in test_sets])
        test_loader = make_loader(test_data, args, seed, False)
        metrics = evaluate_model(global_model, test_loader, "multiclass", device)
        gamma = baseline_gamma(users, train_sets, global_state, args, device, seed)

    return {
        "method": method,
        "seed": seed,
        "test_metrics": metrics.as_dict(),
        "gradient_dissimilarity": gamma,
        "history": history,
    }


def summarize(runs: Sequence[Dict]) -> Dict:
    result = {}
    for method in sorted({run["method"] for run in runs}):
        selected = [run for run in runs if run["method"] == method]
        fields = {
            "accuracy": [run["test_metrics"]["accuracy"] for run in selected],
            "macro_f1": [run["test_metrics"]["macro_f1"] for run in selected],
            "balanced_accuracy": [run["test_metrics"]["balanced_accuracy"] for run in selected],
            "gradient_dissimilarity": [run["gradient_dissimilarity"] for run in selected],
        }
        result[method] = {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
            for name, values in fields.items()
        }
    return result


def main() -> None:
    args = parse_args()
    train_sets = read_leaf_split(args.leaf_root / "data" / "train")
    test_sets = read_leaf_split(args.leaf_root / "data" / "test")
    eligible = sorted(user for user in train_sets if user in test_sets and len(train_sets[user]) > 0)
    rng = np.random.default_rng(args.client_selection_seed)
    rng.shuffle(eligible)
    if args.max_clients > 0:
        eligible = eligible[: args.max_clients]
    if len(eligible) < 2:
        raise RuntimeError("At least two writers shared by train and test splits are required")
    if args.clients_per_round < 1:
        raise ValueError("--clients-per-round must be positive")
    runs = []
    for seed in args.seeds:
        for method in args.methods:
            runs.append(run_method(method, eligible, train_sets, test_sets, args, seed))
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "paper-aligned-v1",
        "runtime": runtime_versions(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "num_selected_writers": len(eligible),
        "num_train_examples": int(sum(len(train_sets[user]) for user in eligible)),
        "num_test_examples": int(sum(len(test_sets[user]) for user in eligible)),
        "runs": runs,
        "summary": summarize(runs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "femnist.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
