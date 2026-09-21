"""MedMNIST experiments for FusedSpaceFed and the comparison methods."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from fusedspacefed_core import (
    FedAvgClient,
    FedProxClient,
    FusedSpaceFedClient,
    ResNet20V2,
    ScaffoldClient,
    UNetSmallAE,
    classifier_gradient_vector,
    clone_state_dict,
    dirichlet_partition,
    evaluate_fused,
    evaluate_model,
    gradient_dissimilarity,
    initial_server_control,
    make_client_loaders,
    mean_metrics,
    pathological_partition,
    runtime_versions,
    seed_everything,
    update_server_control,
    weighted_average_states,
)


METHODS = ("fedavg", "fedprox", "scaffold", "fusedspacefed")
BEST_DZ = {
    "pathmnist": 16,
    "chestmnist": 32,
    "dermamnist": 16,
    "octmnist": 8,
    "pneumoniamnist": 16,
    "retinamnist": 64,
    "breastmnist": 32,
    "bloodmnist": 16,
    "tissuemnist": 32,
    "organamnist": 16,
    "organcmnist": 32,
    "organsmnist": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(BEST_DZ), default="pathmnist")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--partition", choices=("dirichlet", "pathological"), default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--classes-per-client", type=int, default=2)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--classifier-lr", type=float, default=0.01)
    parser.add_argument("--autoencoder-lr", type=float, default=0.001)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--dz", type=int, default=None, help="Defaults to the paper's Best dz for the dataset")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--source-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gamma-batches", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data/medmnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/medmnist"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace):
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    info = INFO[args.dataset]
    dataset_class = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    common = {
        "root": str(args.data_root),
        "transform": transform,
        "download": not args.no_download,
        "size": args.source_size,
    }
    train_set = dataset_class(split="train", **common)
    test_set = dataset_class(split="test", **common)
    task = "multilabel" if "multi-label" in info["task"] else "multiclass"
    return train_set, test_set, task, int(info["n_channels"]), len(info["label"])


def labels_for_partition(dataset, task: str) -> np.ndarray:
    labels = np.asarray(dataset.labels)
    if task != "multilabel":
        return labels.reshape(-1).astype(np.int64)
    labels = labels.astype(np.int64)
    no_finding = labels.sum(axis=1) == 0
    proxy = labels.argmax(axis=1)
    proxy[no_finding] = labels.shape[1]
    return proxy


def aggregate(states, sample_counts: Sequence[int], uniform: bool = True):
    weights = [1] * len(states) if uniform else list(sample_counts)
    return weighted_average_states(states, weights)


def build_clients(
    method: str,
    loaders,
    num_classes: int,
    in_channels: int,
    task: str,
    device: torch.device,
    args: argparse.Namespace,
    classifier_initial: Mapping[str, torch.Tensor],
    autoencoder_initial: Mapping[str, torch.Tensor],
):
    clients = []
    for client_id, loader in enumerate(loaders):
        common = dict(
            client_id=client_id,
            loader=loader,
            num_classes=num_classes,
            in_channels=in_channels,
            task=task,
            device=device,
            lr=args.classifier_lr,
        )
        if method == "fedavg":
            client = FedAvgClient(**common)
            client.set_state(classifier_initial)
        elif method == "fedprox":
            client = FedProxClient(**common, mu=args.fedprox_mu)
            client.set_state(classifier_initial)
        elif method == "scaffold":
            client = ScaffoldClient(**common)
            client.set_state(classifier_initial)
        else:
            common.pop("lr")
            client = FusedSpaceFedClient(
                **common,
                dz=args.dz,
                classifier_lr=args.classifier_lr,
                autoencoder_lr=args.autoencoder_lr,
            )
            client.set_classifier_state(classifier_initial)
            client.set_full_autoencoder_state(autoencoder_initial)
        clients.append(client)
    return clients


def run_method(
    method: str,
    loaders,
    test_loader,
    task: str,
    in_channels: int,
    num_classes: int,
    args: argparse.Namespace,
    seed: int,
) -> Dict:
    seed_everything(seed)
    device = torch.device(args.device)
    classifier_template = ResNet20V2(num_classes, in_channels)
    classifier_initial = clone_state_dict(classifier_template.state_dict())
    seed_everything(seed + 1_000_000)
    ae_template = UNetSmallAE(in_channels, args.dz)
    autoencoder_initial = clone_state_dict(ae_template.state_dict())
    clients = build_clients(
        method,
        loaders,
        num_classes,
        in_channels,
        task,
        device,
        args,
        classifier_initial,
        autoencoder_initial,
    )
    counts = [client.num_samples for client in clients]
    history: List[Dict] = []

    if method == "fusedspacefed":
        global_classifier = clone_state_dict(classifier_initial)
        global_decoder = ae_template.decoder_state()
        for round_index in range(args.rounds):
            round_losses = []
            for client in clients:
                client.set_classifier_state(global_classifier)
                client.set_decoder_state(global_decoder)
                round_losses.append(client.train_round(args.warmup_epochs, args.local_epochs))
            global_classifier = aggregate([client.classifier_state() for client in clients], counts)
            global_decoder = aggregate([client.decoder_state() for client in clients], counts)
            history.append({
                "round": round_index + 1,
                "warmup_reconstruction_loss": float(np.mean([x["warmup_reconstruction_loss"] for x in round_losses])),
                "classification_loss": float(np.mean([x["classification_loss"] for x in round_losses])),
            })
            print(f"[{method}] seed={seed} round={round_index + 1}/{args.rounds} "
                  f"loss={history[-1]['classification_loss']:.6f}")

        for client in clients:
            client.set_classifier_state(global_classifier)
            client.set_decoder_state(global_decoder)
        test_metrics = mean_metrics([evaluate_fused(client, test_loader) for client in clients])
        gradients = []
        for client in clients:
            def fused_transform(inputs, current=client):
                reconstruction, _ = current.autoencoder(inputs)
                return inputs + reconstruction
            gradients.append(classifier_gradient_vector(
                client.classifier,
                client.loader,
                task,
                device,
                transform=fused_transform,
                max_batches=args.gamma_batches,
            ))
    else:
        global_state = clone_state_dict(classifier_initial)
        server_control = initial_server_control(classifier_template) if method == "scaffold" else None
        for round_index in range(args.rounds):
            losses = []
            control_deltas = []
            for client in clients:
                if method == "fedprox":
                    client.set_global_state(global_state)
                else:
                    client.set_state(global_state)
                if method == "scaffold":
                    client.set_server_control(server_control)
                losses.append(client.train(args.local_epochs))
                if method == "scaffold":
                    control_deltas.append(client.control_delta)
            global_state = aggregate([client.state() for client in clients], counts)
            if method == "scaffold":
                server_control = update_server_control(server_control, control_deltas, len(clients), len(clients))
            history.append({"round": round_index + 1, "classification_loss": float(np.mean(losses))})
            print(f"[{method}] seed={seed} round={round_index + 1}/{args.rounds} "
                  f"loss={history[-1]['classification_loss']:.6f}")

        global_model = ResNet20V2(num_classes, in_channels).to(device)
        global_model.load_state_dict({name: value.to(device) for name, value in global_state.items()})
        test_metrics = evaluate_model(global_model, test_loader, task, device)
        gradients = [classifier_gradient_vector(
            global_model,
            client.loader,
            task,
            device,
            max_batches=args.gamma_batches,
        ) for client in clients]

    return {
        "method": method,
        "seed": seed,
        "test_metrics": test_metrics.as_dict(),
        "gradient_dissimilarity": gradient_dissimilarity(gradients),
        "history": history,
    }


def summarize(runs: Sequence[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for method in sorted({run["method"] for run in runs}):
        selected = [run for run in runs if run["method"] == method]
        fields = {
            "accuracy": [run["test_metrics"]["accuracy"] for run in selected],
            "macro_f1": [run["test_metrics"]["macro_f1"] for run in selected],
            "balanced_accuracy": [run["test_metrics"]["balanced_accuracy"] for run in selected],
            "gradient_dissimilarity": [run["gradient_dissimilarity"] for run in selected],
        }
        summary[method] = {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
            for name, values in fields.items()
        }
    return summary


def main() -> None:
    args = parse_args()
    args.dz = args.dz if args.dz is not None else BEST_DZ[args.dataset]
    train_set, test_set, task, in_channels, num_classes = load_dataset(args)
    partition_labels = labels_for_partition(train_set, task)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    runs = []
    partition_audits = {}
    for seed in args.seeds:
        if args.partition == "dirichlet":
            partitions = dirichlet_partition(partition_labels, args.clients, args.alpha, seed)
        else:
            partitions = pathological_partition(
                partition_labels, args.clients, args.classes_per_client, seed
            )
        partition_audits[str(seed)] = {
            "client_sample_counts": [len(part) for part in partitions],
            "assigned_samples": int(sum(map(len, partitions))),
            "unique_assigned_samples": int(len(set(index for part in partitions for index in part))),
        }
        for method in args.methods:
            loaders = make_client_loaders(
                train_set, partitions, args.batch_size, seed, True, args.num_workers
            )
            runs.append(run_method(
                method, loaders, test_loader, task, in_channels, num_classes, args, seed
            ))

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "paper-aligned-v1",
        "runtime": runtime_versions(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "task": task,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "partition_audits": partition_audits,
        "runs": runs,
        "summary": summarize(runs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.dataset}_{args.partition}_alpha-{args.alpha:g}.json"
    output_path = args.output_dir / filename
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
