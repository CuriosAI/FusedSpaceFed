"""Deterministic validation views of the frozen FEMNIST training arrays.

The original partition remains unchanged. Calibration verifies and reads only
its training arrays; test array files are never opened, hashed, or mapped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from femnist_reconstructed_data import (
    FrozenPartition, OriginDataset, canonical_hash, file_hash,
)


SPLIT_RULE = "sha256_rank_within_client_class_v1"
TRAIN_FILES = ("train_images.npy", "train_labels.npy")


def _validate_split_parameters(seed: int, numerator: int, denominator: int) -> None:
    if any(type(value) is not int for value in (seed, numerator, denominator)):
        raise ValueError("Validation seed and fraction must be integers")
    if not 0 < numerator < denominator:
        raise ValueError("Validation fraction must lie strictly between zero and one")


def _verified_training_partition(directory: Path) -> tuple[FrozenPartition, tuple]:
    # This verifies the canonical frozen manifest and global origin/index
    # disjointness, including test metadata, without accessing array files.
    partition = FrozenPartition(directory, verify_files=False)
    metadata = partition.manifest["files"]
    if not all(filename in metadata for filename in TRAIN_FILES):
        raise ValueError("Missing frozen training array metadata")
    arrays = []
    total = partition.manifest["statistics"]["train"]["total"]
    for filename in TRAIN_FILES:
        path = partition.directory / filename
        expected = metadata[filename]
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Training array checksum mismatch: {filename}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
            raise ValueError("Training array shape/dtype differs from manifest")
        image_array = filename == "train_images.npy"
        expected_shape = (total, 1, 28, 28) if image_array else (total,)
        expected_dtype = np.uint8 if image_array else np.int64
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise ValueError("Invalid training input/label dimensions or dtype")
        if not image_array:
            for client in partition.clients.values():
                if any(int(array[row["array_index"]]) != row["label"] for row in client["train"]):
                    raise ValueError("Stored training labels differ from origin manifest")
        arrays.append(array)
    return partition, tuple(arrays)


def _view_statistics(clients: list[dict], num_classes: int) -> dict:
    statistics = {}
    for split in ("train", "validation"):
        sizes = [client[f"num_{split}"] for client in clients]
        statistics[split] = {
            "total": sum(sizes), "mean": float(np.mean(sizes)),
            "min": min(sizes), "max": max(sizes),
            "per_class": [sum(client[f"{split}_per_class"][label] for client in clients)
                          for label in range(num_classes)],
        }
    return statistics


def _split_manifest(partition: FrozenPartition, seed: int, numerator: int,
                    denominator: int) -> dict:
    _validate_split_parameters(seed, numerator, denominator)
    num_classes = len(partition.manifest["recipe"]["classes"])
    clients = []
    for client_id in sorted(partition.clients):
        original = partition.clients[client_id]
        validation_ids = set()
        for label in sorted(original["classes"]):
            rows = [row for row in original["train"] if row["label"] == label]
            count = len(rows) * numerator // denominator
            if not 0 < count < len(rows):
                raise ValueError(f"Validation split must leave every class nonempty in both views: "
                                 f"client={client_id}, class={label}, training_count={len(rows)}")
            # canonical_hash uses SHA256 over an unambiguous canonical JSON
            # tuple. It consumes no Python, NumPy, or Torch random generator.
            ranked = sorted(rows, key=lambda row: (
                canonical_hash([seed, client_id, label, row["origin_id"]]), row["origin_id"]))
            validation_ids.update(row["origin_id"] for row in ranked[:count])
        client = {"id": client_id, "classes": list(original["classes"]),
                  "train": [], "validation": []}
        for row in original["train"]:
            split = "validation" if row["origin_id"] in validation_ids else "train"
            client[split].append({key: row[key] for key in ("origin_id", "array_index", "label")})
        for split in ("train", "validation"):
            counts = [0] * num_classes
            for row in client[split]:
                counts[row["label"]] += 1
            client[f"num_{split}"] = len(client[split])
            client[f"{split}_per_class"] = counts
        clients.append(client)
    manifest = {
        "schema": 1, "kind": "femnist_training_validation",
        "parent_partition_sha256": partition.sha256,
        "source_train_files": {filename: dict(partition.manifest["files"][filename])
                               for filename in TRAIN_FILES},
        "split": {"rule": SPLIT_RULE, "seed": seed, "numerator": numerator,
                  "denominator": denominator, "rounding": "floor_per_client_class"},
        "clients": clients, "statistics": _view_statistics(clients, num_classes),
    }
    manifest["view_sha256"] = canonical_hash(manifest)
    return manifest


def _validate_manifest(partition: FrozenPartition, manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid validation manifest")
    unhashed = {key: value for key, value in manifest.items() if key != "view_sha256"}
    if canonical_hash(unhashed) != manifest.get("view_sha256"):
        raise ValueError("Validation manifest hash mismatch")
    try:
        specification = manifest["split"]
        expected = _split_manifest(partition, specification["seed"], specification["numerator"],
                                   specification["denominator"])
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid validation split specification") from error
    if manifest != expected:
        raise ValueError("Validation manifest differs from its frozen parent or deterministic split")


def prepare_validation_split(partition_directory: Path, destination: Path,
                             seed: int = 20261004, numerator: int = 1,
                             denominator: int = 5) -> dict:
    """Publish an immutable JSON split manifest; reuse only an identical one."""
    _validate_split_parameters(seed, numerator, denominator)
    partition, _ = _verified_training_partition(Path(partition_directory))
    manifest = _split_manifest(partition, seed, numerator, denominator)
    destination = Path(destination)
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError("Validation split destination is already occupied")
        existing = json.loads(destination.read_text())
        _validate_manifest(partition, existing)
        if existing != manifest:
            raise FileExistsError("Existing validation split has different seed/fraction/partition")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write((json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
            # Atomic creation without overwriting a concurrently published split.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                existing = json.loads(destination.read_text())
                _validate_manifest(partition, existing)
                if existing != manifest:
                    raise FileExistsError("Existing validation split has different seed/fraction/partition")
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


class TrainingValidationPartition:
    """Fit/holdout datasets backed exclusively by verified original train arrays."""

    access_policy = "training_arrays_only"

    def __init__(self, partition_directory: Path, split_path: Path):
        partition, self._training_arrays = _verified_training_partition(Path(partition_directory))
        self.directory = partition.directory
        self.manifest, self.clients = partition.manifest, partition.clients
        self.validation_manifest = json.loads(Path(split_path).read_text())
        _validate_manifest(partition, self.validation_manifest)
        self._views = {client["id"]: client for client in self.validation_manifest["clients"]}

    @property
    def sha256(self) -> str:
        return self.manifest["partition_sha256"]

    @property
    def view_sha256(self) -> str:
        return self.validation_manifest["view_sha256"]

    def dataset(self, client_id: str, split: str) -> OriginDataset:
        if split == "test":
            raise ValueError("Calibration partition must never access the definitive test")
        if split not in ("train", "validation"):
            raise ValueError("Unknown calibration split")
        images, labels = self._training_arrays
        return OriginDataset(images, labels, self._views[client_id][split])
