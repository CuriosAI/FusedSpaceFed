"""Frozen NIST a-j partition for the reconstructed FedRep comparison.

No historical code is executed. Sampling uses sorted origin IDs, independent
Python/PCG64 streams and monotonically advancing class cursors (no reuse).
Insufficient class pools use the explicitly approved integer proportional rule.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parent


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Replace a snapshot owned by this operation, never a new run directory."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_untracked_artifacts(root: Path) -> None:
    if root.resolve().is_relative_to(REPO):
        relative = (root.resolve() / "artifact").relative_to(REPO)
        if subprocess.run(["git", "check-ignore", "-q", str(relative)], cwd=REPO).returncode:
            raise ValueError("Artifact directory must be ignored by Git or outside the repository")


def download_archive(source: dict, directory: Path) -> Path:
    """Reuse a verified archive; resume a partial HTTP download when supported."""
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "by_class.zip"
    receipt = directory / "by_class.download.json"
    expected_size = source["source_bytes"]
    if archive.exists():
        if not receipt.exists():
            raise FileExistsError("Existing archive has no receipt; refusing to replace it")
        metadata = json.loads(receipt.read_text())
        if (metadata["url"] != source["source_url"] or archive.stat().st_size != expected_size
                or file_hash(archive) != metadata["sha256"]):
            raise ValueError("Cached archive does not match its receipt/source")
        return archive
    partial = directory / "by_class.zip.part"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError("Partial download is larger than the declared source")
    if shutil.disk_usage(directory).free < expected_size - offset + 256 * 1024**2:
        raise OSError("Insufficient disk space for NIST archive and prepared arrays")
    if offset < expected_size:
        request = urllib.request.Request(source["source_url"],
                                         headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and response.status != 206:
                print("Server ignored Range; restarting the incomplete download", flush=True)
                offset = 0
            if response.status == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Incorrect HTTP range response")
            received = offset
            report_at = received + 64 * 1024**2
            with partial.open("ab" if offset else "wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
                    received += len(block)
                    if received >= report_at:
                        print(f"NIST download: {received}/{expected_size} bytes", flush=True)
                        report_at = received + 64 * 1024**2
                handle.flush()
                os.fsync(handle.fileno())
    if partial.stat().st_size != expected_size or not zipfile.is_zipfile(partial):
        raise ValueError("Incomplete or invalid archive; partial file retained for inspection/resume")
    digest = file_hash(partial)
    # The receipt permits reuse after interruption between receipt and rename.
    atomic_json(receipt, {"url": source["source_url"], "bytes": expected_size, "sha256": digest})
    os.rename(partial, archive)
    return archive


def proportional_quotas(requests: dict[str, int], available: int) -> dict[str, int]:
    """Hamilton allocation: integer products, descending remainder, ascending ID."""
    if not isinstance(available, int) or available < 0 or any(not isinstance(n, int) or n < 0 for n in requests.values()):
        raise ValueError("Availability and requests must be nonnegative integers")
    demand = sum(requests.values())
    if demand <= available:
        return dict(requests)
    assigned = {client: (n * available) // demand for client, n in requests.items()}
    order = sorted(requests, key=lambda client: (-(requests[client] * available % demand), client))
    for client in order[:available - sum(assigned.values())]:
        assigned[client] += 1
    assert sum(assigned.values()) == available
    assert all(0 <= assigned[client] <= requests[client] for client in requests)
    return assigned


def plan_partition(pools: dict[int, list[str]], recipe: dict) -> tuple[list[dict], dict]:
    """Keep original draws; reduce only insufficient pools without image reuse."""
    if recipe["allocation"] != "proportional_largest_remainder_v1":
        raise ValueError("Unknown allocation rule; no implicit fallback")
    num_classes = len(recipe["classes"])
    if not 0 < recipe["classes_per_client"] <= num_classes:
        raise ValueError("Invalid number of assigned classes")
    if set(pools) != set(range(num_classes)):
        raise ValueError("Missing a-j class pool")
    if len({x for pool in pools.values() for x in pool}) != sum(map(len, pools.values())):
        raise ValueError("Duplicate origin IDs in input pools")
    pool_rng = random.Random(recipe["seed"])
    selected = {}
    for label in range(num_classes):
        ids = sorted(pools[label])
        pool_rng.shuffle(ids)
        selected[label] = ids[:recipe["pool_per_class"]]
    draws = np.random.default_rng(recipe["seed"]).lognormal(
        recipe["lognormal_mean"], recipe["lognormal_sigma"], recipe["num_clients"]
    ) + recipe["count_offset"]
    clients = []
    for index, draw in enumerate(draws):
        n_per_class = max(int(draw / recipe["classes_per_client"]), 2)
        labels = [(index + j) % num_classes for j in range(recipe["classes_per_client"])]
        clients.append({"id": f"f_{index:05d}", "classes": labels, "count_draw": float(draw),
                        "original_per_class_request": n_per_class,
                        "requested_per_class": [n_per_class if c in labels else 0 for c in range(num_classes)],
                        "assigned_per_class": [0] * num_classes})
    pool_info = {}
    for label in range(num_classes):
        requests = {client["id"]: client["requested_per_class"][label] for client in clients}
        demand, capacity = sum(requests.values()), len(selected[label])
        assigned = proportional_quotas(requests, capacity)
        for client in clients:
            client["assigned_per_class"][label] = assigned[client["id"]]
        allocated = min(demand, capacity)
        pool_info[str(label)] = {
            "available": len(pools[label]), "selected": capacity, "requested": demand,
            "used": allocated, "reduced": demand > capacity,
            "reduction_numerator": allocated if demand else 1,
            "reduction_denominator": demand if demand else 1,
            "reduction_factor": allocated / demand if demand else 1.0,
            "selected_ids_sha256": canonical_hash(selected[label])}
    cursors = [0] * num_classes
    split_rng = random.Random(recipe["seed"] + 1)
    for client in clients:
        labels = client["classes"]
        if any(client["assigned_per_class"][label] <= 0 for label in labels):
            raise ValueError(f"Capacity reduction removes an assigned class from {client['id']}; stopping")
        examples = []
        for label in labels:
            start, end = cursors[label], cursors[label] + client["assigned_per_class"][label]
            if end > len(selected[label]):
                raise ValueError(f"Pool exhausted: client {client['id']}, class {label}, "
                                 f"need cumulative {end}, available {len(selected[label])}; no reuse")
            examples.extend({"origin_id": origin, "label": label}
                            for origin in selected[label][start:end])
            cursors[label] = end
        split_rng.shuffle(examples)
        train_len = int(recipe["train_fraction"] * len(examples))
        if not 0 < train_len < len(examples):
            raise ValueError("Empty local train/test split")
        client.update(requested_total=sum(client["requested_per_class"]),
                      assigned_total=len(examples), train=examples[:train_len], test=examples[train_len:])
    assert all(cursors[c] == pool_info[str(c)]["used"] for c in range(num_classes))
    return clients, pool_info


def preprocess_png(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        grayscale = image.convert("L")
        grayscale.thumbnail((28, 28), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.uint8).copy()
    if pixels.shape != (28, 28):
        raise ValueError(f"Unexpected thumbnail shape {pixels.shape}; no implicit padding/resize")
    return pixels[np.newaxis]


def partition_statistics(clients: list[dict], num_classes: int) -> dict:
    statistics = {}
    for split in ("train", "test"):
        sizes = [len(client[split]) for client in clients]
        statistics[split] = {"total": sum(sizes), "mean": float(np.mean(sizes)),
                             "min": min(sizes), "max": max(sizes),
                             "per_class": [sum(row["label"] == label for client in clients
                                               for row in client[split]) for label in range(num_classes)]}
    for name, field in (("before_split", "assigned_per_class"), ("original_requests", "requested_per_class")):
        sizes = [sum(client[field]) for client in clients]
        statistics[name] = {"total": sum(sizes), "mean": float(np.mean(sizes)),
                            "min": min(sizes), "max": max(sizes),
                            "per_class": [sum(client[field][c] for client in clients) for c in range(num_classes)]}
    statistics["clients_with_reduced_quotas"] = sum(client["requested_per_class"] != client["assigned_per_class"]
                                                     for client in clients)
    return statistics


def prepare_partition(archive: Path, destination: Path, config: dict) -> dict:
    recipe = config["data"]
    if destination.exists():
        existing = FrozenPartition(destination)
        if existing.manifest["recipe"] != recipe or existing.manifest["source"]["sha256"] != file_hash(archive):
            raise FileExistsError("Existing partition belongs to different data/configuration")
        print("Reusing verified frozen partition", flush=True)
        return existing.manifest
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        pools = {}
        for label, character in enumerate(recipe["classes"]):
            hexadecimal = f"{ord(character):02x}"
            prefix = f"by_class/{hexadecimal}/train_{hexadecimal}/"
            pools[label] = [name for name in names if name.startswith(prefix) and name.endswith(".png")]
        clients, pool_info = plan_partition(pools, recipe)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix=destination.name + ".pending-", dir=destination.parent))
        files = {}
        for split in ("train", "test"):
            count = sum(len(client[split]) for client in clients)
            images = np.empty((count, 1, 28, 28), dtype=np.uint8)
            labels = np.empty(count, dtype=np.int64)
            index = 0
            for client in clients:
                class_counts = [0] * len(recipe["classes"])
                for row in client[split]:
                    content = source.read(row["origin_id"])  # ZipFile verifies member CRC.
                    images[index] = preprocess_png(content)
                    labels[index] = row["label"]
                    row["png_sha256"] = hashlib.sha256(content).hexdigest()
                    row["array_index"] = index
                    class_counts[row["label"]] += 1
                    index += 1
                client[f"num_{split}"] = len(client[split])
                client[f"{split}_per_class"] = class_counts
            for name, array in ((f"{split}_images.npy", images), (f"{split}_labels.npy", labels)):
                np.save(pending / name, array, allow_pickle=False)
                files[name] = {"sha256": file_hash(pending / name), "shape": list(array.shape),
                               "dtype": str(array.dtype)}
            print(f"Prepared {split}: {count} images", flush=True)
    manifest = {"schema": 1, "benchmark": config["benchmark"], "recipe": recipe,
                "source": {"url": recipe["source_url"], "bytes": archive.stat().st_size,
                           "sha256": file_hash(archive)},
                "rng": {"pool": "Python random.Random(seed)", "counts": "NumPy Generator PCG64(seed)",
                        "split": "Python random.Random(seed+1)", "numpy": np.__version__},
                "preprocessing": {"pillow": pillow_version, "filter": "LANCZOS thumbnail",
                                  "stored_dtype": "uint8", "input": "float32/255, 1x28x28"},
                "pools": pool_info, "clients": clients, "files": files,
                "statistics": partition_statistics(clients, len(recipe["classes"]))}
    manifest["partition_sha256"] = canonical_hash(manifest)
    atomic_json(pending / "manifest.json", manifest)
    FrozenPartition(pending)  # Audit before publishing the directory.
    os.rename(pending, destination)
    return manifest


class FrozenPartition:
    def __init__(self, directory: Path, verify_files: bool = True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        expected_hash = self.manifest["partition_sha256"]
        unhashed = {k: v for k, v in self.manifest.items() if k != "partition_sha256"}
        if canonical_hash(unhashed) != expected_hash:
            raise ValueError("Partition manifest hash mismatch")
        self.clients = {client["id"]: client for client in self.manifest["clients"]}
        if len(self.clients) != self.manifest["recipe"]["num_clients"]:
            raise ValueError("Client count/IDs do not match recipe")
        all_ids = [row["origin_id"] for client in self.clients.values()
                   for split in ("train", "test") for row in client[split]]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Origin IDs overlap within/between clients or train/test")
        for split in ("train", "test"):
            indices = [row["array_index"] for client in self.clients.values() for row in client[split]]
            if sorted(indices) != list(range(self.manifest["statistics"][split]["total"])):
                raise ValueError("Array indices are not a bijection")
        for client in self.clients.values():
            labels = client["classes"]
            if len(labels) != self.manifest["recipe"]["classes_per_client"] or len(set(labels)) != len(labels):
                raise ValueError("Client must retain its assigned classes")
            counts = [0] * len(self.manifest["recipe"]["classes"])
            for split in ("train", "test"):
                if not client[split]:
                    raise ValueError("Empty local split")
                for row in client[split]:
                    counts[row["label"]] += 1
            if counts != client["assigned_per_class"] or {c for c, n in enumerate(counts) if n} != set(labels):
                raise ValueError("Actual examples do not match approved quotas/classes")
        if verify_files:
            expected_files = {f"{split}_{kind}.npy" for split in ("train", "test") for kind in ("images", "labels")}
            if set(self.manifest["files"]) != expected_files:
                raise ValueError("Missing or unexpected partition arrays")
            for filename, metadata in self.manifest["files"].items():
                if file_hash(self.directory / filename) != metadata["sha256"]:
                    raise ValueError(f"Partition array checksum mismatch: {filename}")
                array = np.load(self.directory / filename, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
                    raise ValueError("Array shape/dtype differs from manifest")
                split = filename.split("_", 1)[0]
                total = self.manifest["statistics"][split]["total"]
                expected_shape = (total, 1, 28, 28) if "images" in filename else (total,)
                expected_dtype = np.uint8 if "images" in filename else np.int64
                if array.shape != expected_shape or array.dtype != expected_dtype:
                    raise ValueError("Invalid input/label dimensions or dtype")
                if "labels" in filename:
                    for client in self.clients.values():
                        if any(int(array[row["array_index"]]) != row["label"] for row in client[split]):
                            raise ValueError("Stored labels differ from origin manifest")
        self._arrays = {}

    @property
    def sha256(self) -> str:
        return self.manifest["partition_sha256"]

    def dataset(self, client_id: str, split: str) -> Dataset:
        if split not in ("train", "test"):
            raise ValueError("Unknown split")
        if split not in self._arrays:
            self._arrays[split] = (
                np.load(self.directory / f"{split}_images.npy", mmap_mode="r", allow_pickle=False),
                np.load(self.directory / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False))
        images, labels = self._arrays[split]
        return OriginDataset(images, labels, self.clients[client_id][split])


class OriginDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, rows: list[dict]):
        self.images, self.labels, self.rows = images, labels, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.rows[index]["array_index"]
        image = torch.from_numpy(self.images[position].copy()).to(torch.float32).div_(255)
        return image, torch.tensor(int(self.labels[position]), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    require_untracked_artifacts(args.root)
    config = json.loads(args.config.read_text())
    archive = args.root / "source/by_class.zip"
    if args.download or archive.exists():
        archive = download_archive(config["data"], archive.parent)
    elif not archive.exists():
        parser.error("NIST archive missing; pass --download to fetch it")
    manifest = prepare_partition(archive, args.root / "partition", config)
    print(json.dumps({"partition_sha256": manifest["partition_sha256"],
                      "statistics": manifest["statistics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
