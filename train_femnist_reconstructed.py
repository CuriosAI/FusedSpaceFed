"""FusedSpaceFed on the frozen, reconstructed FEMNIST (150,3) benchmark.

The `smoke` subcommand trains for at most three rounds and never loads test
images. Definitive runs evaluate only after rounds 191--200. Checkpoints are
atomic round-boundary snapshots; an interrupted round is replayed on resume.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
import weakref

PROCESS_START = time.perf_counter()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from femnist_reconstructed_data import (
    REPO, FrozenPartition, atomic_json, canonical_hash, file_hash,
    require_untracked_artifacts,
)
from fusedspacefed_core import FusedSpaceFedClient, UNetSmallAE, clone_state_dict, weighted_average_states


class ReferenceMLP(nn.Module):
    """Historical layer dimensions, with logits for standard cross-entropy."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 64),
                                    nn.ReLU(), nn.Linear(64, 10))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
            raise ValueError("Reference MLP requires N x 1 x 28 x 28")
        return self.layers(inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_bytes(state: dict) -> int:
    return sum(value.numel() * value.element_size() for value in state.values())


def optimizer_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(value.numel() * value.element_size() for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))


class BenchmarkClient(FusedSpaceFedClient):
    def __init__(self, client_id: str, loader: DataLoader, settings: dict, device: torch.device):
        super().__init__(client_id, loader, 10, 1, settings["dz"], "multiclass", device,
                         settings["classifier_lr"], settings["autoencoder_lr"],
                         classifier=ReferenceMLP(), use_amp=False)
        weights, biases = [], []
        for name, parameter in self.classifier.named_parameters():
            (biases if name.endswith("bias") else weights).append(parameter)
        self.classifier_optimizer = torch.optim.SGD(
            [{"params": weights, "weight_decay": settings["classifier_weight_decay"]},
             {"params": biases, "weight_decay": settings["classifier_bias_decay"]}],
            lr=settings["classifier_lr"], momentum=settings["classifier_momentum"])
        self.ae_optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), lr=settings["autoencoder_lr"],
            betas=tuple(settings["autoencoder_betas"]), eps=settings["autoencoder_eps"],
            weight_decay=settings["autoencoder_weight_decay"])
        self.gradient_clip_norm = settings["gradient_clip_norm"]
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        self.phase = "classification"
        self.gradient_statistics = {}
        # Hooks must not keep clients/GPU tensors alive after a participation.
        owner = weakref.proxy(self)
        def before_step(optimizer, args, kwargs):
            owner._clip_gradients(optimizer, args, kwargs)
        def after_forward(module, inputs, outputs):
            owner._check_forward(module, inputs, outputs)
        self.ae_optimizer.register_step_pre_hook(before_step)
        self.classifier_optimizer.register_step_pre_hook(before_step)
        self.autoencoder.register_forward_hook(after_forward)
        self.classifier.register_forward_hook(after_forward)

    def _context(self) -> str:
        return f"client={self.client_id}, round={getattr(self, 'round_number', '?')}, phase={self.phase}"

    def _check_forward(self, module, inputs, outputs) -> None:
        tensors = outputs if isinstance(outputs, tuple) else (outputs,)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            component = "autoencoder" if module is self.autoencoder else "classifier"
            raise RuntimeError(f"Non-finite {component} output; {self._context()}")

    def _clip_gradients(self, optimizer, args, kwargs) -> None:
        """Clip each optimizer's active loss gradients before its normal update."""
        component = "autoencoder" if optimizer is self.ae_optimizer else "classifier"
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                      if parameter.requires_grad and parameter.grad is not None]
        if not parameters:
            raise RuntimeError(f"No active {component} gradients; {self._context()}")
        try:
            norm = float(nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm,
                                                 norm_type=2.0, error_if_nonfinite=True))
        except RuntimeError as error:
            raise RuntimeError(f"Invalid {component} gradient norm; {self._context()}: {error}") from error
        key = f"{self.phase}_{component}"
        stats = self.gradient_statistics.setdefault(
            key, {"steps": 0, "clipped_steps": 0, "sum_norm_before_clip": 0.0, "max_norm_before_clip": 0.0})
        stats["steps"] += 1
        stats["clipped_steps"] += int(norm > self.gradient_clip_norm)
        stats["sum_norm_before_clip"] += norm
        stats["max_norm_before_clip"] = max(stats["max_norm_before_clip"], norm)

    def _warmup(self, epochs: int) -> list[float]:
        self.phase = "warmup"
        return super()._warmup(epochs)

    def _joint_train(self, epochs: int) -> list[float]:
        self.phase = "classification"
        return super()._joint_train(epochs)

    def train_round(self, warmup_epochs: int, local_epochs: int) -> dict:
        self.gradient_statistics = {}
        synchronize(self.device)
        start = time.perf_counter()
        warmup = self._warmup(warmup_epochs)
        synchronize(self.device)
        middle = time.perf_counter()
        classification = self._joint_train(local_epochs)
        synchronize(self.device)
        end = time.perf_counter()
        if not warmup or not classification or not all(map(math.isfinite, warmup + classification)):
            raise RuntimeError("Empty phase or non-finite loss; refusing to checkpoint this round")
        return {"warmup_loss": float(np.mean(warmup)),
                "classification_loss": float(np.mean(classification)),
                "warmup_steps": len(warmup), "classification_steps": len(classification),
                "encoder_steps": len(warmup) + len(classification),
                "decoder_steps": len(classification), "classifier_steps": len(classification),
                "warmup_samples": warmup_epochs * self.num_samples,
                "classification_samples": local_epochs * self.num_samples,
                "warmup_seconds": middle - start, "classification_seconds": end - middle,
                "gradient_clipping": {
                    key: {"steps": value["steps"], "clipped_steps": value["clipped_steps"],
                          "max_norm_before_clip": value["max_norm_before_clip"],
                          "mean_norm_before_clip": value["sum_norm_before_clip"] / value["steps"]}
                    for key, value in self.gradient_statistics.items()},
                "optimizer_state_bytes": optimizer_bytes(self.ae_optimizer)
                                         + optimizer_bytes(self.classifier_optimizer)}


def aggregate_shared(classifiers: list[dict], decoders: list[dict]) -> tuple[dict, dict]:
    if any(not name.startswith(UNetSmallAE.DECODER_PREFIXES) for state in decoders for name in state):
        raise ValueError("Only decoder states may be aggregated with classifier states")
    if len(classifiers) != len(decoders):
        raise ValueError("Shared-state list lengths differ")
    weights = [1] * len(classifiers)
    averaged = weighted_average_states(classifiers, weights), weighted_average_states(decoders, weights)
    if any(not bool(torch.isfinite(value).all()) for state in averaged for value in state.values()):
        raise RuntimeError("Non-finite aggregated shared state; refusing to checkpoint this round")
    return averaged


def accuracy_metrics(counts: dict[str, dict]) -> dict:
    if not counts or any(not 0 <= c["correct"] <= c["total"] or c["total"] <= 0 for c in counts.values()):
        raise ValueError("Invalid per-client correct/total counts")
    total = sum(c["total"] for c in counts.values())
    correct = sum(c["correct"] for c in counts.values())
    return {"sample_weighted_accuracy_percent": 100.0 * correct / total,
            "uniform_client_accuracy_percent": float(np.mean([100.0 * c["correct"] / c["total"]
                                                               for c in counts.values()])),
            "correct": correct, "total": total, "clients": counts}


def evaluation_round(round_number: int, config: dict) -> bool:
    evaluation = config["evaluation"]
    return evaluation["first_round"] <= round_number <= evaluation["last_round"]


def final_statistics(history: list[dict], config: dict) -> dict | None:
    window = [row for row in history if evaluation_round(row["round"], config)]
    expected = list(range(config["evaluation"]["first_round"], config["evaluation"]["last_round"] + 1))
    if [row["round"] for row in window] != expected or any(row["evaluation"] is None for row in window):
        return None
    return {metric: float(np.mean([row["evaluation"][metric] for row in window]))
            for metric in ("sample_weighted_accuracy_percent", "uniform_client_accuracy_percent")}


def configure_runtime(device: torch.device, threads: int) -> None:
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and device.index is None):
        raise ValueError("Use cpu or an explicit CUDA device such as cuda:1")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_snapshot(device: torch.device, selection: np.random.Generator) -> dict:
    return {"python": random.getstate(), "numpy_legacy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "selection_pcg64": selection.bit_generator.state,
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(snapshot: dict, device: torch.device, selection: np.random.Generator) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy_legacy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    selection.bit_generator.state = snapshot["selection_pcg64"]
    if device.type == "cuda":
        torch.cuda.set_rng_state(snapshot["torch_cuda"], device)


def git_metadata() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    files = ["fusedspacefed_core.py", "femnist_reconstructed_data.py", "train_femnist_reconstructed.py"]
    return {"commit": commit, "working_tree_clean": clean,
            "code_sha256": {name: file_hash(REPO / name) for name in files}}


def runtime_metadata(device: torch.device) -> dict:
    metadata = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "device": str(device),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "tf32": False, "amp": False, "precision": "float32",
                "torch_threads": torch.get_num_threads(),
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(gpu=properties.name, gpu_total_bytes=properties.total_memory)
    return metadata


class BenchmarkRunner:
    """One run; exclusive directory ownership, resumable only at round boundaries."""
    def __init__(self, config: dict, partition: FrozenPartition, output: Path, seed: int,
                 device: str, mode: str = "definitive", rounds: int | None = None, resume: bool = False):
        if mode not in ("definitive", "smoke"):
            raise ValueError("Unknown run mode")
        self.config, self.partition, self.output = config, partition, Path(output)
        self.seed, self.device, self.mode = seed, torch.device(device), mode
        self.rounds = config["training"]["rounds"] if rounds is None else rounds
        if mode == "smoke" and not 1 <= self.rounds <= 3:
            raise ValueError("Smoke tests are limited to three rounds")
        configure_runtime(self.device, config["training"]["torch_threads"])
        if resume:
            if not self.output.is_dir():
                raise FileNotFoundError("Run directory to resume does not exist")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
        self._lock = (self.output / ".lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.selection = np.random.default_rng(seed)
            current_git = git_metadata()
            identity = {"benchmark": config["benchmark"], "config_sha256": canonical_hash(config),
                        "partition_sha256": partition.sha256, "seed": seed, "mode": mode,
                        "rounds": self.rounds, "device": str(self.device),
                        "code_sha256": current_git["code_sha256"]}
            if partition.manifest["recipe"] != config["data"]:
                raise ValueError("Partition recipe and run configuration differ")
            if resume:
                self.state = torch.load(self.output / "checkpoint.pt", map_location="cpu", weights_only=False)
                if self.state["identity"] != identity:
                    raise ValueError("Resume identity mismatch (config/partition/seed/mode/device/code)")
                if self.state["runtime"] != runtime_metadata(self.device):
                    raise ValueError("Resume software/hardware/runtime mismatch")
                snapshot_path = self.output / "results.json"
                if snapshot_path.exists():
                    snapshot = json.loads(snapshot_path.read_text())
                    if snapshot["completed_round"] > self.state["completed_round"]:
                        raise ValueError("Results are ahead of checkpoint")
                    if snapshot["completed_round"] == self.state["completed_round"]:
                        if snapshot["checkpoint_sha256"] != file_hash(self.output / "checkpoint.pt"):
                            raise ValueError("Checkpoint checksum mismatch")
                if [row["round"] for row in self.state["history"]] != list(range(1, self.state["completed_round"] + 1)):
                    raise ValueError("Invalid checkpoint round history")
                restore_rng(self.state["rng"], self.device, self.selection)
                self.state["resume_commits"].append(current_git["commit"])
            else:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                classifier = ReferenceMLP()
                torch.manual_seed(seed + 1_000_000)
                autoencoder = UNetSmallAE(1, config["training"]["dz"])
                encoder, decoder = autoencoder.encoder_state(), autoencoder.decoder_state()
                self.state = {"schema": 1, "identity": identity, "git": current_git,
                              "runtime": runtime_metadata(self.device), "config": config,
                              "resume_commits": [], "completed_round": 0, "history": [],
                              "rng_description": {"classifier_initialization_seed": seed,
                                                  "autoencoder_initialization_seed": seed + 1_000_000,
                                                  "selection": "Generator PCG64(run_seed)",
                                                  "batch_shuffle": "run_seed*10000000 + (round-1)*10000 + client_index"},
                              "global_classifier": clone_state_dict(classifier.state_dict()),
                              "global_decoder": decoder, "initial_encoder": encoder, "encoders": {},
                              "participations": {client: 0 for client in partition.clients},
                              "parameters": {
                                  "classifier": sum(p.numel() for p in classifier.parameters()),
                                  "private_encoder_per_client": sum(p.numel() for p in autoencoder.encoder_parameters()),
                                  "shared_decoder": sum(p.numel() for p in autoencoder.decoder_parameters()),
                                  "classifier_state_bytes": tensor_bytes(classifier.state_dict()),
                                  "encoder_state_bytes_per_client": tensor_bytes(encoder),
                                  "decoder_state_bytes": tensor_bytes(decoder)}}
                self.state["parameters"]["client_pipeline_total"] = (
                    self.state["parameters"]["classifier"] + self.state["parameters"]["private_encoder_per_client"]
                    + self.state["parameters"]["shared_decoder"])
                self.state["parameters"]["logical_private_encoder_bytes_all_clients"] = tensor_bytes(encoder) * len(partition.clients)
                self.save(reason="initial")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def make_client(self, client_id: str, round_number: int) -> BenchmarkClient:
        index = sorted(self.partition.clients).index(client_id)
        shuffle_seed = self.seed * 10_000_000 + (round_number - 1) * 10_000 + index
        generator = torch.Generator().manual_seed(shuffle_seed)
        loader = DataLoader(self.partition.dataset(client_id, "train"),
                            batch_size=self.config["training"]["batch_size"], shuffle=True,
                            generator=generator, num_workers=0, drop_last=False)
        client = BenchmarkClient(client_id, loader, self.config["training"], self.device)
        client.round_number = round_number
        client.set_encoder_state(self.state["encoders"].get(client_id, self.state["initial_encoder"]))
        client.set_decoder_state(self.state["global_decoder"])
        client.set_classifier_state(self.state["global_classifier"])
        return client

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.mode == "smoke":
            raise RuntimeError("Smoke mode must never evaluate the definitive test")
        devices = [self.device.index] if self.device.type == "cuda" else []
        # Constructor and DataLoader RNG consumption must not alter training.
        with torch.random.fork_rng(devices=devices):
            classifier = ReferenceMLP().to(self.device).eval()
            autoencoder = UNetSmallAE(1, self.config["training"]["dz"]).to(self.device).eval()
            classifier.load_state_dict(self.state["global_classifier"])
            autoencoder.load_decoder_state(self.state["global_decoder"])
            counts = {}
            for client_id in sorted(self.partition.clients):
                autoencoder.load_encoder_state(self.state["encoders"].get(client_id, self.state["initial_encoder"]))
                loader = DataLoader(self.partition.dataset(client_id, "test"),
                                    batch_size=self.config["training"]["batch_size"], shuffle=False, num_workers=0)
                correct, total = 0, 0
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    reconstruction, _ = autoencoder(inputs)
                    prediction = classifier(inputs + reconstruction).argmax(dim=1)
                    correct += int((prediction == targets).sum().item())
                    total += len(targets)
                counts[client_id] = {"correct": correct, "total": total,
                                     "participations": self.state["participations"][client_id]}
        return accuracy_metrics(counts)

    @torch.no_grad()
    def profile_training_inference(self) -> dict:
        """Timing proxy on one training batch/client; no predictions or accuracy."""
        if self.mode != "smoke":
            raise ValueError("Training-only inference timing is a smoke operation")
        devices = [self.device.index] if self.device.type == "cuda" else []
        load_seconds, forward_seconds, samples, batches = 0.0, 0.0, 0, 0
        with torch.random.fork_rng(devices=devices):
            classifier = ReferenceMLP().to(self.device).eval()
            autoencoder = UNetSmallAE(1, self.config["training"]["dz"]).to(self.device).eval()
            classifier.load_state_dict(self.state["global_classifier"])
            autoencoder.load_decoder_state(self.state["global_decoder"])
            for client_id in sorted(self.partition.clients):
                synchronize(self.device)
                start = time.perf_counter()
                autoencoder.load_encoder_state(self.state["encoders"].get(client_id, self.state["initial_encoder"]))
                synchronize(self.device)
                load_seconds += time.perf_counter() - start
                loader = DataLoader(self.partition.dataset(client_id, "train"),
                                    batch_size=self.config["training"]["batch_size"], shuffle=False, num_workers=0)
                inputs, _ = next(iter(loader))
                inputs = inputs.to(self.device)
                synchronize(self.device)
                start = time.perf_counter()
                reconstruction, _ = autoencoder(inputs)
                classifier(inputs + reconstruction)
                synchronize(self.device)
                forward_seconds += time.perf_counter() - start
                samples += len(inputs)
                batches += 1
        return {"split": "train", "accuracy_computed": False, "batches": batches, "samples": samples,
                "encoder_load_seconds": load_seconds, "forward_seconds": forward_seconds,
                "seconds_per_batch": forward_seconds / batches,
                "seconds_per_encoder_load": load_seconds / batches}

    def save(self, reason: str = "refresh") -> None:
        self.state["rng"] = rng_snapshot(self.device, self.selection)
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(dir=self.output, prefix="checkpoint.", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(self.state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.output / "checkpoint.pt")
        checkpoint_seconds = time.perf_counter() - start
        result = {key: self.state[key] for key in ("schema", "identity", "git", "runtime", "config",
                                                  "resume_commits", "completed_round", "history",
                                                  "participations", "parameters", "rng_description")}
        result["status"] = "completed" if self.state["completed_round"] == self.rounds else "paused"
        result["summary"] = final_statistics(self.state["history"], self.config) if self.mode == "definitive" else None
        result["checkpoint_bytes"] = (self.output / "checkpoint.pt").stat().st_size
        result["checkpoint_sha256"] = file_hash(self.output / "checkpoint.pt")
        result["last_checkpoint_seconds"] = checkpoint_seconds
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if self.device.type == "cuda":
            result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 1024**2
            result["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved(self.device) / 1024**2
        atomic_json(self.output / "results.json", result)
        event = {"reason": reason, "round": self.state["completed_round"],
                 "checkpoint_seconds": checkpoint_seconds, "total_io_seconds": time.perf_counter() - start,
                 "checkpoint_bytes": result["checkpoint_bytes"]}
        with (self.output / "timings.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def run(self, stop_after: int | None = None) -> dict:
        initial_round = self.state["completed_round"]
        stop = self.rounds if stop_after is None else min(stop_after, self.rounds)
        if stop < self.state["completed_round"]:
            raise ValueError("Cannot move a completed run backwards")
        for round_number in range(self.state["completed_round"] + 1, stop + 1):
            synchronize(self.device)
            start = time.perf_counter()
            active = self.selection.choice(sorted(self.partition.clients),
                                           self.config["training"]["clients_per_round"], replace=False).tolist()
            classifiers, decoders, local_records = [], [], []
            for client_id in active:
                local_start = time.perf_counter()
                client = self.make_client(client_id, round_number)
                metrics = client.train_round(self.config["training"]["warmup_epochs"],
                                             self.config["training"]["classification_epochs"])
                self.state["encoders"][client_id] = client.encoder_state()
                classifiers.append(client.classifier_state())
                decoders.append(client.decoder_state())
                self.state["participations"][client_id] += 1
                metrics.update(client=client_id, training_samples=client.num_samples,
                               participations=self.state["participations"][client_id],
                               total_seconds=time.perf_counter() - local_start)
                local_records.append(metrics)
                del client
            aggregation_start = time.perf_counter()
            classifier, decoder = aggregate_shared(classifiers, decoders)
            self.state["global_classifier"], self.state["global_decoder"] = classifier, decoder
            aggregation_seconds = time.perf_counter() - aggregation_start
            evaluation_start = time.perf_counter()
            evaluation = self.evaluate() if self.mode == "definitive" and evaluation_round(round_number, self.config) else None
            synchronize(self.device)
            record = {"round": round_number, "active_clients": active, "clients": local_records,
                      "aggregation_seconds": aggregation_seconds,
                      "evaluation_seconds": time.perf_counter() - evaluation_start if evaluation is not None else 0.0,
                      "evaluation": evaluation, "compute_seconds": time.perf_counter() - start,
                      "communication_bytes": 2 * len(active) * (
                          tensor_bytes(classifier) + tensor_bytes(decoder))}
            self.state["history"].append(record)
            self.state["completed_round"] = round_number
            self.save(reason="round")
            print(f"{self.mode}: round {round_number}/{self.rounds}, "
                  f"compute {record['compute_seconds']:.3f}s, checkpoint saved", flush=True)
        # Reconcile a results snapshot if a process stopped after checkpoint replacement.
        if self.state["completed_round"] == initial_round:
            self.save()
        return json.loads((self.output / "results.json").read_text())


def validate_definitive_config(config: dict, partition: FrozenPartition, seed: int) -> None:
    reference = json.loads((REPO / "configs/femnist_reconstructed.json").read_text())
    if config != reference or seed not in config["run_seeds"]:
        raise ValueError("Use the explicit approved configuration and one of seeds 41--45")
    if partition.manifest["recipe"] != reference["data"]:
        raise ValueError("Definitive run requires the approved frozen partition")
    if (partition.sha256 != reference["partition_sha256"]
            or partition.manifest["source"]["sha256"] != reference["source_archive_sha256"]):
        raise ValueError("Frozen partition/source hash does not match approved configuration")


def summarize_runs(config: dict, directories: list[Path], output: Path) -> dict:
    rows = [json.loads((directory / "results.json").read_text()) for directory in directories]
    if len(rows) != 5 or sorted(r["identity"]["seed"] for r in rows) != config["run_seeds"]:
        raise ValueError("Summary requires exactly five runs, seeds 41--45")
    if any(r["status"] != "completed" or r["identity"]["mode"] != "definitive" or r["summary"] is None for r in rows):
        raise ValueError("All runs must be completed definitive runs with the full reporting window")
    if any(r["completed_round"] != config["training"]["rounds"]
           or [h["round"] for h in r["history"] if evaluation_round(h["round"], config)]
           != list(range(config["evaluation"]["first_round"], config["evaluation"]["last_round"] + 1))
           for r in rows):
        raise ValueError("Incomplete round history/reporting window")
    for key in ("config_sha256", "partition_sha256", "code_sha256"):
        if any(r["identity"][key] != rows[0]["identity"][key] for r in rows):
            raise ValueError(f"Incompatible runs: {key}")
    if rows[0]["identity"]["config_sha256"] != canonical_hash(config):
        raise ValueError("Summary configuration mismatch")
    if rows[0]["identity"]["partition_sha256"] != config["partition_sha256"]:
        raise ValueError("Summary requires the frozen partition")
    summary = {"result_origin": "ours", "benchmark": config["benchmark"],
               "partition_sha256": rows[0]["identity"]["partition_sha256"], "seeds": config["run_seeds"],
               "uncertainty": "sample standard deviation across five run means, ddof=1", "unit": "percent"}
    for metric in ("sample_weighted_accuracy_percent", "uniform_client_accuracy_percent"):
        values = [row["summary"][metric] for row in sorted(rows, key=lambda r: r["identity"]["seed"])]
        summary[metric] = {"values": values, "mean": float(np.mean(values)),
                           "std": float(np.std(values, ddof=1))}
    if output.exists():
        raise FileExistsError("Refusing to overwrite summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--partition", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--seed", type=int, required=True)
        sub.add_argument("--device", required=True, help="Explicit device, e.g. cuda:1")
        sub.add_argument("--resume", action="store_true")
        if command == "smoke":
            sub.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
            sub.add_argument("--stop-after", type=int, choices=(1, 2, 3))
            sub.add_argument("--profile-training-inference", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--config", type=Path, required=True)
    summary.add_argument("--runs", type=Path, nargs=5, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    require_untracked_artifacts(args.output)
    if args.command == "summarize":
        print(json.dumps(summarize_runs(config, args.runs, args.output), indent=2))
        return
    partition = FrozenPartition(args.partition)
    validate_definitive_config(config, partition, args.seed)
    mode = "definitive" if args.command == "run" else "smoke"
    if mode == "definitive" and "smoke" in args.output.parts:
        parser.error("Definitive outputs must be separate from smoke artifacts")
    if mode == "smoke" and "smoke" not in args.output.parts:
        parser.error("Smoke output must be inside a directory named smoke")
    with BenchmarkRunner(config, partition, args.output, args.seed, args.device, mode,
                         getattr(args, "rounds", None), args.resume) as runner:
        result = runner.run(getattr(args, "stop_after", None))
        if getattr(args, "profile_training_inference", False):
            profile = runner.profile_training_inference()
            result["training_inference_profile"] = profile
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if runner.device.type == "cuda":
            result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(runner.device) / 1024**2
            result["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved(runner.device) / 1024**2
        result["last_session_wall_seconds"] = time.perf_counter() - PROCESS_START
        atomic_json(args.output / "results.json", result)
        print(json.dumps({key: result[key] for key in ("status", "completed_round", "summary", "parameters")}, indent=2))


if __name__ == "__main__":
    main()
