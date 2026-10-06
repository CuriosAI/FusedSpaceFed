"""Fixed PathMNIST run with complete, CPU-loadable diagnostic checkpoints.

The scientific client, losses, AMP, optimizers and aggregation are reused
unchanged from fusedspacefed_core. Each GPU owns five persistent clients.
Only independent local updates are parallelized; the server aggregates in
client-ID order once all ten updates have finished. PIL preprocessing and
standard DataLoader/RandomSampler ordering are preserved by a resident cache.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusedspacefed_core import (  # noqa: E402
    FusedSpaceFedClient, ResNet20V2, UNetSmallAE, clone_state_dict,
    evaluate_fused, pathological_partition, seed_everything,
    weighted_average_states,
)

PUBLIC = ROOT / "research/pathmnist_pathological"
PRIVATE = ROOT / "_local/pathmnist_pathological"
SOURCES = ("fusedspacefed_core.py", "train_medmnist.py",
           "research/pathmnist_pathological/run.py",
           "research/pathmnist_pathological/config.json")


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite checkpoint tensor")
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("Non-finite numeric value")


def rng_state(cuda=False):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.get_rng_state()}
    if cuda:
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if "torch_cuda" in value:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


class OriginIndices(Dataset):
    def __init__(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.indices[index]


class ResidentCollate:
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __call__(self, origin_indices):
        indices = torch.tensor(origin_indices, dtype=torch.long, device=self.images.device)
        return self.images[indices], self.labels[indices]


def resident_loader(images, labels, indices, config, client_id, shuffle=True):
    return DataLoader(OriginIndices(indices), batch_size=config["batch_size"],
                      shuffle=shuffle, num_workers=0, pin_memory=False,
                      generator=torch.Generator().manual_seed(config["seed"] * 10000 + client_id),
                      collate_fn=ResidentCollate(images, labels))


def make_client(client_id, loader, config, device, classifier, autoencoder):
    client = FusedSpaceFedClient(client_id, loader, config["num_classes"],
                                config["in_channels"], config["dz"], "multiclass", device,
                                classifier_lr=config["classifier_lr"],
                                autoencoder_lr=config["autoencoder_lr"],
                                use_amp=config["use_amp"])
    client.set_classifier_state(classifier)
    client.set_full_autoencoder_state(autoencoder)
    return client


def client_snapshot(client):
    value = {
        "client_id": client.client_id,
        "classifier": client.classifier_state(),
        "autoencoder": clone_state_dict(client.autoencoder.state_dict()),
        "encoder": client.encoder_state(), "decoder": client.decoder_state(),
        "classifier_optimizer": cpu_copy(client.classifier_optimizer.state_dict()),
        "ae_optimizer": cpu_copy(client.ae_optimizer.state_dict()),
        "scaler": cpu_copy(client.scaler.state_dict()) if client.scaler else None,
        "loader_generator": client.loader.generator.get_state(),
        "modes": {"classifier": client.classifier.training, "autoencoder": client.autoencoder.training},
        "requires_grad": {"classifier": {n: p.requires_grad for n, p in client.classifier.named_parameters()},
                          "autoencoder": {n: p.requires_grad for n, p in client.autoencoder.named_parameters()}},
    }
    assert_finite(value)
    return value


def restore_client(client, value):
    client.set_classifier_state(value["classifier"])
    client.set_full_autoencoder_state(value["autoencoder"])
    client.classifier_optimizer.load_state_dict(value["classifier_optimizer"])
    client.ae_optimizer.load_state_dict(value["ae_optimizer"])
    if client.scaler:
        client.scaler.load_state_dict(value["scaler"])
    client.loader.generator.set_state(value["loader_generator"])
    client.classifier.train(value["modes"]["classifier"])
    client.autoencoder.train(value["modes"]["autoencoder"])
    for component in ("classifier", "autoencoder"):
        for name, parameter in getattr(client, component).named_parameters():
            parameter.requires_grad_(value["requires_grad"][component][name])


def prepare(config):
    from medmnist import INFO, PathMNIST
    import medmnist
    destination = PRIVATE / "prepared"
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        for name, digest in manifest["files_sha256"].items():
            if file_hash(destination / name) != digest:
                raise ValueError("Prepared file changed: " + name)
        if manifest["config_sha256"] != json_hash(config):
            raise ValueError("Prepared configuration differs")
        return manifest
    source = PRIVATE / "data/pathmnist_64.npz"
    info = INFO["pathmnist"]
    if file_hash(source, "md5") != info["MD5_64"]:
        raise ValueError("Source MD5 differs from official MedMNIST metadata")
    destination.mkdir()
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    manifest = {"medmnist_version": medmnist.__version__, "source_url": info["url_64"],
                "source_md5": info["MD5_64"], "source_sha256": file_hash(source),
                "config_sha256": json_hash(config), "preprocessing": "PIL Resize(32,32) then ToTensor; no augmentation",
                "splits": {}, "files_sha256": {}}
    partitions = None
    for split in ("train", "test"):
        dataset = PathMNIST(split=split, root=str(source.parent), size=64, download=False, transform=transform)
        labels = np.asarray(dataset.labels).reshape(-1).astype(np.int64)
        cache = np.lib.format.open_memmap(destination / (split + "-images.npy"), mode="w+",
                                        dtype=np.float32, shape=(len(dataset), 3, 32, 32))
        for index in range(len(dataset)):
            cache[index] = dataset[index][0].numpy()
        cache.flush()
        del cache
        np.save(destination / (split + "-labels.npy"), labels)
        manifest["splits"][split] = {"samples": len(dataset), "class_counts": np.bincount(labels, minlength=9).tolist()}
        if split == "train":
            partitions = pathological_partition(labels, config["clients"], config["classes_per_client"], config["seed"])
            flat = [i for part in partitions for i in part]
            assert sorted(flat) == list(range(len(dataset)))
            manifest["clients"] = [{"client_id": i, "samples": len(part),
                                     "classes": np.unique(labels[part]).tolist(),
                                     "class_counts": np.bincount(labels[part], minlength=9).tolist()}
                                    for i, part in enumerate(partitions)]
            assert all(len(row["classes"]) == 2 for row in manifest["clients"])
        print("Prepared", split, len(dataset), flush=True)
        del dataset
    write_json(destination / "partitions.json", {"seed": config["seed"], "index_space": "official PathMNIST-64 train array row",
                                               "clients": {str(i): part for i, part in enumerate(partitions)}})
    for path in sorted(destination.iterdir()):
        manifest["files_sha256"][path.name] = file_hash(path)
    manifest["partition_sha256"] = manifest["files_sha256"]["partitions.json"]
    write_json(destination / "manifest.json", manifest)
    return manifest


def send(pipe, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    pipe.send_bytes(buffer.getvalue())


def receive(pipe):
    value = torch.load(io.BytesIO(pipe.recv_bytes()), map_location="cpu", weights_only=False)
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def worker(pipe, physical_gpu, ids, config, classifier, autoencoder, resume):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        torch.set_num_threads(config["torch_threads_per_worker"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        seed_everything(config["seed"] + 1_000_000)
        prepared = PRIVATE / "prepared"
        parts = json.loads((prepared / "partitions.json").read_text())["clients"]
        images = torch.from_numpy(np.load(prepared / "train-images.npy")).to(device)
        labels = torch.from_numpy(np.load(prepared / "train-labels.npy")).to(device)
        clients = [make_client(i, resident_loader(images, labels, parts[str(i)], config, i),
                               config, device, classifier, autoencoder) for i in ids]
        if resume:
            for client in clients:
                restore_client(client, resume["clients"][str(client.client_id)])
            restore_rng(resume["workers"][str(physical_gpu)]["rng"])
        send(pipe, {"clients": {str(c.client_id): client_snapshot(c) for c in clients},
                    "rng": rng_state(True), "gpu_name": torch.cuda.get_device_name(device)})
        while True:
            command = receive(pipe)
            if command["op"] == "stop":
                break
            for client in clients:
                client.set_classifier_state(command["classifier"])
                client.set_decoder_state(command["decoder"])
            if command["op"] == "round":
                torch.cuda.reset_peak_memory_stats(device)
                records = []
                for client in clients:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    losses = client.train_round(config["warmup_epochs"], config["local_epochs"])
                    torch.cuda.synchronize(device)
                    assert_finite(losses)
                    steps = math.ceil(client.num_samples / config["batch_size"])
                    records.append({"client_id": client.client_id, **losses,
                                    "warmup_steps": steps * config["warmup_epochs"],
                                    "classification_steps": steps * config["local_epochs"],
                                    "seconds": time.perf_counter() - start})
                send(pipe, {"clients": {str(c.client_id): client_snapshot(c) for c in clients},
                            "records": records, "rng": rng_state(True),
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                            "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            elif command["op"] == "snapshot":
                send(pipe, {"clients": {str(c.client_id): client_snapshot(c) for c in clients}, "rng": rng_state(True)})
            elif command["op"] == "evaluate":
                if command["round"] != config["rounds"]:
                    raise ValueError("Test evaluation is allowed only after round 50")
                torch.cuda.reset_peak_memory_stats(device)
                test_images = torch.from_numpy(np.load(prepared / "test-images.npy")).to(device)
                test_labels = torch.from_numpy(np.load(prepared / "test-labels.npy")).to(device)
                loader = resident_loader(test_images, test_labels, range(len(test_labels)), config, 100, shuffle=False)
                metrics = []
                for client in clients:
                    start = time.perf_counter()
                    result = evaluate_fused(client, loader).as_dict()
                    result.update(client_id=client.client_id, samples=len(test_labels),
                                  correct=int(round(result["accuracy"] * len(test_labels))),
                                  seconds=time.perf_counter() - start)
                    assert_finite(result)
                    metrics.append(result)
                send(pipe, {"metrics": metrics,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            else:
                raise ValueError("Unknown worker operation")
    except BaseException:
        try:
            send(pipe, {"error": traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        pipe.close()


def code_identity():
    return {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: file_hash(ROOT / name) for name in SOURCES},
            "note": "Runner is identified by source hashes before the final archive commit."}


def checkpoint(config, manifest, identity, classifier, decoder, replies, history, round_index, elapsed):
    clients = {key: value for reply in replies.values() for key, value in reply["clients"].items()}
    value = {"format": "pathmnist-diagnostic-v1", "seed": config["seed"], "round": round_index,
             "config": config, "config_sha256": json_hash(config), "data_manifest": manifest,
             "partition_sha256": manifest["partition_sha256"],
             "partitions": json.loads((PRIVATE / "prepared/partitions.json").read_text()),
             "code": identity, "classifier": classifier, "decoder": decoder,
             "encoders": {key: client["encoder"] for key, client in clients.items()},
             "clients": clients, "workers": {str(key): {"rng": reply["rng"]} for key, reply in replies.items()},
             "coordinator_rng": rng_state(), "history": history, "elapsed_training_seconds": elapsed,
             "shared_state_note": "Top-level classifier/decoder are server states. Client copies in latest.pt may be pre-aggregation; final.pt copies are synchronized.",
             "rng_note": "CUDA_VISIBLE_DEVICES maps each worker's cuda:0 to its recorded physical GPU."}
    assert_finite(value)
    return value


def run(config, resume=False):
    torch.set_num_threads(1)
    manifest = prepare(config)
    output = PRIVATE / "seed-42"
    if resume:
        old = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        if old["config_sha256"] != json_hash(config) or old["partition_sha256"] != manifest["partition_sha256"]:
            raise ValueError("Resume identity differs")
        if old["code"]["source_sha256"] != code_identity()["source_sha256"]:
            raise ValueError("Source code changed since checkpoint")
        if (output / "results.json").exists():
            raise FileExistsError("Completed run must not be resumed")
        identity, history = old["code"], old["history"]
        start_round, prior_elapsed = old["round"], old["elapsed_training_seconds"]
        classifier, decoder = old["classifier"], old["decoder"]
        autoencoder = {**old["clients"]["0"]["encoder"], **decoder}
        restore_rng(old["coordinator_rng"])
    else:
        output.mkdir(exist_ok=False)
        identity = code_identity()
        seed_everything(config["seed"])
        classifier = clone_state_dict(ResNet20V2(9, 3).state_dict())
        seed_everything(config["seed"] + 1_000_000)
        ae = UNetSmallAE(3, config["dz"])
        autoencoder, decoder = clone_state_dict(ae.state_dict()), ae.decoder_state()
        old, history, start_round, prior_elapsed = None, [], 0, 0.0
        write_json(output / "identity.json", {"config": config, "config_sha256": json_hash(config),
                                             "data": manifest, "code": identity})
    ctx = mp.get_context("spawn")
    pipes, processes = {}, {}
    groups = {gpu: [] for gpu in config["devices"]}
    loads = {gpu: 0 for gpu in config["devices"]}
    for row in sorted(manifest["clients"], key=lambda row: (-row["samples"], row["client_id"])):
        gpu = min(config["devices"], key=lambda gpu: (loads[gpu], gpu))
        groups[gpu].append(row["client_id"])
        loads[gpu] += math.ceil(row["samples"] / config["batch_size"])
    try:
        for gpu in config["devices"]:
            ids = sorted(groups[gpu])
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(child, gpu, ids, config, classifier, autoencoder, old))
            process.start()
            child.close()
            pipes[gpu], processes[gpu] = parent, process
        replies = {gpu: receive(pipe) for gpu, pipe in pipes.items()}
        hardware = {str(gpu): {"name": reply["gpu_name"],
                              "clients": sorted(map(int, reply["clients"]))} for gpu, reply in replies.items()}
        import medmnist, torchvision
        write_json(output / "runtime.json", {"python": sys.version, "torch": torch.__version__,
                                            "torchvision": torchvision.__version__, "medmnist": medmnist.__version__,
                                            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                                            "gpus": hardware, "started_at_utc": datetime.now(timezone.utc).isoformat()})
        if not resume:
            atomic_save(output / "initial.pt", checkpoint(config, manifest, identity, classifier, decoder, replies, [], 0, 0.0))
        begin = time.perf_counter()
        for round_index in range(start_round + 1, config["rounds"] + 1):
            if code_identity()["source_sha256"] != identity["source_sha256"]:
                raise ValueError("Scientific sources changed during the run")
            start = time.perf_counter()
            for pipe in pipes.values():
                send(pipe, {"op": "round", "classifier": classifier, "decoder": decoder})
            replies = {gpu: receive(pipe) for gpu, pipe in pipes.items()}
            clients = {k: v for reply in replies.values() for k, v in reply["clients"].items()}
            classifier = weighted_average_states([clients[str(i)]["classifier"] for i in range(config["clients"])], [1] * config["clients"])
            decoder = weighted_average_states([clients[str(i)]["decoder"] for i in range(config["clients"])], [1] * config["clients"])
            records = sorted([row for reply in replies.values() for row in reply["records"]], key=lambda row: row["client_id"])
            timing = {"round": round_index, "seconds": time.perf_counter() - start,
                      "clients": records, "gpus": {str(gpu): {k: v for k, v in reply.items() if k.startswith(("cuda_", "host_"))} for gpu, reply in replies.items()}}
            history.append({"round": round_index,
                            "warmup_reconstruction_loss": float(np.mean([row["warmup_reconstruction_loss"] for row in records])),
                            "classification_loss": float(np.mean([row["classification_loss"] for row in records]))})
            elapsed = prior_elapsed + time.perf_counter() - begin
            atomic_save(output / "latest.pt", checkpoint(config, manifest, identity, classifier, decoder, replies, history, round_index, elapsed))
            with (output / "timings.jsonl").open("a") as stream:
                stream.write(json.dumps(timing, allow_nan=False) + "\n")
            print(json.dumps({**history[-1], "seconds": timing["seconds"],
                              "elapsed_seconds": elapsed, "estimated_remaining_seconds":
                              (time.perf_counter() - begin) / (round_index - start_round) * (config["rounds"] - round_index)}, allow_nan=False), flush=True)
        for pipe in pipes.values():
            send(pipe, {"op": "snapshot", "classifier": classifier, "decoder": decoder})
        replies = {gpu: receive(pipe) for gpu, pipe in pipes.items()}
        elapsed = prior_elapsed + time.perf_counter() - begin
        final = checkpoint(config, manifest, identity, classifier, decoder, replies, history, config["rounds"], elapsed)
        atomic_save(output / "final.pt", final)
        eval_start = time.perf_counter()
        for pipe in pipes.values():
            send(pipe, {"op": "evaluate", "classifier": classifier, "decoder": decoder, "round": config["rounds"]})
        evaluations = {gpu: receive(pipe) for gpu, pipe in pipes.items()}
        metrics = sorted([row for reply in evaluations.values() for row in reply["metrics"]], key=lambda row: row["client_id"])
        aggregate = {field: float(np.mean([row[field] for row in metrics])) for field in ("accuracy", "macro_f1", "balanced_accuracy")}
        result = {"status": "completed", "seed": config["seed"], "rounds": config["rounds"],
                  "test_evaluations": 1, "test_round": config["rounds"], "test_samples_per_pipeline": 7180,
                  "test_metrics": aggregate, "client_test_metrics": metrics, "history": history,
                  "elapsed_training_seconds": elapsed, "evaluation_seconds": time.perf_counter() - eval_start,
                  "session_seconds": time.perf_counter() - begin, "resumed": resume,
                  "evaluation_memory": {str(gpu): {k: v for k, v in reply.items() if k.startswith("cuda_")} for gpu, reply in evaluations.items()},
                  "identity": json.loads((output / "identity.json").read_text()),
                  "checkpoint_sha256": {name: file_hash(output / name) for name in ("initial.pt", "final.pt")}}
        assert_finite(result)
        write_json(output / "results.json", result)
        print(json.dumps({"completed": result["test_metrics"], "training_seconds": elapsed}, allow_nan=False), flush=True)
    finally:
        for gpu, pipe in pipes.items():
            if processes[gpu].is_alive():
                try:
                    send(pipe, {"op": "stop"})
                except (EOFError, BrokenPipeError):
                    pass
            pipe.close()
        for process in processes.values():
            process.join(timeout=30)


def verify():
    torch.set_num_threads(1)
    output = PRIVATE / "seed-42"
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "completed" and result["rounds"] == 50
    assert result["test_evaluations"] == 1 and result["test_round"] == 50
    prepared = PRIVATE / "prepared"
    labels = np.load(prepared / "train-labels.npy")
    images = torch.from_numpy(np.load(prepared / "train-images.npy", mmap_mode="c")[0:2].copy())
    checks = {}
    for name, expected_round in (("initial.pt", 0), ("final.pt", 50)):
        saved = torch.load(output / name, map_location="cpu", weights_only=False)
        assert saved["round"] == expected_round and saved["seed"] == 42
        assert saved["config_sha256"] == json_hash(saved["config"])
        assert saved["code"]["source_sha256"] == code_identity()["source_sha256"]
        assert file_hash(output / name) == result["checkpoint_sha256"][name]
        assert set(saved["encoders"]) == set(saved["clients"]) == {str(i) for i in range(10)}
        parts = saved["partitions"]["clients"]
        assert sorted(i for part in parts.values() for i in part) == list(range(len(labels)))
        model = ResNet20V2(9, 3)
        model.load_state_dict(saved["classifier"], strict=True)
        model.eval()
        bn_keys = [key for key in saved["classifier"] if key.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        assert bn_keys
        for i in range(10):
            key = str(i)
            assert len(np.unique(labels[parts[key]])) == 2
            ae = UNetSmallAE(3, 16)
            ae.load_state_dict({**saved["encoders"][key], **saved["decoder"]}, strict=True)
            ae.eval()
            with torch.no_grad():
                reconstruction, _ = ae(images)
                logits = model(images + reconstruction)
            assert reconstruction.shape == images.shape and logits.shape == (2, 9)
            assert_finite((reconstruction, logits))
            loader = resident_loader(images, torch.zeros(2, dtype=torch.long), [0, 1], saved["config"], i)
            restored = make_client(i, loader, saved["config"], torch.device("cpu"), saved["classifier"], ae.state_dict())
            restore_client(restored, saved["clients"][key])
            assert restored.ae_optimizer.state_dict()["param_groups"] == saved["clients"][key]["ae_optimizer"]["param_groups"]
            # GPU AMP scaler dictionaries are validated without requiring CUDA on the reviewer machine.
            assert saved["clients"][key]["scaler"] is not None
            assert len(saved["clients"][key]["ae_optimizer"]["state"]) > 0 if expected_round else not saved["clients"][key]["ae_optimizer"]["state"]
            assert torch.equal(restored.loader.generator.get_state(), saved["clients"][key]["loader_generator"])
            if expected_round:
                assert all(torch.equal(saved["clients"][key]["classifier"][n], value) for n, value in saved["classifier"].items())
                assert all(torch.equal(saved["clients"][key]["decoder"][n], value) for n, value in saved["decoder"].items())
        assert_finite(saved)
        checks[name] = {"round": expected_round, "seed": 42, "all_10_encoders_loaded": True,
                        "batchnorm_buffers": len(bn_keys), "all_10_optimizers_loaded": True,
                        "training_probe_shapes_finite": True, "bytes": (output / name).stat().st_size,
                        "sha256": file_hash(output / name)}
    rows = result["client_test_metrics"]
    assert len(rows) == 10 and sorted(row["client_id"] for row in rows) == list(range(10))
    for row in rows:
        assert row["samples"] == 7180
        assert abs(row["accuracy"] - row["correct"] / row["samples"]) < 1e-12
    for field in ("accuracy", "macro_f1", "balanced_accuracy"):
        assert abs(result["test_metrics"][field] - np.mean([row[field] for row in rows])) < 1e-12
    write_json(output / "verification.json", {"status": "passed", "checkpoints": checks,
                                             "test_accuracies_reconstructed_from_counts": True,
                                             "partition_exhaustive_disjoint_two_classes_per_client": True})
    print(json.dumps(checks, indent=2))


def archive():
    output = PRIVATE / "seed-42"
    verification = json.loads((output / "verification.json").read_text())
    assert verification["status"] == "passed"
    destination = PUBLIC / "artifacts"
    destination.mkdir(exist_ok=False)
    import shutil
    for name in ("results.json", "identity.json", "runtime.json", "verification.json", "timings.jsonl"):
        shutil.copyfile(output / name, destination / name)
    shutil.copyfile(PRIVATE / "prepared/manifest.json", destination / "data_manifest.json")
    with (PRIVATE / "prepared/partitions.json").open("rb") as source, gzip.GzipFile(filename="", fileobj=(destination / "partitions.json.gz").open("wb"), mode="wb", mtime=0) as target:
        shutil.copyfileobj(source, target)
    log = PRIVATE / "logs/run.log"
    shutil.copyfile(log, destination / "run.log")
    write_json(destination / "manifest.json", {"private_checkpoint_directory": str(output),
                                              "checkpoint_sha256": verification["checkpoints"],
                                              "files_sha256": {p.name: file_hash(p) for p in sorted(destination.iterdir())}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify", "archive"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((PUBLIC / "config.json").read_text())
    if args.command == "prepare":
        prepare(config)
    elif args.command == "run":
        run(config, args.resume)
    elif args.command == "verify":
        verify()
    elif args.command == "archive":
        archive()


if __name__ == "__main__":
    main()
