"""Synthetic verification; these tests never read the real FEMNIST test set."""

import copy
import io
import json
from pathlib import Path
import random
import shutil
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from femnist_reconstructed_data import (
    FrozenPartition, canonical_hash, download_archive, file_hash, plan_partition,
    prepare_partition, proportional_quotas,
)
from fusedspacefed_core import UNetSmallAE
from train_femnist_reconstructed import (
    BenchmarkClient, BenchmarkRunner, ReferenceMLP, accuracy_metrics,
    aggregate_shared, evaluation_round, final_statistics, summarize_runs, validate_definitive_config,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def synthetic_partition(tmp_path):
    config = json.loads(Path("configs/femnist_reconstructed.json").read_text())
    config.pop("partition_sha256")
    config.pop("source_archive_sha256")
    config["data"].update(num_clients=4, pool_per_class=40, lognormal_mean=2.0,
                          lognormal_sigma=0.1, count_offset=30.0)
    config["training"].update(rounds=3, clients_per_round=2, batch_size=16,
                              warmup_epochs=1, classification_epochs=1, dz=4, torch_threads=1)
    config["evaluation"].update(first_round=2, last_round=3)
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for label, character in enumerate("abcdefghij"):
            hexadecimal = f"{ord(character):02x}"
            for index in range(20 if label in (2, 3) else 40):
                pixels = np.full((32, 32), (label * 23 + index) % 256, dtype=np.uint8)
                content = io.BytesIO()
                Image.fromarray(pixels).save(content, format="PNG")
                output.writestr(f"by_class/{hexadecimal}/train_{hexadecimal}/{index:04d}.png", content.getvalue())
    config["data"]["source_bytes"] = archive.stat().st_size
    directory = tmp_path / "partition"
    prepare_partition(archive, directory, config)
    return config, FrozenPartition(directory), archive


def assert_states_equal(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def test_integer_apportionment_ties_and_absent_class():
    requests = {"f_00002": 2, "f_00000": 2, "f_00001": 2, "absent": 0}
    result = proportional_quotas(requests, 4)
    assert result == {"f_00002": 1, "f_00000": 2, "f_00001": 1, "absent": 0}
    assert result == proportional_quotas(dict(reversed(list(requests.items()))), 4)
    assert sum(result.values()) == 4


def test_large_integer_apportionment_without_float_rounding():
    huge = 2**60
    assert proportional_quotas({"z": huge + 1, "a": huge}, huge + 1) == {"z": huge // 2 + 1, "a": huge // 2}


def test_sufficient_pool_preserves_original_requests():
    original = {"f_00000": 3, "f_00001": 7, "absent": 0}
    assert proportional_quotas(original, 12) == original
    assert proportional_quotas(original, 10) == original
    assert proportional_quotas(original, 0) == dict.fromkeys(original, 0)
    with pytest.raises(ValueError, match="integers"):
        proportional_quotas(original, 1.5)


def test_reproducible_disjoint_partition_and_quotas(synthetic_partition):
    config, partition, archive = synthetic_partition
    with zipfile.ZipFile(archive) as source:
        pools = {label: [name for name in source.namelist()
                         if name.startswith(f"by_class/{ord(c):02x}/train_{ord(c):02x}/")]
                 for label, c in enumerate("abcdefghij")}
    clients, info = plan_partition(pools, config["data"])
    reversed_pools = {c: list(reversed(pool)) for c, pool in reversed(list(pools.items()))}
    assert (clients, info) == plan_partition(reversed_pools, config["data"])
    all_ids = []
    for client in clients:
        all_ids.extend(row["origin_id"] for split in ("train", "test") for row in client[split])
        assert client["train"] and client["test"]
        assert len(client["train"]) == int(0.9 * client["assigned_total"])
        assert {row["label"] for split in ("train", "test") for row in client[split]} == set(client["classes"])
        for label in range(10):
            if not info[str(label)]["reduced"]:
                assert client["assigned_per_class"][label] == client["requested_per_class"][label]
            assert 0 <= client["assigned_per_class"][label] <= client["requested_per_class"][label]
    assert len(set(all_ids)) == len(all_ids)
    for label in range(10):
        assert sum(c["assigned_per_class"][label] for c in clients) == min(info[str(label)]["requested"], info[str(label)]["selected"])
    assert info["2"]["used"] == info["3"]["used"] == 20
    second = prepare_partition(archive, partition.directory.parent / "rebuild", config)
    assert second["partition_sha256"] == partition.sha256
    assert prepare_partition(archive, partition.directory, config)["partition_sha256"] == partition.sha256


def test_empty_pool_stops_instead_of_removing_client_class(synthetic_partition):
    config, _, _ = synthetic_partition
    pools = {c: [f"{c}/{i}" for i in range(40)] for c in range(10)}
    pools[0] = []
    with pytest.raises(ValueError, match="removes an assigned class"):
        plan_partition(pools, config["data"])
    pools[0] = pools[1].copy()
    with pytest.raises(ValueError, match="Duplicate origin IDs"):
        plan_partition(pools, config["data"])


def test_manifest_detects_global_overlap(synthetic_partition):
    _, partition, _ = synthetic_partition
    path = partition.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["clients"][1]["test"][0]["origin_id"] = manifest["clients"][0]["train"][0]["origin_id"]
    manifest["partition_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "partition_sha256"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        FrozenPartition(partition.directory)


def test_arrays_detect_corruption(synthetic_partition):
    _, partition, _ = synthetic_partition
    path = partition.directory / "train_images.npy"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        FrozenPartition(partition.directory)


def test_definitive_cli_rejects_other_partition(synthetic_partition):
    config = json.loads(Path("configs/femnist_reconstructed.json").read_text())
    _, partition, _ = synthetic_partition
    partition.manifest["recipe"] = config["data"]
    with pytest.raises(ValueError, match="hash does not match"):
        validate_definitive_config(config, partition, 41)


def test_cached_source_is_reused_without_network(synthetic_partition, tmp_path, monkeypatch):
    config, _, archive = synthetic_partition
    cache = tmp_path / "source"
    cache.mkdir()
    shutil.copyfile(archive, cache / "by_class.zip")
    (cache / "by_class.download.json").write_text(json.dumps({"url": config["data"]["source_url"],
                                                            "sha256": file_hash(archive)}))
    def forbidden_network(*args, **kwargs):
        raise AssertionError("Cached source must not be downloaded again")
    monkeypatch.setattr("urllib.request.urlopen", forbidden_network)
    assert download_archive(config["data"], cache) == cache / "by_class.zip"


def test_input_reconstruction_latent_and_logits_shapes(synthetic_partition):
    _, partition, _ = synthetic_partition
    inputs, targets = next(iter(DataLoader(partition.dataset("f_00000", "train"), batch_size=2)))
    assert inputs.shape == (2, 1, 28, 28) and inputs.dtype == torch.float32
    assert targets.dtype == torch.long and 0 <= inputs.min() <= inputs.max() <= 1
    autoencoder = UNetSmallAE(1, 64)
    reconstruction, latent = autoencoder(inputs)
    assert reconstruction.shape == inputs.shape
    assert latent.shape == (2, 64, 7, 7)
    classifier = ReferenceMLP()
    assert sum(p.numel() for p in classifier.parameters()) == 550346
    assert classifier(inputs + reconstruction).shape == (2, 10)
    with pytest.raises(ValueError, match="28"):
        classifier(torch.rand(2, 1, 32, 32))
    for parameter in classifier.parameters():
        parameter.data.zero_()
    classifier.layers[-1].bias.data[0] = -2
    classifier.layers[-1].bias.data[1] = 2
    assert classifier(inputs)[0, :2].tolist() == [-2, 2]  # Raw logits, not probabilities.


def make_test_client():
    torch.manual_seed(43)
    config = json.loads(Path("configs/femnist_reconstructed.json").read_text())
    loader = DataLoader(TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])), batch_size=2)
    return BenchmarkClient("test", loader, config["training"], torch.device("cpu"))


def test_warmup_updates_only_encoder_and_keeps_decoder_gradient_path():
    client = make_test_client()
    encoder, decoder, classifier = client.encoder_state(), client.decoder_state(), client.classifier_state()
    client._warmup(1)
    assert any(not torch.equal(value, encoder[name]) for name, value in client.encoder_state().items())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in client.autoencoder.encoder_parameters())
    assert_states_equal(decoder, client.decoder_state())
    assert_states_equal(classifier, client.classifier_state())
    assert all(p.grad is None for p in client.autoencoder.decoder_parameters())
    assert all(p.grad is None for p in client.classifier.parameters())
    assert client.scaler is None and client.use_amp is False


def test_classification_updates_encoder_decoder_and_classifier():
    client = make_test_client()
    client._warmup(1)
    before = [client.encoder_state(), client.decoder_state(), client.classifier_state()]
    client._joint_train(1)
    for old, new in zip(before, [client.encoder_state(), client.decoder_state(), client.classifier_state()]):
        assert any(not torch.equal(value, old[name]) for name, value in new.items())
    for parameters in (client.autoencoder.encoder_parameters(), client.autoencoder.decoder_parameters(), client.classifier.parameters()):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)


def test_phase_steps_and_optimizer_continuity():
    client = make_test_client()
    metrics = client.train_round(1, 1)
    assert metrics["warmup_steps"] == metrics["classification_steps"] == 2
    assert metrics["encoder_steps"] == 4 and metrics["classifier_steps"] == metrics["decoder_steps"] == 2
    for parameter in client.autoencoder.encoder_parameters():
        assert int(client.ae_optimizer.state[parameter]["step"]) == 4
    for parameter in client.autoencoder.decoder_parameters():
        assert int(client.ae_optimizer.state[parameter]["step"]) == 2
    assert metrics["warmup_samples"] == metrics["classification_samples"] == 4


def test_uniform_aggregation_only_shared_components():
    classifier, decoder = aggregate_shared([{"w": torch.tensor([1.0])}, {"w": torch.tensor([5.0])}],
                                           [{"final.weight": torch.tensor([2.0])}, {"final.weight": torch.tensor([6.0])}])
    assert classifier["w"].item() == 3 and decoder["final.weight"].item() == 4
    with pytest.raises(ValueError, match="Only decoder"):
        aggregate_shared([{"w": torch.tensor([1.0])}], [{"enc1.weight": torch.tensor([2.0])}])


def test_persistent_encoders_current_shared_states_and_optimizer_reset(synthetic_partition, tmp_path):
    config, partition, _ = synthetic_partition
    with BenchmarkRunner(config, partition, tmp_path / "persist", 41, "cpu", "smoke", rounds=2) as runner:
        runner.run(stop_after=1)
        active = runner.state["history"][0]["active_clients"][0]
        inactive = next(c for c in partition.clients if c not in runner.state["encoders"])
        client = runner.make_client(active, 2)
        assert_states_equal(client.encoder_state(), runner.state["encoders"][active])
        assert_states_equal(client.decoder_state(), runner.state["global_decoder"])
        assert_states_equal(client.classifier_state(), runner.state["global_classifier"])
        assert not client.ae_optimizer.state and not client.classifier_optimizer.state
        untrained = runner.make_client(inactive, 2)
        assert_states_equal(untrained.encoder_state(), runner.state["initial_encoder"])
        assert sum(runner.state["participations"].values()) == 2


def test_weighted_and_uniform_client_accuracies():
    metrics = accuracy_metrics({"a": {"correct": 1, "total": 1}, "b": {"correct": 0, "total": 9}})
    assert metrics["sample_weighted_accuracy_percent"] == 10
    assert metrics["uniform_client_accuracy_percent"] == 50
    assert metrics["correct"] == 1 and metrics["total"] == 10
    with pytest.raises(ValueError):
        accuracy_metrics({"empty": {"correct": 0, "total": 0}})


def test_exact_reporting_window_and_no_best_checkpoint():
    config = json.loads(Path("configs/femnist_reconstructed.json").read_text())
    assert [r for r in range(0, 202) if evaluation_round(r, config)] == list(range(191, 201))
    history = [{"round": r, "evaluation": {"sample_weighted_accuracy_percent": float(r - 190),
                                            "uniform_client_accuracy_percent": float(r - 190)}} for r in range(191, 201)]
    assert final_statistics(history, config)["sample_weighted_accuracy_percent"] == 5.5
    assert final_statistics(history[:-1], config) is None


def test_checkpoint_resume_matches_uninterrupted_training(synthetic_partition, tmp_path):
    config, partition, _ = synthetic_partition
    with BenchmarkRunner(config, partition, tmp_path / "full", 41, "cpu") as full:
        full.run()
        expected = copy.deepcopy(full.state)
    with BenchmarkRunner(config, partition, tmp_path / "split", 41, "cpu") as split:
        split.run(stop_after=1)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    with BenchmarkRunner(config, partition, tmp_path / "split", 41, "cpu", resume=True) as resumed:
        result = resumed.run()
        assert result["status"] == "completed"
        assert [r["active_clients"] for r in resumed.state["history"]] == [r["active_clients"] for r in expected["history"]]
        assert [r["evaluation"] for r in resumed.state["history"]] == [r["evaluation"] for r in expected["history"]]
        assert resumed.state["participations"] == expected["participations"]
        for key in ("initial_encoder", "global_classifier", "global_decoder"):
            assert_states_equal(expected[key], resumed.state[key])
        assert resumed.state["encoders"].keys() == expected["encoders"].keys()
        for client in expected["encoders"]:
            assert_states_equal(expected["encoders"][client], resumed.state["encoders"][client])
        assert torch.equal(expected["rng"]["torch_cpu"], resumed.state["rng"]["torch_cpu"])
        assert expected["rng"]["selection_pcg64"] == resumed.state["rng"]["selection_pcg64"]


def test_evaluation_does_not_change_training_states_or_rng(synthetic_partition, tmp_path):
    config, partition, _ = synthetic_partition
    with BenchmarkRunner(config, partition, tmp_path / "eval", 41, "cpu") as runner:
        before = torch.get_rng_state().clone()
        encoder = copy.deepcopy(runner.state["initial_encoder"])
        shared = copy.deepcopy(runner.state["global_classifier"])
        metrics = runner.evaluate()
        assert metrics["total"] == partition.manifest["statistics"]["test"]["total"]
        assert torch.equal(before, torch.get_rng_state())
        assert_states_equal(encoder, runner.state["initial_encoder"])
        assert_states_equal(shared, runner.state["global_classifier"])
        assert all(count["participations"] == 0 for count in metrics["clients"].values())


def test_smoke_never_loads_test_even_inside_evaluation_window(synthetic_partition, tmp_path, monkeypatch):
    config, partition, _ = synthetic_partition
    config["evaluation"].update(first_round=1, last_round=2)
    original = partition.dataset
    def only_training(client, split):
        assert split == "train", "Smoke accessed test pixels"
        return original(client, split)
    monkeypatch.setattr(partition, "dataset", only_training)
    with BenchmarkRunner(config, partition, tmp_path / "smoke", 41, "cpu", "smoke", rounds=2) as runner:
        result = runner.run()
        assert result["summary"] is None and all(r["evaluation"] is None for r in result["history"])
        rng = torch.get_rng_state().clone()
        profile = runner.profile_training_inference()
        assert profile["split"] == "train" and profile["accuracy_computed"] is False
        assert torch.equal(rng, torch.get_rng_state())
        with pytest.raises(RuntimeError, match="never evaluate"):
            runner.evaluate()
    assert "test" not in partition._arrays


def test_output_lock_no_overwrite_and_resume_identity(synthetic_partition, tmp_path):
    config, partition, _ = synthetic_partition
    output = tmp_path / "owned"
    with BenchmarkRunner(config, partition, output, 41, "cpu", "smoke", rounds=1):
        with pytest.raises(BlockingIOError):
            BenchmarkRunner(config, partition, output, 41, "cpu", "smoke", rounds=1, resume=True)
    with pytest.raises(FileExistsError):
        BenchmarkRunner(config, partition, output, 41, "cpu", "smoke", rounds=1)
    with pytest.raises(ValueError, match="identity mismatch"):
        BenchmarkRunner(config, partition, output, 42, "cpu", "smoke", rounds=1, resume=True)


def test_five_run_reporting_uses_run_means_and_sample_std(tmp_path):
    config = json.loads(Path("configs/femnist_reconstructed.json").read_text())
    directories = []
    for value, seed in enumerate(config["run_seeds"], 1):
        directory = tmp_path / f"synthetic-summary-{seed}"
        directory.mkdir()
        row = {"identity": {"seed": seed, "mode": "definitive", "config_sha256": canonical_hash(config),
                            "partition_sha256": config["partition_sha256"], "code_sha256": {}}, "status": "completed",
               "completed_round": 200, "history": [{"round": r} for r in range(1, 201)],
               "summary": {"sample_weighted_accuracy_percent": value, "uniform_client_accuracy_percent": value}}
        (directory / "results.json").write_text(json.dumps(row))
        directories.append(directory)
    output = tmp_path / "summary.json"
    summary = summarize_runs(config, directories, output)
    assert summary["sample_weighted_accuracy_percent"]["mean"] == 3
    assert summary["sample_weighted_accuracy_percent"]["std"] == pytest.approx(2.5**0.5)
    with pytest.raises(FileExistsError):
        summarize_runs(config, directories, output)
