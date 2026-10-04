"""Synthetic tests for training-only calibration data and split provenance."""

import builtins
import copy
import io
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import femnist_calibration_data as calibration
import femnist_reconstructed_data as reconstructed
from femnist_calibration_data import TrainingValidationPartition, prepare_validation_split
from femnist_reconstructed_data import canonical_hash, file_hash


def _make_partition(directory, train_counts=((10, 12, 14), (11, 13, 15), (12, 14, 16))):
    directory.mkdir()
    clients, arrays = [], {}
    for index, counts in enumerate(train_counts):
        client = {"id": f"f_{index:05d}", "classes": [0, 1, 2],
                  "assigned_per_class": [count + 2 for count in counts]}
        for split in ("train", "test"):
            rows = []
            for label, count in enumerate(counts if split == "train" else (2, 2, 2)):
                for item in range(count):
                    rows.append({"origin_id": f"{client['id']}/{label}/{split}/{item}",
                                 "label": label})
            # The original local order is deliberately different from class order.
            client[split] = rows[1::2] + rows[::2]
            client[f"num_{split}"] = len(rows)
        clients.append(client)
    files, statistics = {}, {}
    for split in ("train", "test"):
        rows = [row for client in clients for row in client[split]]
        images = np.empty((len(rows), 1, 28, 28), dtype=np.uint8)
        labels = np.empty(len(rows), dtype=np.int64)
        for index, row in enumerate(rows):
            row["array_index"] = index
            row["png_sha256"] = canonical_hash(row["origin_id"])
            images[index].fill(index % 256)
            labels[index] = row["label"]
        arrays[f"{split}_images.npy"] = images
        arrays[f"{split}_labels.npy"] = labels
        for name in (f"{split}_images.npy", f"{split}_labels.npy"):
            np.save(directory / name, arrays[name], allow_pickle=False)
            files[name] = {"sha256": file_hash(directory / name),
                           "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        sizes = [len(client[split]) for client in clients]
        statistics[split] = {"total": len(rows), "mean": float(np.mean(sizes)),
                             "min": min(sizes), "max": max(sizes),
                             "per_class": [sum(row["label"] == label for row in rows)
                                           for label in range(3)]}
    manifest = {"schema": 1, "benchmark": "synthetic_calibration",
                "recipe": {"num_clients": len(clients), "classes": "abc", "classes_per_client": 3},
                "source": {"sha256": "synthetic-source"}, "clients": clients,
                "files": files, "statistics": statistics}
    manifest["partition_sha256"] = canonical_hash(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


@pytest.fixture
def synthetic_training_partition(tmp_path):
    return _make_partition(tmp_path / "partition")


def _forbid_test_array_access(monkeypatch):
    forbidden = {"test_images.npy", "test_labels.npy"}
    seen = []

    def guard_path(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(value)).name
            assert name not in forbidden, f"Calibration accessed {name}"
            if name.endswith(".npy"):
                seen.append(name)

    def guarded_call(original):
        def wrapped(path, *args, **kwargs):
            guard_path(path)
            return original(path, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(builtins, "open", guarded_call(builtins.open))
    monkeypatch.setattr(io, "open", guarded_call(io.open))
    monkeypatch.setattr(os, "open", guarded_call(os.open))
    monkeypatch.setattr(np, "load", guarded_call(np.load))
    monkeypatch.setattr(calibration, "file_hash", guarded_call(calibration.file_hash))
    monkeypatch.setattr(reconstructed, "file_hash", guarded_call(reconstructed.file_hash))
    return seen


def test_deterministic_stratified_split_preserves_original_partition(synthetic_training_partition, tmp_path):
    directory, original = synthetic_training_partition
    original_bytes = (directory / "manifest.json").read_bytes()
    original_files = {name: (directory / name).read_bytes() for name in original["files"]}
    first = prepare_validation_split(directory, tmp_path / "first.json")
    second = prepare_validation_split(directory, tmp_path / "second.json")
    assert first == second
    assert first["parent_partition_sha256"] == original["partition_sha256"]
    assert first["source_train_files"] == {name: original["files"][name] for name in calibration.TRAIN_FILES}
    assert canonical_hash({key: value for key, value in first.items() if key != "view_sha256"}) == first["view_sha256"]
    all_fit, all_validation, all_original = set(), set(), set()
    for derived, frozen in zip(first["clients"], original["clients"]):
        assert derived["id"] == frozen["id"]
        fit_ids = {row["origin_id"] for row in derived["train"]}
        validation_ids = {row["origin_id"] for row in derived["validation"]}
        train_ids = {row["origin_id"] for row in frozen["train"]}
        assert not fit_ids & validation_ids and fit_ids | validation_ids == train_ids
        assert not (fit_ids | validation_ids) & {row["origin_id"] for row in frozen["test"]}
        for split, selected_ids in (("train", fit_ids), ("validation", validation_ids)):
            assert [row["origin_id"] for row in derived[split]] == [
                row["origin_id"] for row in frozen["train"] if row["origin_id"] in selected_ids]
            assert {row["label"] for row in derived[split]} == set(frozen["classes"])
            assert derived[f"num_{split}"] == len(derived[split])
        for label in frozen["classes"]:
            original_count = sum(row["label"] == label for row in frozen["train"])
            assert derived["validation_per_class"][label] == original_count // 5
            assert derived["train_per_class"][label] == original_count - original_count // 5
        all_fit.update(fit_ids)
        all_validation.update(validation_ids)
        all_original.update(train_ids)
    assert not all_fit & all_validation and all_fit | all_validation == all_original
    assert first["statistics"]["train"]["total"] == len(all_fit)
    assert first["statistics"]["validation"]["total"] == len(all_validation)
    assert (directory / "manifest.json").read_bytes() == original_bytes
    assert {name: (directory / name).read_bytes() for name in original["files"]} == original_files


def test_constructor_and_all_dataset_access_never_open_test_files(synthetic_training_partition, tmp_path, monkeypatch):
    directory, original = synthetic_training_partition
    # Calibration must work even if test arrays are physically unavailable.
    (directory / "test_images.npy").unlink()
    (directory / "test_labels.npy").unlink()
    seen = _forbid_test_array_access(monkeypatch)
    split_path = tmp_path / "validation.json"
    split = prepare_validation_split(directory, split_path)
    view = TrainingValidationPartition(directory, split_path)
    assert view.manifest == original
    assert view.clients == {client["id"]: client for client in original["clients"]}
    assert view.sha256 == original["partition_sha256"]
    assert view.view_sha256 == split["view_sha256"]
    assert view.validation_manifest == split and view.access_policy == "training_arrays_only"
    for client_id in view.clients:
        for name in ("train", "validation"):
            dataset = view.dataset(client_id, name)
            rows = view._views[client_id][name]
            observed = []
            for inputs, targets in DataLoader(dataset, batch_size=7, shuffle=False,
                                               generator=torch.Generator().manual_seed(5)):
                assert inputs.dtype == torch.float32 and inputs.shape[1:] == (1, 28, 28)
                assert targets.dtype == torch.int64
                observed.extend(targets.tolist())
            assert observed == [row["label"] for row in rows]
            image, target = dataset[0]
            assert image[0, 0, 0].item() == pytest.approx((rows[0]["array_index"] % 256) / 255)
            assert int(target) == rows[0]["label"]
        with pytest.raises(ValueError, match="never access"):
            view.dataset(client_id, "test")
    with pytest.raises(ValueError, match="Unknown"):
        view.dataset(next(iter(view.clients)), "other")
    assert set(seen) == set(calibration.TRAIN_FILES)


def test_split_idempotency_and_destination_mismatch(synthetic_training_partition, tmp_path):
    directory, _ = synthetic_training_partition
    split_path = tmp_path / "validation.json"
    expected = prepare_validation_split(directory, split_path)
    before = split_path.read_bytes()
    assert prepare_validation_split(directory, split_path) == expected
    assert split_path.read_bytes() == before
    for kwargs in ({"seed": 20261005}, {"numerator": 1, "denominator": 4}):
        with pytest.raises(FileExistsError, match="different"):
            prepare_validation_split(directory, split_path, **kwargs)
        assert split_path.read_bytes() == before
    other = prepare_validation_split(directory, tmp_path / "different.json", seed=20261005)
    assert other["view_sha256"] != expected["view_sha256"]
    assert [client["validation"] for client in other["clients"]] != [client["validation"] for client in expected["clients"]]
    with pytest.raises(FileExistsError, match="occupied"):
        prepare_validation_split(directory, directory)


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": "20261004"}, {"numerator": 1.0}, {"denominator": True},
    {"numerator": 0}, {"numerator": -1}, {"denominator": 0}, {"numerator": 5},
])
def test_invalid_split_parameters_fail_before_creating_output(synthetic_training_partition, tmp_path, kwargs):
    directory, _ = synthetic_training_partition
    destination = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        prepare_validation_split(directory, destination, **kwargs)
    assert not destination.exists()


def test_insufficient_class_count_has_no_fallback(tmp_path):
    directory, _ = _make_partition(tmp_path / "small", train_counts=((4, 10, 10),))
    destination = tmp_path / "small.json"
    with pytest.raises(ValueError, match="every class nonempty"):
        prepare_validation_split(directory, destination)
    assert not destination.exists()


@pytest.mark.parametrize("rehash", [False, True])
def test_tampered_split_rows_are_rejected_even_with_updated_hash(synthetic_training_partition, tmp_path, rehash):
    directory, _ = synthetic_training_partition
    split_path = tmp_path / "validation.json"
    manifest = prepare_validation_split(directory, split_path)
    manifest["clients"][0]["validation"][0] = copy.deepcopy(manifest["clients"][0]["train"][0])
    if rehash:
        manifest["view_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "view_sha256"})
    split_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="deterministic split" if rehash else "hash mismatch"):
        TrainingValidationPartition(directory, split_path)
    with pytest.raises(ValueError):
        prepare_validation_split(directory, split_path)


@pytest.mark.parametrize("field", ["parent", "source", "statistics", "rule"])
def test_rehashed_provenance_or_statistics_tampering_is_rejected(synthetic_training_partition, tmp_path, field):
    directory, _ = synthetic_training_partition
    split_path = tmp_path / "validation.json"
    manifest = prepare_validation_split(directory, split_path)
    if field == "parent":
        manifest["parent_partition_sha256"] = "different-parent"
    elif field == "source":
        manifest["source_train_files"]["train_images.npy"]["sha256"] = "different-array"
    elif field == "statistics":
        manifest["statistics"]["validation"]["total"] += 1
    else:
        manifest["split"]["rule"] = "other-rule"
    manifest["view_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "view_sha256"})
    split_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="deterministic split"):
        TrainingValidationPartition(directory, split_path)


def test_changed_training_array_is_rejected_on_every_instantiation(synthetic_training_partition, tmp_path):
    directory, _ = synthetic_training_partition
    split_path = tmp_path / "validation.json"
    prepare_validation_split(directory, split_path)
    path = directory / "train_images.npy"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        TrainingValidationPartition(directory, split_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_validation_split(directory, split_path)


@pytest.mark.parametrize("change", ["labels", "shape", "dtype"])
def test_training_array_contents_and_structure_are_validated(synthetic_training_partition, tmp_path, change):
    directory, manifest = synthetic_training_partition
    if change == "labels":
        filename = "train_labels.npy"
        array = np.load(directory / filename, allow_pickle=False)
        array[0] = (array[0] + 1) % 3
    else:
        filename = "train_images.npy"
        array = np.load(directory / filename, allow_pickle=False)
        array = array[:, :, :-1, :] if change == "shape" else array.astype(np.float32)
    np.save(directory / filename, array, allow_pickle=False)
    manifest["files"][filename] = {"sha256": file_hash(directory / filename),
                                   "shape": list(array.shape), "dtype": str(array.dtype)}
    manifest["partition_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "partition_sha256"})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="labels differ" if change == "labels" else "dimensions or dtype"):
        prepare_validation_split(directory, tmp_path / "validation.json")


def test_validation_view_refuses_a_different_frozen_partition(synthetic_training_partition, tmp_path):
    directory, _ = synthetic_training_partition
    split_path = tmp_path / "validation.json"
    prepare_validation_split(directory, split_path)
    other, _ = _make_partition(tmp_path / "other", train_counts=((10, 12, 15), (11, 13, 15), (12, 14, 16)))
    with pytest.raises(ValueError, match="frozen parent"):
        TrainingValidationPartition(other, split_path)


def test_split_and_dataset_access_preserve_all_training_rng(synthetic_training_partition, tmp_path):
    directory, _ = synthetic_training_partition
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    split_path = tmp_path / "validation.json"
    first = prepare_validation_split(directory, split_path)
    view = TrainingValidationPartition(directory, split_path)
    for client_id in view.clients:
        view.dataset(client_id, "train")[0]
        view.dataset(client_id, "validation")[0]
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert torch.equal(torch_before, torch.get_rng_state())
    # Changing all model/training RNGs cannot change the split selection.
    random.seed(5)
    np.random.seed(6)
    torch.set_rng_state(torch.Generator().manual_seed(7).get_state())
    try:
        assert prepare_validation_split(directory, tmp_path / "other-rng.json") == first
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(torch_before)
