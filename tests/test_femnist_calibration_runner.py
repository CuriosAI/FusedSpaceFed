"""Synthetic CPU verification of calibration isolation, resume, and selection."""

import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import calibrate_femnist_reconstructed as calibration
import train_femnist_reconstructed as benchmark
from femnist_calibration_data import TrainingValidationPartition, prepare_validation_split
from femnist_reconstructed_data import FrozenPartition, canonical_hash
from test_femnist_calibration_data import _forbid_test_array_access, _make_partition


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def synthetic_calibration(tmp_path, monkeypatch):
    directory, manifest = _make_partition(tmp_path / "partition")
    reference = json.loads(calibration.REFERENCE.read_text())
    reference["benchmark"] = manifest["benchmark"]
    reference["data"] = copy.deepcopy(manifest["recipe"])
    reference["partition_sha256"] = manifest["partition_sha256"]
    reference["source_archive_sha256"] = manifest["source"]["sha256"]
    reference["training"].update(rounds=3, clients_per_round=2, batch_size=16, dz=4,
                                  warmup_epochs=1, classification_epochs=1, torch_threads=1)
    reference["evaluation"].update(first_round=2, last_round=3)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setattr(calibration, "REFERENCE", reference_path)
    metadata = {"commit": "synthetic-code", "working_tree_clean": True,
                "code_sha256": {name: "synthetic-hash-" + name for name in (
                    "fusedspacefed_core.py", "femnist_reconstructed_data.py",
                    "train_femnist_reconstructed.py", "femnist_calibration_data.py",
                    "calibrate_femnist_reconstructed.py")}}
    # Exercise actual runner/checkpoint paths without invoking Git or GPU tools.
    monkeypatch.setattr(benchmark, "git_metadata", lambda: copy.deepcopy(metadata))
    monkeypatch.setattr(calibration, "git_metadata", lambda: copy.deepcopy(metadata))
    config = copy.deepcopy(reference)
    config["calibration"] = {
        "plan_sha256": "synthetic-plan", "candidate_id": "00-reference",
        "screening": {"rounds": 1, "first_round": 1, "last_round": 1},
        "confirmation": {"rounds": 3, "first_round": 2, "last_round": 3},
    }
    return {"directory": directory, "manifest": manifest, "reference": reference,
            "config": config, "metadata": metadata}


def _view(context, tmp_path, name="validation.json", seed=20261004):
    path = tmp_path / name
    prepare_validation_split(context["directory"], path, seed=seed)
    return TrainingValidationPartition(context["directory"], path)


def _assert_tensor_states_equal(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def _assert_model_states_equal(left, right):
    for name in ("initial_encoder", "global_classifier", "global_decoder"):
        _assert_tensor_states_equal(left[name], right[name])
    assert left["encoders"].keys() == right["encoders"].keys()
    for client_id in left["encoders"]:
        _assert_tensor_states_equal(left["encoders"][client_id], right["encoders"][client_id])


def _assert_rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy_legacy"][0] == right["numpy_legacy"][0]
    np.testing.assert_array_equal(left["numpy_legacy"][1], right["numpy_legacy"][1])
    assert left["numpy_legacy"][2:] == right["numpy_legacy"][2:]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    assert left["selection_pcg64"] == right["selection_pcg64"]
    assert left["torch_cuda"] is None and right["torch_cuda"] is None


def test_calibration_never_accesses_test_arrays_through_training_and_evaluation(synthetic_calibration, tmp_path, monkeypatch):
    context = synthetic_calibration
    (context["directory"] / "test_images.npy").unlink()
    (context["directory"] / "test_labels.npy").unlink()
    seen = _forbid_test_array_access(monkeypatch)
    view = _view(context, tmp_path)
    with calibration.CalibrationRunner(context["config"], view, tmp_path / "calibration", 141, "cpu") as runner:
        result = runner.run()
        assert result["status"] == "completed"
        assert result["identity"]["mode"] == "calibration"
        assert result["identity"]["evaluation_split"] == "validation"
        assert result["identity"]["data_access_policy"] == "training_arrays_only"
        assert result["identity"]["data_view_sha256"] == view.view_sha256
        assert result["identity"]["partition_sha256"] == view.sha256
        assert result["validation_definition"]["view_sha256"] == view.view_sha256
        counts = {client["id"]: client["num_train"] for client in view.validation_manifest["clients"]}
        total = view.validation_manifest["statistics"]["validation"]["total"]
        for row in result["history"]:
            assert row["evaluation"]["split"] == "validation"
            assert row["evaluation"]["total"] == total
            for local in row["clients"]:
                assert local["training_samples"] == counts[local["client"]]
                assert local["training_samples"] < len(view.clients[local["client"]]["train"])
        assert calibration.validation_score(result, 2, 3) == result["summary"]
        with pytest.raises(ValueError, match="never access"):
            view.dataset(next(iter(view.clients)), "test")
    assert set(seen) == {"train_images.npy", "train_labels.npy"}


def test_validation_preserves_models_participations_and_training_rng(synthetic_calibration, tmp_path):
    context = synthetic_calibration
    view = _view(context, tmp_path)
    with calibration.CalibrationRunner(context["config"], view, tmp_path / "evaluation", 141, "cpu") as runner:
        runner.run(stop_after=1)
        state_before = copy.deepcopy(runner.state)
        rng_before = benchmark.rng_snapshot(runner.device, runner.selection)
        metrics = runner.evaluate()
        _assert_rng_equal(rng_before, benchmark.rng_snapshot(runner.device, runner.selection))
        _assert_model_states_equal(state_before, runner.state)
        assert runner.state["participations"] == state_before["participations"]
        assert runner.state["history"] == state_before["history"]
        assert runner.state["completed_round"] == 1
        assert metrics["total"] == view.validation_manifest["statistics"]["validation"]["total"]
        assert all(c["participations"] == runner.state["participations"][client]
                   for client, c in metrics["clients"].items())


def test_calibration_and_definitive_outputs_keep_distinct_metric_identity(synthetic_calibration, tmp_path):
    context = synthetic_calibration
    view = _view(context, tmp_path)
    with calibration.CalibrationRunner(context["config"], view, tmp_path / "calibration", 141, "cpu") as runner:
        calibrated = runner.run()
    frozen = FrozenPartition(context["directory"])
    with benchmark.BenchmarkRunner(context["reference"], frozen, tmp_path / "definitive", 41, "cpu") as runner:
        definitive = runner.run()
    for result, mode, split, total in (
        (calibrated, "calibration", "validation", view.validation_manifest["statistics"]["validation"]["total"]),
        (definitive, "definitive", "test", frozen.manifest["statistics"]["test"]["total"]),
    ):
        assert result["identity"]["mode"] == mode
        rows = result["history"][1:]
        assert all(row["evaluation"]["split"] == split and row["evaluation"]["total"] == total for row in rows)
        for metric in (calibration.PRIMARY, calibration.SECONDARY):
            assert result["summary"][metric] == pytest.approx(np.mean([row["evaluation"][metric] for row in rows]))
    assert definitive["history"][0]["evaluation"] is None
    assert "validation_definition" not in definitive
    assert "data_view_sha256" not in definitive["identity"]
    directories = []
    for seed in context["reference"]["run_seeds"]:
        directory = tmp_path / f"calibration-summary-{seed}"
        directory.mkdir()
        record = copy.deepcopy(calibrated)
        record["identity"]["seed"] = seed
        (directory / "results.json").write_text(json.dumps(record))
        directories.append(directory)
    with pytest.raises(ValueError, match="completed definitive"):
        benchmark.summarize_runs(context["reference"], directories, tmp_path / "summary.json")


def test_calibration_resume_matches_uninterrupted_training_and_preserves_private_encoders(synthetic_calibration, tmp_path):
    context = synthetic_calibration
    view = _view(context, tmp_path)
    with calibration.CalibrationRunner(context["config"], view, tmp_path / "full", 141, "cpu") as full:
        expected_result = full.run()
        expected = copy.deepcopy(full.state)
    with calibration.CalibrationRunner(context["config"], view, tmp_path / "resumed", 141, "cpu") as split:
        first = split.run(stop_after=1)
        assert first["status"] == "paused" and first["summary"] is None
        active = first["history"][0]["active_clients"][0]
        inactive = next(client for client in view.clients if client not in split.state["encoders"])
        persistent = split.make_client(active, 2)
        _assert_tensor_states_equal(persistent.encoder_state(), split.state["encoders"][active])
        _assert_tensor_states_equal(persistent.decoder_state(), split.state["global_decoder"])
        _assert_tensor_states_equal(persistent.classifier_state(), split.state["global_classifier"])
        assert not persistent.ae_optimizer.state and not persistent.classifier_optimizer.state
        assert persistent.num_samples == len(view.dataset(active, "train"))
        initial = split.make_client(inactive, 2)
        _assert_tensor_states_equal(initial.encoder_state(), split.state["initial_encoder"])
        assert sum(split.state["participations"].values()) == 2
        del persistent, initial
    random.seed(991)
    np.random.seed(992)
    torch.set_rng_state(torch.Generator().manual_seed(993).get_state())
    with calibration.CalibrationRunner(context["config"], view, tmp_path / "resumed", 141, "cpu", resume=True) as resumed:
        actual_result = resumed.run()
        assert actual_result["status"] == "completed"
        assert actual_result["summary"] == expected_result["summary"]
        assert actual_result["identity"] == expected_result["identity"]
        assert actual_result["validation_definition"] == expected_result["validation_definition"]
        assert resumed.state["participations"] == expected["participations"]
        assert [row["active_clients"] for row in resumed.state["history"]] == [row["active_clients"] for row in expected["history"]]
        assert [row["evaluation"] for row in resumed.state["history"]] == [row["evaluation"] for row in expected["history"]]
        _assert_model_states_equal(expected, resumed.state)
        _assert_rng_equal(expected["rng"], resumed.state["rng"])


def test_resume_rejects_a_changed_validation_view_or_run_mode(synthetic_calibration, tmp_path):
    context = synthetic_calibration
    original = _view(context, tmp_path)
    output = tmp_path / "identity"
    with calibration.CalibrationRunner(context["config"], original, output, 141, "cpu") as runner:
        runner.run(stop_after=1)
    changed = _view(context, tmp_path, name="changed.json", seed=20261005)
    assert changed.sha256 == original.sha256 and changed.view_sha256 != original.view_sha256
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        calibration.CalibrationRunner(context["config"], changed, output, 141, "cpu", resume=True)
    with pytest.raises(ValueError, match="Resume identity mismatch"):
        benchmark.BenchmarkRunner(context["config"], original, output, 141, "cpu", resume=True)


def test_calibration_requires_metadata_and_a_training_only_partition(synthetic_calibration, tmp_path):
    context = synthetic_calibration
    view = _view(context, tmp_path)
    with pytest.raises(ValueError, match="metadata is required"):
        calibration.CalibrationRunner(context["reference"], view, tmp_path / "missing", 141, "cpu")
    frozen = FrozenPartition(context["directory"], verify_files=False)
    with pytest.raises(ValueError, match="training-only validation"):
        calibration.CalibrationRunner(context["config"], frozen, tmp_path / "test-capable", 141, "cpu")
    assert not (tmp_path / "missing").exists() and not (tmp_path / "test-capable").exists()


def test_nonfinite_validation_logits_fail_instead_of_producing_accuracy(synthetic_calibration, tmp_path, monkeypatch):
    context = synthetic_calibration
    view = _view(context, tmp_path)
    with calibration.CalibrationRunner(context["config"], view, tmp_path / "nonfinite", 141, "cpu") as runner:
        rng_before = benchmark.rng_snapshot(runner.device, runner.selection)
        monkeypatch.setattr(benchmark.ReferenceMLP, "forward", lambda self, inputs: torch.full(
            (len(inputs), 10), float("nan"), dtype=inputs.dtype, device=inputs.device))
        with pytest.raises(RuntimeError, match="Non-finite validation output; client="):
            runner.evaluate()
        _assert_rng_equal(rng_before, benchmark.rng_snapshot(runner.device, runner.selection))


@pytest.mark.parametrize("section,key,value", [
    ("training", "dz", 8), ("training", "classifier_dims", [784, 256, 256, 64, 10]),
    ("training", "warmup_epochs", 0), ("training", "classification_epochs", 2),
    ("training", "batch_size", 8), ("training", "rounds", 4),
    ("data", "classes", "abd"), ("evaluation", "first_round", 1),
    ("evaluation", "adaptation_epochs", 1),
])
def test_protected_architecture_data_and_protocol_settings_are_rejected(synthetic_calibration, section, key, value):
    config = copy.deepcopy(synthetic_calibration["config"])
    config[section][key] = value
    with pytest.raises(ValueError, match="protected experimental protocol"):
        calibration.validate_training_config(config)


def test_three_prespecified_tunables_are_accepted(synthetic_calibration):
    config = copy.deepcopy(synthetic_calibration["config"])
    config["training"].update(classifier_lr=0.005, autoencoder_lr=0.0003, gradient_clip_norm=0.5)
    calibration.validate_training_config(config)


@pytest.mark.parametrize("name,value", [
    ("classifier_lr", 0), ("autoencoder_lr", -1), ("gradient_clip_norm", float("nan")),
    ("classifier_lr", float("inf")),
])
def test_invalid_tunable_values_are_rejected(synthetic_calibration, name, value):
    config = copy.deepcopy(synthetic_calibration["config"])
    config["training"][name] = value
    with pytest.raises(ValueError, match="Invalid tunable setting"):
        calibration.validate_training_config(config)


def _score(primary, secondary):
    return {calibration.PRIMARY: primary, calibration.SECONDARY: secondary}


def test_selection_uses_two_seed_primary_means_and_identifier_ties():
    scores = {
        "00-reference": {141: _score(40, 40), 142: _score(40, 40)},
        "01-best-single-seed": {141: _score(90, 99), 142: _score(10, 99)},
        "02-best-mean": {141: _score(60, 0), 142: _score(60, 0)},
        "03-best-secondary": {141: _score(60, 100), 142: _score(60, 100)},
    }
    selected = calibration.select_configuration(scores, [141, 142])
    assert selected["selected_candidate"] == "02-best-mean"
    assert selected["mean_validation_scores"]["01-best-single-seed"][calibration.PRIMARY] == 50
    assert selected["mean_validation_scores"]["02-best-mean"][calibration.PRIMARY] == 60
    assert selected["selection_metric"] == calibration.PRIMARY
    assert selected["tuning_seeds"] == [141, 142]
    assert selected["tie_break"] == "candidate identifier ascending"
    incomplete = copy.deepcopy(scores)
    incomplete["02-best-mean"].pop(142)
    with pytest.raises(ValueError, match="same complete tuning-seed set"):
        calibration.select_configuration(incomplete, [141, 142])


def test_screening_always_promotes_reference_and_uses_primary_only():
    scores = {"00-reference": _score(1, 1), "01-primary": _score(80, 0),
              "02-secondary": _score(70, 100), "03-tied-primary": _score(80, 100)}
    assert calibration.select_candidates(scores, 1) == ["00-reference", "01-primary"]
    assert calibration.select_candidates(scores, 2) == ["00-reference", "01-primary", "03-tied-primary"]
    scores.pop("00-reference")
    with pytest.raises(RuntimeError, match="Reference failed"):
        calibration.select_candidates(scores, 1)


def _audit_result():
    result = {"identity": {"mode": "calibration"}, "participations": {"a": 0, "b": 1},
              "validation_definition": {"statistics": {"validation": {"total": 10}}}, "history": []}
    for round_number, correct_a, correct_b in ((1, 1, 9), (2, 1, 0), (3, 0, 9), (4, 1, 9)):
        counts = {"a": {"correct": correct_a, "total": 1}, "b": {"correct": correct_b, "total": 9}}
        result["history"].append({"round": round_number, "evaluation": {
            "split": "validation", **benchmark.accuracy_metrics(counts)}})
    return result


def test_validation_score_averages_the_fixed_window_without_best_round_selection():
    result = _audit_result()
    assert calibration.validation_score(result, 2, 3) == _score(50, 50)
    assert calibration.validation_score(result, 1, 1) == _score(100, 100)


@pytest.mark.parametrize("change", [
    "test_mode", "test_split", "missing_metric", "missing_round", "duplicate_round",
    "missing_client", "negative_correct", "too_many_correct", "empty_client", "holdout_total",
    "primary", "secondary",
])
def test_validation_score_audits_client_counts_metrics_and_window(change):
    result = _audit_result()
    metric = result["history"][1]["evaluation"]
    if change == "test_mode":
        result["identity"]["mode"] = "definitive"
    elif change == "test_split":
        metric["split"] = "test"
    elif change == "missing_metric":
        result["history"][1]["evaluation"] = None
    elif change == "missing_round":
        result["history"].pop(2)
    elif change == "duplicate_round":
        result["history"].insert(2, copy.deepcopy(result["history"][1]))
    elif change == "missing_client":
        metric["clients"].pop("b")
    elif change == "negative_correct":
        metric["clients"]["a"]["correct"] = -1
    elif change == "too_many_correct":
        metric["clients"]["a"]["correct"] = 2
    elif change == "empty_client":
        metric["clients"]["b"]["total"] = 0
    elif change == "holdout_total":
        result["validation_definition"]["statistics"]["validation"]["total"] = 11
    elif change == "primary":
        metric[calibration.PRIMARY] += 0.1
    else:
        metric[calibration.SECONDARY] += 0.1
    with pytest.raises(ValueError):
        calibration.validation_score(result, 2, 3)


@pytest.mark.parametrize("name,value", [
    (calibration.PRIMARY, float("nan")), (calibration.SECONDARY, float("nan")),
    (calibration.PRIMARY, float("inf")), (calibration.SECONDARY, float("-inf")),
])
def test_validation_score_rejects_nonfinite_saved_metrics(name, value):
    result = _audit_result()
    result["history"][1]["evaluation"][name] = value
    with pytest.raises(ValueError):
        calibration.validation_score(result, 2, 3)


def _frozen_receipt(context, selected):
    return {"status": "frozen", "selection_data": "training-derived validation only",
            "selected_config_sha256": canonical_hash(selected),
            "calibration_code": copy.deepcopy(context["metadata"])}


def test_selected_config_requires_matching_frozen_receipt(synthetic_calibration, tmp_path, monkeypatch):
    context = synthetic_calibration
    selected_path, receipt_path = tmp_path / "selected.json", tmp_path / "selection.json"
    monkeypatch.setattr(calibration, "SELECTED", selected_path)
    monkeypatch.setattr(calibration, "FROZEN_RECEIPT", receipt_path)
    selected = copy.deepcopy(context["reference"])
    selected["training"]["classifier_lr"] = 0.005
    with pytest.raises(ValueError, match="has not been frozen"):
        calibration.validate_selected_config(selected)
    selected_path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="has not been frozen"):
        calibration.validate_selected_config(selected)
    receipt = _frozen_receipt(context, selected)
    bad_receipt = copy.deepcopy(receipt)
    bad_receipt["selected_config_sha256"] = "mismatch"
    receipt_path.write_text(json.dumps(bad_receipt))
    with pytest.raises(ValueError, match="frozen selection"):
        calibration.validate_selected_config(selected)
    receipt_path.write_text(json.dumps(receipt))
    calibration.validate_selected_config(selected)
    changed = copy.deepcopy(selected)
    changed["training"]["autoencoder_lr"] *= 0.5
    with pytest.raises(ValueError, match="frozen selection"):
        calibration.validate_selected_config(changed)


@pytest.mark.parametrize("field", ["status", "selection_data", "code_sha256"])
def test_selected_config_rejects_invalid_receipt_or_changed_calibration_code(synthetic_calibration, tmp_path, monkeypatch, field):
    context = synthetic_calibration
    selected_path, receipt_path = tmp_path / "selected.json", tmp_path / "selection.json"
    monkeypatch.setattr(calibration, "SELECTED", selected_path)
    monkeypatch.setattr(calibration, "FROZEN_RECEIPT", receipt_path)
    selected = copy.deepcopy(context["reference"])
    selected["training"]["autoencoder_lr"] = 0.0003
    receipt = _frozen_receipt(context, selected)
    if field == "status":
        receipt["status"] = "calibrating"
    elif field == "selection_data":
        receipt["selection_data"] = "test"
    else:
        receipt["calibration_code"]["code_sha256"]["femnist_calibration_data.py"] = "different-code"
    selected_path.write_text(json.dumps(selected))
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        calibration.validate_selected_config(selected)
