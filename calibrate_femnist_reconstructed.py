"""Training-only hyperparameter calibration, then one immutable configuration.

This entry point reuses the benchmark training phases and round-boundary
checkpointing. Calibration views cannot open the definitive test arrays.
The search has fixed candidates, two independent tuning seeds and fixed
validation windows. It does not run baselines or definitive test evaluation.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from femnist_reconstructed_data import (
    REPO, atomic_json, canonical_hash, file_hash, require_untracked_artifacts,
)
from femnist_calibration_data import prepare_validation_split, TrainingValidationPartition
from train_femnist_reconstructed import BenchmarkRunner, git_metadata

REFERENCE = REPO / "configs/femnist_reconstructed.json"
SELECTED = REPO / "configs/femnist_reconstructed_calibrated.json"
FROZEN_RECEIPT = REPO / "artifacts/femnist_calibration/selection.json"
TUNABLE = {"classifier_lr", "autoencoder_lr", "gradient_clip_norm"}
PRIMARY = "sample_weighted_accuracy_percent"
SECONDARY = "uniform_client_accuracy_percent"


def validate_training_config(config: dict, reference: dict | None = None) -> None:
    reference = json.loads(REFERENCE.read_text()) if reference is None else reference
    stripped = copy.deepcopy(config)
    stripped.pop("calibration", None)
    for name in TUNABLE:
        value = stripped["training"][name]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid tunable setting: {name}")
        stripped["training"][name] = reference["training"][name]
    if stripped != reference:
        raise ValueError("Architecture, data and protected experimental protocol must remain unchanged")


def validate_selected_config(config: dict) -> None:
    validate_training_config(config)
    if not SELECTED.exists() or not FROZEN_RECEIPT.exists():
        raise ValueError("Calibrated configuration has not been frozen")
    selected = json.loads(SELECTED.read_text())
    receipt = json.loads(FROZEN_RECEIPT.read_text())
    if config != selected or canonical_hash(config) != receipt["selected_config_sha256"]:
        raise ValueError("Definitive configuration differs from the validation-only frozen selection")
    if receipt["status"] != "frozen" or receipt["selection_data"] != "training-derived validation only":
        raise ValueError("Invalid selection receipt")
    if receipt["calibration_code"]["code_sha256"] != git_metadata()["code_sha256"]:
        raise ValueError("Training implementation differs from the frozen validation selection")


def validate_plan(plan: dict) -> dict:
    reference = json.loads(REFERENCE.read_text())
    if canonical_hash(reference) != plan["reference_config_sha256"]:
        raise ValueError("Reference configuration changed")
    if plan["partition_sha256"] != reference["partition_sha256"]:
        raise ValueError("Plan requires the original frozen partition")
    if plan["tuning_seeds"] != [141, 142]:
        raise ValueError("Use the prespecified tuning seeds, distinct from final seeds")
    if plan["confirmation"] != {"rounds": 200, "first_round": 191, "last_round": 200}:
        raise ValueError("Full validation confirmation must retain the agreed reporting horizon")
    screen = plan["screening"]
    if not 1 <= screen["first_round"] <= screen["last_round"] == screen["rounds"] < 191:
        raise ValueError("Invalid prespecified screening window")
    candidates = plan["candidates"]
    if len(candidates) != plan["budget"]["max_candidates"] or len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("Invalid fixed candidate set")
    if candidates[0] != {"id": "00-reference", "overrides": {}}:
        raise ValueError("Previous stabilized reference must be included")
    finalists = min(screen["promote_alternatives"], len(candidates) - 1) + 1
    budget = len(candidates) * screen["rounds"] + finalists * (400 - screen["rounds"])
    if budget != plan["budget"]["max_calibration_rounds"] or plan["budget"]["new_definitive_rounds"] != 1000:
        raise ValueError("Stage schedule differs from the declared search/final budget")
    for row in candidates:
        if not set(row["overrides"]) <= TUNABLE:
            raise ValueError("Search contains a protected setting")
        candidate = copy.deepcopy(reference)
        candidate["training"].update(row["overrides"])
        validate_training_config(candidate, reference)
    return reference


def candidate_config(plan: dict, row: dict) -> dict:
    config = copy.deepcopy(validate_plan(plan))
    config["training"].update(row["overrides"])
    config["calibration"] = {"plan_sha256": canonical_hash(plan), "candidate_id": row["id"],
                             "screening": plan["screening"], "confirmation": plan["confirmation"]}
    return config


class CalibrationRunner(BenchmarkRunner):
    def __init__(self, config, partition, output, seed, device, resume=False):
        validate_training_config(config)
        if "calibration" not in config:
            raise ValueError("Calibration metadata is required")
        super().__init__(config, partition, output, seed, device, mode="calibration", resume=resume)

    def should_evaluate(self, round_number: int) -> bool:
        windows = [self.config["calibration"][key] for key in ("screening", "confirmation")]
        return any(window["first_round"] <= round_number <= window["last_round"] for window in windows)


def validation_score(result: dict, first: int, last: int) -> dict:
    if result["identity"]["mode"] != "calibration":
        raise ValueError("Selection accepts calibration results only")
    rows = [r for r in result["history"] if first <= r["round"] <= last]
    if [r["round"] for r in rows] != list(range(first, last + 1)):
        raise ValueError("Incomplete fixed validation scoring window")
    for row in rows:
        metric = row["evaluation"]
        if not metric or metric["split"] != "validation":
            raise ValueError("Test or missing metrics cannot enter calibration selection")
        counts = metric["clients"]
        if set(counts) != set(result["participations"]) or any(
                not 0 <= c["correct"] <= c["total"] or c["total"] <= 0 for c in counts.values()):
            raise ValueError("Invalid or incomplete validation client counts")
        total, correct = sum(c["total"] for c in counts.values()), sum(c["correct"] for c in counts.values())
        if total != result["validation_definition"]["statistics"]["validation"]["total"]:
            raise ValueError("Validation count differs from frozen training-derived holdout")
        primary = 100.0 * correct / total
        secondary = sum(100.0 * c["correct"] / c["total"] for c in counts.values()) / len(counts)
        if (not all(math.isfinite(value) for value in (primary, secondary, metric[PRIMARY], metric[SECONDARY]))
                or abs(primary - metric[PRIMARY]) > 1e-10 or abs(secondary - metric[SECONDARY]) > 1e-10):
            raise ValueError("Validation metrics do not match saved client counts")
    return {name: sum(r["evaluation"][name] for r in rows) / len(rows) for name in (PRIMARY, SECONDARY)}


def select_candidates(scores: dict[str, dict], number: int) -> list[str]:
    alternatives = [name for name in scores if name != "00-reference"]
    alternatives.sort(key=lambda name: (-scores[name][PRIMARY], name))
    if "00-reference" not in scores:
        raise RuntimeError("Reference failed; cannot continue a comparable calibration")
    return ["00-reference", *alternatives[:number]]


def select_configuration(full_scores: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    means = {}
    for candidate, seed_scores in full_scores.items():
        if sorted(seed_scores) != sorted(seeds):
            raise ValueError("Every finalist needs the same complete tuning-seed set")
        means[candidate] = {name: sum(seed_scores[seed][name] for seed in seeds) / len(seeds)
                            for name in (PRIMARY, SECONDARY)}
    if not means or any(not math.isfinite(r[name]) for r in means.values() for name in (PRIMARY, SECONDARY)):
        raise ValueError("Missing/non-finite validation selection scores")
    winner = min(means, key=lambda candidate: (-means[candidate][PRIMARY], candidate))
    return {"selected_candidate": winner, "mean_validation_scores": means,
            "selection_metric": PRIMARY, "tuning_seeds": seeds,
            "tie_break": "candidate identifier ascending"}


def check_gpu_idle(device: str) -> dict:
    if not device.startswith("cuda:"):
        raise ValueError("Campaign requires an explicit CUDA GPU")
    index = int(device.split(":")[1])
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).splitlines()
    gpu = next(row for row in gpu_lines if int(row.split(",")[0]) == index).split(",")
    uuid = gpu[1].strip()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True).splitlines()
    occupied = [row for row in processes if row.split(",")[0].strip() == uuid]
    if occupied or int(gpu[3]) != 0 or int(gpu[2]) > 100:
        raise RuntimeError(f"GPU {index} is occupied; no other work will be interrupted: {occupied}")
    return {"device": device, "uuid": uuid, "memory_used_mib": int(gpu[2]), "utilization_percent": int(gpu[3]),
            "existing_other_gpu_processes": processes}


def search(plan_path: Path, partition: Path, output: Path, device: str, resume: bool) -> dict:
    require_untracked_artifacts(output)
    if not git_metadata()["working_tree_clean"]:
        raise RuntimeError("Commit the reviewed implementation before starting calibration")
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError("Search directory to resume does not exist")
    with (output / ".search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _search_locked(plan, partition, output, device)


def _search_locked(plan: dict, partition: Path, output: Path, device: str) -> dict:
    plan_hash = canonical_hash(plan)
    plan_copy = output / "plan.json"
    if plan_copy.exists():
        if json.loads(plan_copy.read_text()) != plan:
            raise ValueError("Cannot change a started calibration plan")
    else:
        atomic_json(plan_copy, plan)
    split_path = output / "validation_split.json"
    split = prepare_validation_split(partition, split_path, **plan["validation"])
    if split["parent_partition_sha256"] != plan["partition_sha256"]:
        raise ValueError("Validation parent partition mismatch")
    config_dir, logs = output / "configs", output / "logs"
    config_dir.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    by_id = {r["id"]: r for r in plan["candidates"]}
    for row in plan["candidates"]:
        config = candidate_config(plan, row)
        destination = config_dir / (row["id"] + ".json")
        if destination.exists() and json.loads(destination.read_text()) != config:
            raise ValueError("Refusing to overwrite candidate configuration")
        if not destination.exists():
            atomic_json(destination, config)
    campaign = output / "campaign.json"
    records = json.loads(campaign.read_text()) if campaign.exists() else {
        "schema": 1, "status": "calibrating", "plan_sha256": plan_hash,
        "view_sha256": split["view_sha256"], "git": git_metadata(), "device": device,
        "sessions": [], "attempts": [], "failures": []}
    if records["plan_sha256"] != plan_hash or records["view_sha256"] != split["view_sha256"] or records["device"] != device:
        raise ValueError("Calibration campaign identity mismatch")
    if records["git"]["code_sha256"] != git_metadata()["code_sha256"]:
        raise ValueError("Calibration code changed during campaign")
    if (output / "selection.json").exists():
        frozen = json.loads((output / "selection.json").read_text())
        selected = copy.deepcopy(json.loads(REFERENCE.read_text()))
        selected["training"].update(by_id[frozen["selected_candidate"]]["overrides"])
        if (frozen["plan_sha256"] != plan_hash or frozen["view_sha256"] != split["view_sha256"]
                or canonical_hash(selected) != frozen["selected_config_sha256"]):
            raise ValueError("Existing frozen selection identity mismatch")
        selected_path = output / "selected_config.json"
        if selected_path.exists() and json.loads(selected_path.read_text()) != selected:
            raise ValueError("Existing selected configuration differs from frozen choice")
        if not selected_path.exists():
            atomic_json(selected_path, selected)
        records["status"] = "completed"
        records["selection_sha256"] = file_hash(output / "selection.json")
        atomic_json(campaign, records)
        return frozen
    if records["status"] == "completed":
        raise ValueError("Completed campaign is missing its frozen selection")
    session_start = time.perf_counter()
    session = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "process_commit": git_metadata()["commit"]}
    records["sessions"].append(session)
    atomic_json(campaign, records)

    def execute(candidate: str, seed: int, stop: int) -> dict | None:
        run_dir = output / "runs" / candidate / f"seed-{seed}"
        config_path = config_dir / (candidate + ".json")
        if run_dir.exists():
            if not (run_dir / "results.json").exists():
                raise RuntimeError(f"Incomplete initial output preserved at {run_dir}; missing first results/checkpoint")
            saved = json.loads((run_dir / "results.json").read_text())
            config = json.loads(config_path.read_text())
            identity = saved["identity"]
            if (identity["seed"] != seed or identity["config_sha256"] != canonical_hash(config)
                or identity["data_view_sha256"] != split["view_sha256"] or identity["code_sha256"] != records["git"]["code_sha256"]):
                raise ValueError("Existing calibration output identity mismatch")
            if saved["completed_round"] >= stop:
                return saved
        check = check_gpu_idle(device)
        attempt_index = len(records["attempts"]) + 1
        stdout = logs / f"attempt-{attempt_index:03d}-{candidate}-seed-{seed}.log"
        command = [sys.executable, str(REPO / "calibrate_femnist_reconstructed.py"), "trial",
                   "--plan", str(plan_copy), "--config", str(config_path), "--partition", str(partition), "--split", str(split_path),
                   "--output", str(run_dir), "--seed", str(seed), "--device", device, "--stop-after", str(stop)]
        if run_dir.exists():
            command.append("--resume")
        attempt = {"candidate": candidate, "seed": seed, "stop_after": stop,
                   "command": command, "stdout_stderr": str(stdout.relative_to(output)), "gpu_check": check,
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running"}
        records["attempts"].append(attempt)
        atomic_json(campaign, records)
        print(f"START {candidate} seed={seed} target_round={stop}", flush=True)
        start = time.perf_counter()
        with stdout.open("x") as handle:
            process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
            attempt["pid"] = process.pid
            atomic_json(campaign, records)
            try:
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        if (run_dir / "results.json").exists():
                            progress = json.loads((run_dir / "results.json").read_text())
                            print(f"PROGRESS {candidate} seed={seed} round={progress['completed_round']}/{stop}", flush=True)
            except BaseException:
                process.terminate()
                code = process.wait()
                attempt.update(exit_code=code, process_wall_seconds=time.perf_counter() - start,
                               status="interrupted")
                session["wall_seconds"] = time.perf_counter() - session_start
                atomic_json(campaign, records)
                raise
        attempt.update(exit_code=code, process_wall_seconds=time.perf_counter() - start,
                       status="completed" if code == 0 else "failed")
        if (run_dir / "results.json").exists():
            snapshot = logs / f"attempt-{attempt_index:03d}-results.json"
            shutil.copyfile(run_dir / "results.json", snapshot)
            saved = json.loads(snapshot.read_text())
            attempt.update(results_snapshot=str(snapshot.relative_to(output)), results_sha256=file_hash(snapshot),
                           completed_round=saved["completed_round"], peak_rss_mib=saved["peak_rss_mib"],
                           peak_cuda_allocated_mib=saved.get("peak_cuda_allocated_mib"),
                           peak_cuda_reserved_mib=saved.get("peak_cuda_reserved_mib"))
        atomic_json(campaign, records)
        if code:
            records["failures"].append({"candidate": candidate, "seed": seed, "attempt": attempt_index,
                                         "exit_code": code, "log": str(stdout.relative_to(output))})
            atomic_json(campaign, records)
            print(f"FAILED {candidate} seed={seed} exit={code}; artifacts preserved", flush=True)
            return None
        print(f"FINISHED {candidate} seed={seed} seconds={attempt['process_wall_seconds']:.2f}", flush=True)
        return json.loads((run_dir / "results.json").read_text())

    screen = plan["screening"]
    scores = {}
    for candidate in by_id:
        result = execute(candidate, plan["tuning_seeds"][0], screen["rounds"])
        if result is not None:
            scores[candidate] = validation_score(result, screen["first_round"], screen["last_round"])
    finalists = select_candidates(scores, screen["promote_alternatives"])
    promotion = {"scores": scores, "promoted": finalists, "plan_sha256": plan_hash,
                 "metric": PRIMARY, "window": [screen["first_round"], screen["last_round"]]}
    if (output / "promotion.json").exists() and json.loads((output / "promotion.json").read_text()) != promotion:
        raise ValueError("Promotion differs from previously fixed decision")
    atomic_json(output / "promotion.json", promotion)
    print("PROMOTED " + ", ".join(finalists), flush=True)
    full_scores = {}
    window = plan["confirmation"]
    for candidate in finalists:
        seed_scores = {}
        for seed in plan["tuning_seeds"]:
            result = execute(candidate, seed, window["rounds"])
            if result is None:
                break
            seed_scores[seed] = validation_score(result, window["first_round"], window["last_round"])
        if len(seed_scores) == len(plan["tuning_seeds"]):
            full_scores[candidate] = seed_scores
    if "00-reference" not in full_scores:
        raise RuntimeError("Reference did not complete full confirmation; campaign stopped")
    selection = select_configuration(full_scores, plan["tuning_seeds"])
    selected = copy.deepcopy(json.loads(REFERENCE.read_text()))
    selected["training"].update(by_id[selection["selected_candidate"]]["overrides"])
    validate_training_config(selected)
    selection.update(status="frozen", selection_data="training-derived validation only",
                     plan_sha256=plan_hash, view_sha256=split["view_sha256"],
                     partition_sha256=plan["partition_sha256"], full_validation_scores=full_scores,
                     screen_scores=scores, promoted=finalists,
                     selected_config_sha256=canonical_hash(selected),
                     reference_config_sha256=plan["reference_config_sha256"],
                     calibration_code=records["git"],
                     frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for filename, value in [("selection.json", selection), ("selected_config.json", selected)]:
        destination = output / filename
        if destination.exists():
            raise FileExistsError("Refusing to overwrite a frozen selection")
        atomic_json(destination, value)
    session["wall_seconds"] = time.perf_counter() - session_start
    records["status"] = "completed"
    records["selection_sha256"] = file_hash(output / "selection.json")
    atomic_json(campaign, records)
    print(json.dumps(selection, indent=2), flush=True)
    return selection


def finite_tree(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite result or timing")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def verify_definitive_result(result: dict, config: dict, seed: int, code_hashes: dict,
                             test_counts: dict[str, int]) -> dict:
    finite_tree(result)
    identity = result["identity"]
    if (identity["seed"] != seed or identity["mode"] != "definitive"
            or identity["config_sha256"] != canonical_hash(config)
            or identity["partition_sha256"] != config["partition_sha256"]
            or identity["code_sha256"] != code_hashes):
        raise ValueError("Definitive run identity mismatch")
    if result["status"] != "completed" or result["completed_round"] != 200:
        raise ValueError("Definitive run is incomplete")
    history = result["history"]
    if [r["round"] for r in history] != list(range(1, 201)):
        raise ValueError("Definitive round history is incomplete")
    evaluated = [r for r in history if r["evaluation"] is not None]
    if [r["round"] for r in evaluated] != list(range(191, 201)):
        raise ValueError("Definitive test must be restricted to rounds 191--200")
    if any(type(n) is not int or n < 0 for n in result["participations"].values()) or sum(result["participations"].values()) != 3000:
        raise ValueError("Invalid definitive participation counts")
    for row in evaluated:
        metric = row["evaluation"]
        if metric.get("split", "test") != "test":
            raise ValueError("Definitive run contains a different evaluation split")
        counts = metric["clients"]
        if len(counts) != 150 or set(counts) != set(result["participations"]) or set(counts) != set(test_counts):
            raise ValueError("Definitive evaluation must include all 150 clients")
        if any(type(c[name]) is not int for c in counts.values() for name in ("correct", "total", "participations")):
            raise ValueError("Definitive counts and participations must be integers")
        if any(not 0 <= c["correct"] <= c["total"] or c["total"] != test_counts[client]
               or c["participations"] != result["participations"][client] for client, c in counts.items()):
            raise ValueError("Invalid definitive counts")
        total, correct = sum(c["total"] for c in counts.values()), sum(c["correct"] for c in counts.values())
        if total != 2603 or metric["total"] != total or metric["correct"] != correct:
            raise ValueError("Definitive evaluation must contain 2603 original test examples")
        actual = {PRIMARY: 100.0 * correct / total,
                  SECONDARY: sum(100.0 * c["correct"] / c["total"] for c in counts.values()) / len(counts)}
        if any(abs(actual[name] - metric[name]) > 1e-10 for name in actual):
            raise ValueError("Definitive metrics cannot be reconstructed from client counts")
    expected = {name: sum(r["evaluation"][name] for r in evaluated) / 10 for name in (PRIMARY, SECONDARY)}
    if any(abs(expected[name] - result["summary"][name]) > 1e-10 for name in expected):
        raise ValueError("Definitive summary differs from the fixed ten-round mean")
    return expected


def final_campaign(partition: Path, output: Path, device: str, resume: bool) -> dict:
    """Fresh final seed initializations; frozen selection is never reopened."""
    require_untracked_artifacts(output)
    config = json.loads(SELECTED.read_text())
    validate_selected_config(config)
    current_git = git_metadata()
    if not current_git["working_tree_clean"]:
        raise RuntimeError("Commit and push the frozen configuration before definitive training")
    original = json.loads((partition / "manifest.json").read_text())
    if canonical_hash({k: v for k, v in original.items() if k != "partition_sha256"}) != config["partition_sha256"]:
        raise ValueError("Frozen partition manifest changed")
    test_counts = {client["id"]: len(client["test"]) for client in original["clients"]}
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError("Final campaign to resume does not exist")
    with (output / ".final.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records_path = output / "campaign.json"
        identity = {"config_sha256": canonical_hash(config), "partition_sha256": config["partition_sha256"],
                    "selection_sha256": file_hash(FROZEN_RECEIPT), "device": device,
                    "code_sha256": current_git["code_sha256"]}
        records = json.loads(records_path.read_text()) if records_path.exists() else {
            "status": "running", "identity": identity, "git": current_git,
            "attempts": [], "sessions": []}
        if records["identity"] != identity:
            raise ValueError("Final campaign cannot change configuration/code/data/seed policy/device")
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        start = time.perf_counter()
        session = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": current_git["commit"]}
        records["sessions"].append(session)
        atomic_json(records_path, records)
        for seed in config["run_seeds"]:
            run = output / "runs" / f"seed-{seed}"
            if run.exists():
                if not (run / "results.json").exists():
                    raise RuntimeError("Incomplete initial output preserved; missing first result")
                previous = json.loads((run / "results.json").read_text())
                if previous["status"] == "completed":
                    verify_definitive_result(previous, config, seed, identity["code_sha256"], test_counts)
                    if previous["checkpoint_sha256"] != file_hash(run / "checkpoint.pt"):
                        raise ValueError("Completed final checkpoint checksum mismatch")
                    continue
            gpu = check_gpu_idle(device)
            attempt_number = len(records["attempts"]) + 1
            stdout = logs / f"seed-{seed}-attempt-{attempt_number:02d}.log"
            command = [sys.executable, str(REPO / "train_femnist_reconstructed.py"), "run",
                       "--config", str(SELECTED), "--partition", str(partition), "--output", str(run),
                       "--seed", str(seed), "--device", device]
            if run.exists():
                command.append("--resume")
            attempt = {"seed": seed, "command": command, "gpu_check": gpu, "stdout_stderr": str(stdout.relative_to(output)),
                       "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running"}
            records["attempts"].append(attempt)
            atomic_json(records_path, records)
            print(f"START FINAL seed={seed} configuration={identity['config_sha256']}", flush=True)
            run_start = time.perf_counter()
            with stdout.open("x") as handle:
                process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                attempt["pid"] = process.pid
                atomic_json(records_path, records)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            if (run / "results.json").exists():
                                progress = json.loads((run / "results.json").read_text())
                                print(f"PROGRESS FINAL seed={seed} round={progress['completed_round']}/200", flush=True)
                except BaseException:
                    process.terminate()
                    code = process.wait()
                    attempt.update(exit_code=code, process_wall_seconds=time.perf_counter() - run_start, status="interrupted")
                    session["wall_seconds"] = time.perf_counter() - start
                    atomic_json(records_path, records)
                    raise
            attempt.update(exit_code=code, process_wall_seconds=time.perf_counter() - run_start,
                           status="completed" if code == 0 else "failed")
            if (run / "results.json").exists():
                snapshot = logs / f"seed-{seed}-attempt-{attempt_number:02d}-results.json"
                shutil.copyfile(run / "results.json", snapshot)
                result = json.loads(snapshot.read_text())
                attempt.update(results_snapshot=str(snapshot.relative_to(output)), results_sha256=file_hash(snapshot),
                               completed_round=result["completed_round"], peak_rss_mib=result["peak_rss_mib"],
                               peak_cuda_allocated_mib=result.get("peak_cuda_allocated_mib"),
                               peak_cuda_reserved_mib=result.get("peak_cuda_reserved_mib"))
            atomic_json(records_path, records)
            if code:
                session["wall_seconds"] = time.perf_counter() - start
                records["status"] = "failed"
                atomic_json(records_path, records)
                raise RuntimeError(f"Final seed {seed} failed; artifacts preserved, no retuning")
            verify_definitive_result(result, config, seed, identity["code_sha256"], test_counts)
            if result["checkpoint_sha256"] != file_hash(run / "checkpoint.pt"):
                raise ValueError("Final checkpoint checksum mismatch")
            print(f"FINISHED FINAL seed={seed} seconds={attempt['process_wall_seconds']:.2f}", flush=True)
        from train_femnist_reconstructed import summarize_runs
        summary_path = output / "summary.json"
        runs = [output / "runs" / f"seed-{seed}" for seed in config["run_seeds"]]
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            finite_tree(summary)
            values = [json.loads((run / "results.json").read_text())["summary"] for run in runs]
            for name in (PRIMARY, SECONDARY):
                expected = [value[name] for value in values]
                if (summary[name]["values"] != expected
                        or abs(summary[name]["mean"] - statistics.mean(expected)) > 1e-10
                        or abs(summary[name]["std"] - statistics.stdev(expected)) > 1e-10):
                    raise ValueError("Existing summary differs from the five completed run means")
        else:
            summary = summarize_runs(config, runs, summary_path)
        session["wall_seconds"] = time.perf_counter() - start
        records["status"] = "completed"
        records["summary_sha256"] = file_hash(summary_path)
        atomic_json(records_path, records)
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser("search")
    campaign.add_argument("--plan", type=Path, required=True)
    campaign.add_argument("--partition", type=Path, required=True)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--device", required=True)
    campaign.add_argument("--resume", action="store_true")
    definitive = commands.add_parser("final")
    definitive.add_argument("--partition", type=Path, required=True)
    definitive.add_argument("--output", type=Path, required=True)
    definitive.add_argument("--device", required=True)
    definitive.add_argument("--resume", action="store_true")
    trial = commands.add_parser("trial")
    for name in ("plan", "config", "partition", "split", "output"):
        trial.add_argument("--" + name, type=Path, required=True)
    trial.add_argument("--seed", type=int, required=True)
    trial.add_argument("--device", required=True)
    trial.add_argument("--stop-after", type=int, required=True)
    trial.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "search":
        search(args.plan, args.partition, args.output, args.device, args.resume)
        return
    if args.command == "final":
        final_campaign(args.partition, args.output, args.device, args.resume)
        return
    require_untracked_artifacts(args.output)
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    row = next(row for row in plan["candidates"] if row["id"] == config["calibration"]["candidate_id"])
    if config != candidate_config(plan, row) or args.seed not in plan["tuning_seeds"]:
        raise ValueError("Trial does not match the fixed plan")
    partition = TrainingValidationPartition(args.partition, args.split)
    if partition.sha256 != plan["partition_sha256"]:
        raise ValueError("Calibration requires the original partition")
    if args.stop_after not in (plan["screening"]["rounds"], plan["confirmation"]["rounds"]):
        raise ValueError("Use a prespecified stage horizon")
    with CalibrationRunner(config, partition, args.output, args.seed, args.device, args.resume) as runner:
        start = time.perf_counter()
        result = runner.run(args.stop_after)
        result["last_session_wall_seconds"] = time.perf_counter() - start
        atomic_json(args.output / "results.json", result)
        print(json.dumps({key: result[key] for key in ("status", "completed_round", "summary")}, indent=2))


if __name__ == "__main__":
    main()
