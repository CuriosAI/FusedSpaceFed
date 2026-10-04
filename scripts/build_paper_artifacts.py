"""Reproduce manuscript tables/figures from archived numbers, without training.

Legacy means are transcribed from the immutable pre-revision manuscript, not
represented as recovered run-level results. New FEMNIST means are rebuilt from
all saved client counts. --check compares a fresh build byte for byte.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
BASE_COMMIT = "fc067b423115a2b5e0c51bae5a025b86aeaf37d1"
CODE_COMMIT = "cd433abf545647d64d1ec94db86d998de38e3401"
os.environ.setdefault("MPLCONFIGDIR", str(REPO / "_local/paper_revision/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
sys.path.insert(0, str(REPO))
from fusedspacefed_core import ResNet20V2, UNetSmallAE

METHODS = ["FedAvg", "FedProx", "SCAFFOLD", "FusedSpaceFed"]
DATASETS = ["Path", "Chest", "Derma", "OCT", "Pneumonia", "Retina", "Breast",
            "Blood", "Tissue", "OrganA", "OrganC", "OrganS"]
SHAPES = dict(zip(DATASETS, [(3, 9), (1, 14), (3, 7), (1, 4), (1, 2), (1, 5),
                            (1, 2), (3, 8), (1, 8), (1, 11), (1, 11), (1, 11)]))
LABELS = {"tab:performance": ("dirichlet_0.05", "accuracy_percent"),
          "tab:performance_1": ("dirichlet_0.50", "accuracy_percent"),
          "tab:gradient": ("legacy_gradient", "classifier_gradient_dispersion"),
          "tab:pathological": ("pathological_2", "accuracy_percent"),
          "tab:femnist_results": ("writer_femnist", None)}
CSV_FIELDS = ["setting", "dataset", "method", "metric", "mean", "uncertainty",
              "uncertainty_type", "unit", "reported_runs", "dz", "source_commit", "source_table"]
METRICS = ["sample_weighted_accuracy_percent", "uniform_client_accuracy_percent"]
OUTPUTS: dict[str, bytes] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put(path: str, data: str | bytes) -> None:
    OUTPUTS[path] = data.encode("utf-8") if isinstance(data, str) else data


def json_text(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def legacy_source() -> bytes:
    return subprocess.check_output(["git", "show", f"{BASE_COMMIT}:paper/aistats_2027.tex"], cwd=REPO)


def transcribe_legacy() -> tuple[list[dict], dict]:
    source = legacy_source()
    blocks = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", source.decode(), re.S)
    rows = []
    for block in blocks:
        label = re.search(r"\\label\{([^}]+)\}", block).group(1)
        require(label in LABELS, "Unexpected legacy table")
        setting, metric = LABELS[label]
        body = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        for line in body.splitlines():
            if "&" not in line:
                continue
            cells = [re.sub(r"\\textbf\{([^{}]+)\}", r"\1", c.strip()) for c in line.split("&")]
            cells[-1] = cells[-1].replace(r"\\", "").strip()
            dz = cells[-1] if label in ("tab:performance", "tab:performance_1") else ""
            if metric is None:
                names = ["accuracy_percent", "classifier_gradient_dispersion", "macro_f1_percent", "balanced_accuracy_percent"]
                entries = zip([cells[0]] * 4, names, cells[1:])
                dataset = "FEMNIST_writers"
            else:
                entries = zip(METHODS, [metric] * 4, cells[1:5])
                dataset = cells[0]
            for method, name, cell in entries:
                values = re.findall(r"\d+\.\d+", cell)
                require(1 <= len(values) <= 2, "Malformed numerical legacy cell")
                rows.append(dict(setting=setting, dataset=dataset, method=method, metric=name,
                                 mean=values[0], uncertainty=values[1] if len(values) == 2 else "",
                                 uncertainty_type="reported_dispersion_unverified" if len(values) == 2 else "not_available",
                                 unit="squared_gradient_norm" if "gradient" in name else "percent",
                                 reported_runs="5", dz=dz, source_commit=BASE_COMMIT, source_table=label))
    require(len(rows) == 144, "Incomplete legacy transcription")
    metadata = {"source_commit": BASE_COMMIT, "source_path": "paper/aistats_2027.tex",
                "source_byte_sha256": sha(source), "rows": len(rows),
                "origin": "previously reported manuscript aggregates, not recovered run outputs",
                "missing": ["MedMNIST per-seed outputs/variances", "writer FEMNIST per-seed outputs",
                            "numerical source data of legacy stress-test JPG figures"],
                "uncertainty_policy": "Missing values stay empty. Writer FEMNIST ± values are preserved as reported dispersion; "
                                      "the convention cannot be independently audited from run-level files."}
    return rows, metadata


def table(label: str, caption: str, columns: str, header: str, rows: list[str], wide=True) -> str:
    env = "table*" if wide else "table"
    return (f"% Generated by scripts/build_paper_artifacts.py; do not edit.\n"
            f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\small\n\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{{columns}}}\n"
            f"\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) +
            f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def build_legacy(rows: list[dict]) -> dict:
    by_key = {(r["setting"], r["dataset"], r["method"], r["metric"]): r for r in rows}
    require(len(by_key) == len(rows), "Duplicate legacy cell")
    gains, overview = [], {}
    for setting, filename, label, descriptor, datasets in [
        ("dirichlet_0.05", "medical_strong.tex", "tab:performance", r"strong label skew ($\alpha=0.05$)", DATASETS),
        ("dirichlet_0.50", "medical_moderate.tex", "tab:performance_1", r"moderate label skew ($\alpha=0.50$)", DATASETS),
        ("pathological_2", "pathological.tex", "tab:pathological", "two-class-per-client partitions", ["Path", "Derma", "Retina", "Blood"]),
    ]:
        lines, differences = [], []
        for dataset in datasets:
            cells = [by_key[(setting, dataset, method, "accuracy_percent")] for method in METHODS]
            means = [Decimal(cell["mean"]) for cell in cells]
            baseline = max(range(3), key=lambda index: means[index])
            difference = means[3] - means[baseline]
            differences.append(difference)
            gains.append(dict(setting=setting, dataset=dataset, best_global_method=METHODS[baseline],
                              best_global_mean=str(means[baseline]), fusedspacefed_mean=str(means[3]),
                              difference_percentage_points=str(difference), uncertainty="not_available"))
            printed = [r"\textbf{" + str(mean) + "}" if mean == max(means) else str(mean) for mean in means]
            extra = " & " + cells[0]["dz"] if setting.startswith("dirichlet") else ""
            lines.append(dataset + " & " + " & ".join(printed) + extra + " & " + f"{difference:+.2f}" + r" \\")
        caption = (f"Previously reported MedMNIST mean accuracies (\\%) under {descriptor}. "
                   r"$\Delta$ is FusedSpaceFed minus the highest mean among FedAvg, FedProx and SCAFFOLD, in percentage points. "
                   "Bold identifies the highest reported mean, not statistical significance. Per-seed records and variances are unavailable; "
                   "no uncertainty is inferred. Chest accuracy is element-wise label agreement.")
        has_dz = setting.startswith("dirichlet")
        put("paper/generated/" + filename,
            table(label, caption, "l" + "c" * (6 if has_dz else 5),
                  "Dataset & FedAvg & FedProx & SCAFFOLD & FusedSpaceFed" +
                  (r" & $d_z$" if has_dz else "") + r" & $\Delta$", lines))
        overview[setting] = {"datasets": len(datasets), "positive_mean_margins": sum(d > 0 for d in differences),
                             "min_margin_percentage_points": float(min(differences)),
                             "max_margin_percentage_points": float(max(differences)),
                             "median_margin_percentage_points": float(statistics.median(differences)),
                             "all_margins_descriptive": True}
    put("artifacts/paper_revision/mean_gains.csv", csv_text(gains, list(gains[0])))
    lines = []
    for dataset in ["Path", "Derma", "Retina", "Blood"]:
        means = [Decimal(by_key[("legacy_gradient", dataset, method, "classifier_gradient_dispersion")]["mean"])
                 for method in METHODS]
        lines.append(dataset + " & " + " & ".join(r"\textbf{" + str(m) + "}" if m == min(means) else str(m)
                                                  for m in means) + r" \\")
    put("paper/generated/gradient.tex", table("tab:gradient",
        r"Previously reported final-state classifier-gradient dispersion $\Gamma_t$. "
        "Methods are measured at their own final states, using a small batch estimate; these are descriptive values, "
        "not a same-state test of the decomposition. Per-seed variances and numerical stress-sweep data are unavailable.",
        "lcccc", "Dataset & FedAvg & FedProx & SCAFFOLD & FusedSpaceFed", lines, wide=True))
    lines = []
    names = ["accuracy_percent", "classifier_gradient_dispersion", "macro_f1_percent", "balanced_accuracy_percent"]
    for method in METHODS:
        cells = [by_key[("writer_femnist", "FEMNIST_writers", method, metric)] for metric in names]
        printed = [cell["mean"] + r" $\pm$ " + cell["uncertainty"] for cell in cells]
        lines.append((r"\textbf{FusedSpaceFed}" if method == "FusedSpaceFed" else method) + " & " + " & ".join(printed) + r" \\")
    put("paper/generated/writer_femnist.tex", table("tab:femnist_results",
        "Previously reported natural-writer FEMNIST results, kept distinct from the synthetic-client reconstruction. "
        r"Accuracy, macro F1 and balanced accuracy are percentages; $\Gamma_t$ is gradient dispersion. "
        r"The original mean $\pm$ dispersion is preserved; per-run records and the uncertainty convention cannot be independently audited.",
        "lcccc", r"Method & Accuracy & $\Gamma_t$ & Macro F1 & Balanced accuracy", lines))
    return {"gains": gains, "overview": overview, "by_key": by_key}


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite archived value")
    elif isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, list):
        for child in value: finite(child)


def build_reconstructed() -> dict:
    root = REPO / "artifacts/femnist_reconstructed"
    manifest = json.loads((root / "manifest.json").read_text())
    archived_summary = json.loads((root / "summary.json").read_text())
    for name, metadata in manifest["files"].items():
        contents = (root / name).read_bytes()
        require(sha(contents) == metadata["sha256"] and len(contents) == metadata["bytes"], "Archive integrity mismatch")
    runs, round_rows = [], []
    for entry in manifest["runs"]:
        raw = gzip.decompress((root / entry["results"]).read_bytes())
        require(sha(raw) == entry["results_uncompressed_sha256"], "Decompressed result checksum mismatch")
        result = json.loads(raw)
        finite(result)
        identity = result["identity"]
        require(result["status"] == "completed" and result["completed_round"] == 200, "Incomplete run")
        require(result["git"]["commit"] == CODE_COMMIT and identity["mode"] == "definitive"
                and identity["seed"] == entry["seed"] and identity["config_sha256"] == manifest["config_sha256"]
                and identity["partition_sha256"] == manifest["partition_sha256"], "Run identity mismatch")
        require([h["round"] for h in result["history"]] == list(range(1, 201)), "Incomplete round history")
        evaluated = [h for h in result["history"] if h["evaluation"] is not None]
        require([h["round"] for h in evaluated] == list(range(191, 201)), "Wrong test window")
        for record in evaluated:
            evaluation = record["evaluation"]
            counts = evaluation["clients"]
            require(len(counts) == 150 and all(isinstance(c["correct"], int) and
                    0 <= c["correct"] <= c["total"] and c["total"] > 0 for c in counts.values()), "Invalid client counts")
            correct, total = sum(c["correct"] for c in counts.values()), sum(c["total"] for c in counts.values())
            require(total == 2603 == evaluation["total"] and correct == evaluation["correct"], "Count totals differ")
            weighted = 100.0 * correct / total
            uniform = statistics.mean(100.0 * c["correct"] / c["total"] for c in counts.values())
            for metric, value in zip(METRICS, [weighted, uniform]):
                require(abs(evaluation[metric] - value) < 1e-10, "Round accuracy differs from counts")
            round_rows.append(dict(seed=entry["seed"], round=record["round"], correct=correct, total=total,
                                   sample_weighted_accuracy_percent=weighted, uniform_client_accuracy_percent=uniform))
        run = dict(seed=entry["seed"], sample_weighted_accuracy_percent=statistics.mean(
                       r["sample_weighted_accuracy_percent"] for r in round_rows if r["seed"] == entry["seed"]),
                   uniform_client_accuracy_percent=statistics.mean(
                       r["uniform_client_accuracy_percent"] for r in round_rows if r["seed"] == entry["seed"]),
                   process_wall_seconds=entry["process_wall_seconds"], peak_rss_mib=entry["peak_rss_mib"],
                   peak_cuda_allocated_mib=entry["peak_cuda_allocated_mib"],
                   peak_cuda_reserved_mib=entry["peak_cuda_reserved_mib"])
        for metric in METRICS:
            require(abs(run[metric] - result["summary"][metric]) < 1e-10, "Seed mean differs from counts")
        runs.append(run)
    require([r["seed"] for r in runs] == [41, 42, 43, 44, 45], "Missing/reordered seed")
    stats = {metric: {"mean": statistics.mean(r[metric] for r in runs),
                      "std": statistics.stdev(r[metric] for r in runs)} for metric in METRICS}
    for metric in METRICS:
        for field in ["mean", "std"]:
            require(abs(stats[metric][field] - archived_summary[metric][field]) < 1e-10, "Summary differs from independently derived statistic")
    put("artifacts/paper_revision/reconstructed_seed_metrics.csv", csv_text(runs, list(runs[0])))
    put("artifacts/paper_revision/reconstructed_round_metrics.csv", csv_text(round_rows, list(round_rows[0])))
    lines = [f"{r['seed']} & {r[METRICS[0]]:.4f} & {r[METRICS[1]]:.4f}" + r" \\" for r in runs]
    lines += [r"\midrule", f"Mean & {stats[METRICS[0]]['mean']:.4f} & {stats[METRICS[1]]['mean']:.4f}" + r" \\",
              f"Sample SD & {stats[METRICS[0]]['std']:.4f} & {stats[METRICS[1]]['std']:.4f}" + r" \\"]
    put("paper/generated/reconstructed_femnist.tex", table("tab:femnist_reconstructed",
        "Separate reconstructed FEMNIST study: all five run means over rounds 191--200. "
        "Both accuracies are percentages; sample SD is across these five means, ddof=1, in percentage points. "
        "The first metric is primary. These are not controlled FedRep comparisons.", "lcc",
        "Seed & Sample-weighted & Client-uniform", lines, wide=False))
    with (REPO / "docs/femnist_fedrep_reference.csv").open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    require(len(reference) == 15 and all(r["uncertainty"] == "" and r["result_origin"] == "reported" for r in reference),
            "Published reference changed")
    reference_lines = [r["method"].replace("FedRep (Ours)", "FedRep") + " & " + r["value"] + r" \\" for r in reference]
    put("paper/generated/fedrep_published.tex", table("tab:fedrep_published",
        r"External published references, transcribed from Collins et al., Table 1, FEMNIST (150,3). "
        "Accuracy in percent; uncertainties are not reported. No method in this table was rerun here. "
        "FusedSpaceFed is deliberately reported in a separate table because datasets/protocols are not controlled jointly.",
        "lc", "Published method & Reported accuracy", reference_lines, wide=False))
    costs = [f"{r['seed']} & {r['process_wall_seconds']/60:.2f} & {r['peak_rss_mib']:.2f}" + r" \\" for r in runs]
    put("paper/generated/reconstructed_costs.tex", table("tab:reconstructed_costs",
        "Measured costs of the five reconstructed runs on one RTX 6000 Ada GPU, run sequentially. "
        "Time is the full process wall time; RSS is process RAM. Torch CUDA peaks were 78.64 MiB allocated "
        "and 96 MiB reserved in every run, excluding unmeasured driver/context memory.",
        "lcc", "Seed & Wall time (min) & RSS (MiB)", costs, wide=False))
    return {"runs": runs, "statistics": stats, "manifest": manifest}


def build_capacity(legacy) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        channels, classes = SHAPES[dataset]
        dz = int(legacy["by_key"][("dirichlet_0.05", dataset, "FusedSpaceFed", "accuracy_percent")]["dz"])
        classifier, ae = ResNet20V2(classes, channels), UNetSmallAE(channels, dz)
        c = sum(p.numel() for p in classifier.parameters())
        e, d = sum(p.numel() for p in ae.encoder_parameters()), sum(p.numel() for p in ae.decoder_parameters())
        c_bytes = sum(t.numel()*t.element_size() for t in classifier.state_dict().values())
        d_bytes = sum(t.numel()*t.element_size() for t in ae.decoder_state().values())
        rows.append(dict(dataset=dataset, in_channels=channels, classes=classes, dz=dz,
                         classifier_parameters=c, private_encoder_parameters=e, shared_decoder_parameters=d,
                         total_pipeline_parameters=c+e+d, parameter_overhead_percent=100*(e+d)/c,
                         classifier_state_bytes=c_bytes, decoder_state_bytes=d_bytes,
                         fedavg_bidirectional_bytes_round_10_clients=20*c_bytes,
                         fusedspacefed_bidirectional_bytes_round_10_clients=20*(c_bytes+d_bytes)))
    put("artifacts/paper_revision/model_capacity.csv", csv_text(rows, list(rows[0])))
    lines = [f"{r['dataset']} & {r['dz']} & {r['classifier_parameters']} & {r['private_encoder_parameters']} & "
             f"{r['shared_decoder_parameters']} & {r['parameter_overhead_percent']:.1f}" + r" \\" for r in rows]
    put("paper/generated/model_capacity.tex", table("tab:model_capacity",
        "Static parameter accounting from the archived implementation for the medical configurations; no training is performed. "
        "C is the shared ResNet classifier, E is one private encoder and D is the shared decoder. "
        "Overhead is 100(E+D)/C. These counts quantify added capacity, not measured medical runtime or a matched-budget comparison.",
        "lccccc", r"Dataset & $d_z$ & C & E & D & Overhead (\%)", lines))
    return rows


def save_figure(fig, name):
    for extension in ("pdf", "svg"):
        stream = io.BytesIO()
        metadata = {"CreationDate": None, "ModDate": None, "Creator": "FusedSpaceFed reproducible analysis"} if extension == "pdf" else {"Date": None}
        fig.savefig(stream, format=extension, metadata=metadata, bbox_inches="tight")
        contents = stream.getvalue()
        if extension == "svg":
            contents = b"\n".join(line.rstrip() for line in contents.splitlines()) + b"\n"
        put(f"paper/figures/{name}.{extension}", contents)
    plt.close(fig)


def build_figures(legacy, reconstructed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.hashsalt": "FusedSpaceFed-paper-revision-v1",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    y = np.arange(len(DATASETS))
    for setting, offset, color, marker, label in [("dirichlet_0.05", -.12, "#0072B2", "o", r"Strong skew ($\alpha=0.05$)"),
                                                 ("dirichlet_0.50", .12, "#D55E00", "s", r"Moderate skew ($\alpha=0.50$)")]:
        rows = [next(r for r in legacy["gains"] if r["setting"] == setting and r["dataset"] == d) for d in DATASETS]
        x = [float(r["difference_percentage_points"]) for r in rows]
        ax.scatter(x, y+offset, color=color, marker=marker, s=36, label=label, zorder=3)
        for value, position in zip(x, y+offset):
            ax.annotate(f"{value:+.2f}", (value, position), xytext=(6 if value >= 0 else -6, 0),
                        textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=8, color=color)
    ax.axvline(0, color="0.4", lw=1)
    ax.set(yticks=y, yticklabels=DATASETS, xlabel="FusedSpaceFed − highest global-baseline mean (percentage points)", xlim=(-1.2, 8.6))
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); save_figure(fig, "heterogeneity_gains")
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    datasets = ["Path", "Derma", "Retina", "Blood"]
    for j, (method, color, marker) in enumerate(zip(METHODS, ["#666666", "#E69F00", "#009E73", "#0072B2"], ["o", "s", "^", "D"])):
        x = [float(legacy["by_key"][("pathological_2", d, method, "accuracy_percent")]["mean"]) for d in datasets]
        positions = np.arange(4) + (j-1.5)*.12
        ax.scatter(x, positions, color=color, marker=marker, s=45, label=method, zorder=3)
        if method == "FusedSpaceFed":
            for value, pos in zip(x, positions):
                ax.annotate(f"{value:.2f}", (value,pos), xytext=(7,0), textcoords="offset points", va="center", fontsize=9, color=color)
    ax.set(yticks=np.arange(4), yticklabels=datasets, xlabel="Reported mean accuracy (%) — two classes per client", xlim=(0,60))
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
    ax.legend(loc="lower right", ncol=2, frameon=False); fig.tight_layout(); save_figure(fig, "pathological_accuracy")
    fig, axes = plt.subplots(1,2,figsize=(8.2,3.4),sharey=True)
    for ax, metric, title in zip(axes, METRICS, ["Sample-weighted (primary)", "Client-uniform"]):
        values = [r[metric] for r in reconstructed["runs"]]
        mean, std = reconstructed["statistics"][metric].values()
        ax.scatter(range(5), values, color="#0072B2", s=45, zorder=3)
        ax.errorbar(5.5, mean, yerr=std, color="#D55E00", marker="D", capsize=5, linestyle="none", label="Mean ± sample SD")
        ax.axhline(mean, color="#D55E00", lw=.9, linestyle="--", alpha=.6)
        ax.set(xticks=[0,1,2,3,4,5.5],xticklabels=[41,42,43,44,45,"Mean"],title=title,ylim=(40,92),xlim=(-.5,6.2),xlabel="Run seed")
        ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Mean accuracy in rounds 191–200 (%)")
    axes[1].legend(loc="lower right",frameon=False,fontsize=8)
    fig.tight_layout(); save_figure(fig,"femnist_reconstructed_seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only byte comparison with generated files")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows, provenance = transcribe_legacy()
    put("artifacts/paper_revision/legacy_metrics.csv", csv_text(rows, CSV_FIELDS))
    put("artifacts/paper_revision/legacy_source_manifest.json", json_text(provenance))
    legacy = build_legacy(rows)
    reconstructed = build_reconstructed()
    capacity = build_capacity(legacy)
    build_figures(legacy,reconstructed)
    notes = {"legacy_mean_comparisons": legacy["overview"], "reconstructed_statistics": reconstructed["statistics"],
             "primary_metric": METRICS[0], "primary_metric_fixed_before_analysis": True,
             "training_or_model_evaluation_performed": False, "new_campaigns": 0,
             "capacity_overhead_range_percent": [min(r["parameter_overhead_percent"] for r in capacity),
                                                 max(r["parameter_overhead_percent"] for r in capacity)],
             "legacy_significance_claims": False, "legacy_error_bars_invented": False}
    put("artifacts/paper_revision/analysis_summary.json", json_text(notes))
    inputs = ["artifacts/femnist_reconstructed/manifest.json", "docs/femnist_fedrep_reference.csv", "fusedspacefed_core.py"]
    report = {"schema": 1, "builder_sha256": sha(Path(__file__).read_bytes()), "source_commit": BASE_COMMIT,
              "python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__, "torch": torch.__version__,
              "inputs_byte_sha256": {name: sha((REPO/name).read_bytes()) for name in inputs},
              "legacy_source": provenance, "output_byte_sha256": {name: sha(data) for name,data in sorted(OUTPUTS.items())},
              "notes": "Reproduces aggregates and figures only; original medical runs and legacy JPG curves are not regenerated."}
    put("artifacts/paper_revision/build_manifest.json",json_text(report))
    for name, contents in OUTPUTS.items():
        path = REPO/name
        if args.check:
            require(path.exists() and path.read_bytes()==contents, "Stale/missing generated artifact: "+name)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(contents)
    print(json_text({"status":"checked" if args.check else "built", "files":len(OUTPUTS), **notes}))


if __name__ == "__main__":
    main()
