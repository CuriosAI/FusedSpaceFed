# Reproducible manuscript analysis

This package derives tables, figures and static capacity counts from existing
records. It performs no training, model evaluation, tuning or new experiment.

From the repository root, using the existing `general_ml` environment:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python scripts/build_paper_artifacts.py
/home/schroeder/miniconda3/envs/general_ml/bin/python scripts/build_paper_artifacts.py --check
```

The second command rebuilds in memory and requires byte equality for every
generated file, including PDF/SVG figures. It also checks the original FEMNIST
archive checksums and reconstructs the five means and sample SDs from saved
client correct/total counts. No GPU or dataset download is required. Runtime:
Python 3.12.7, NumPy 2.0.2, Matplotlib 3.9.2, Torch 2.5.1+cu124.

## Data origins and statistical scope

- `legacy_metrics.csv` preserves all 144 metric entries of the five original
  manuscript tables at commit `fc067b423115a2b5e0c51bae5a025b86aeaf37d1`.
  The builder reads that immutable Git object; `legacy_source_manifest.json`
  records its SHA-256. These are manuscript aggregates, not recovered raw runs.
- The MedMNIST tables contain means only. Variances, per-seed outputs and
  covariance between methods are unavailable. Blank uncertainty fields remain
  blank. The writer FEMNIST `±` numbers are preserved as originally reported
  dispersion; their convention cannot be independently audited without the
  original run records. No missing SD, confidence interval or p-value is inferred.
- `mean_gains.csv` subtracts the highest reported mean of FedAvg, FedProx and
  SCAFFOLD from the FusedSpaceFed mean per dataset/setting. It uses decimal
  arithmetic on the two-decimal source values. Wins and margins are descriptive,
  not tests of significance. Chest uses multi-label element-wise agreement;
  its score is not pooled with multiclass scores into a clinical performance
  aggregate. Negative margins at alpha 0.50 are retained.
- `reconstructed_seed_metrics.csv` and `reconstructed_round_metrics.csv` derive
  from the five full results in `../femnist_reconstructed/`. All 150 client
  counts, 2603 test examples and rounds 191–200 are checked for every run.
  Primary metric: sample-weighted accuracy. Secondary: uniform client accuracy.
  Each seed is a mean of ten rounds; SD is across five run means with `ddof=1`,
  not across 50 independent test sets. All seeds are included.
- Published references are read from `../../docs/femnist_fedrep_reference.csv`,
  kept unchanged and generated into a separate table. The reconstruction is
  not a controlled rerun of those methods or an exact FedRep replication.
- `model_capacity.csv` instantiates the archived ResNet20-v2 and UNetSmallAE on
  CPU to count parameters/state bytes. Dataset channels/classes agree with the
  existing MedMNIST 3.0.2 metadata; dz comes from the original tables. This is
  static accounting, not measured medical training time, FLOPs or energy.
  Transferred state bytes include model buffers; parameter counts do not.

## Generated files

`../../paper/generated/` contains nine complete LaTeX table environments.
`../../paper/figures/` contains three plots in PDF and SVG:

1. `heterogeneity_gains`: every dataset at alpha 0.05 and 0.50, including losses;
2. `pathological_accuracy`: all four methods on all four two-class settings;
3. `femnist_reconstructed_seeds`: all five seeds for both metrics, with a
   separate mean and sample-SD bar; no published baseline is overlaid.

The legacy mean plots have no error bars because their variances are missing.
The original JPG stress-test curves are preserved unchanged; their numerical
source data were not available and they cannot be faithfully regenerated.
The generated plots use the name FusedSpaceFed consistently. Their PDF/SVG
metadata omits timestamps and uses a fixed SVG hash salt for reproducibility.

`analysis_summary.json` records descriptive wins/margins and independently
reconstructed statistics. `build_manifest.json` records code, input/output
hashes and library versions, excluding its own hash to avoid self-reference.
The complete earlier FEMNIST results, scientific code, configuration,
partition and training checkpoints are not modified by this analysis.
