# PathMNIST: reduced calibration budget

The active budget is 24 predefined candidates, 20 rounds, screening seed 142,
then one fresh full-training run with seed 42 and 50 rounds. There are no
confirmations or extensions. `BUDGET_REDUCTION.md` supersedes the original
budget in the immutable `search_plan.json`; `PROTOCOL.md` describes the
training-derived validation and the two declared BN inference modes.

Use the existing environment, without installing packages:

```bash
PYTHON=/home/schroeder/miniconda3/envs/general_ml/bin/python
```

The original screening and the identical-configuration retries are recorded
by these controller commands. Retries address an auxiliary import-name
collision, rather than a change to model updates. Completed outputs must not
be overwritten; these commands are historical execution records.

```bash
$PYTHON -u research/pathmnist_calibrated/controller.py \
  --queue research/pathmnist_calibrated/screening_queue.json \
  --receipt _local/pathmnist_calibrated/screening_campaign.json \
  --logs _local/pathmnist_calibrated/logs/screening

$PYTHON -u research/pathmnist_calibrated/controller.py \
  --queue research/pathmnist_calibrated/screening_retry_queue.json \
  --receipt _local/pathmnist_calibrated/screening_retry_campaign.json \
  --logs _local/pathmnist_calibrated/logs/screening_retry
```

After every screening process has exited, freeze the maximum validation
score at round 20. Exact count ties prefer native BN, then the original
candidate order. Commit and push the calibration and frozen configuration
before launching the final run.

```bash
$PYTHON research/pathmnist_calibrated/selection_tools.py freeze-screening
$PYTHON research/pathmnist_calibrated/reduced_archive.py calibration
$PYTHON research/pathmnist_calibrated/reduced_archive.py verify

$PYTHON -u research/pathmnist_calibrated/controller.py \
  --queue research/pathmnist_calibrated/final_queue.json \
  --receipt _local/pathmnist_calibrated/final_campaign.json \
  --logs _local/pathmnist_calibrated/logs/final

$PYTHON research/pathmnist_calibrated/verify_checkpoints.py
$PYTHON research/pathmnist_calibrated/reduced_archive.py final
$PYTHON research/pathmnist_calibrated/reduced_archive.py verify
```

The final controller runs the existing `runner.py` once on physical GPU 1.
It preserves every selected training hyperparameter, including a cosine
schedule's 100-round horizon, and evaluates only the frozen inference mode
on test after round 50. No checkpoint selection or post-test tuning occurs.

`CALIBRATION_REPORT.md`, `FINAL_REPORT.md`, and `artifacts/` contain the
numerical reports, lossless result archives, timings and integrity manifests.
All complete checkpoints and logs remain on thanos under
`_local/pathmnist_calibrated/`. Checkpoint audits cover initial, native final
and selected final states, including all ten private encoders, shared
components, BN buffers, optimizers, scaler when applicable, RNG and exact
training indices. The single final seed has no between-seed SD.

The manuscript and other experiments are unchanged. No baseline, ablation or
decoder/gradient diagnostic is run as part of this reduced campaign.
