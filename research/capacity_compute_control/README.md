# Minimal capacity/compute control

See PROTOCOL.md and search_plan_v2.json for decisions frozen before calibration.
The initial three-candidate plan is retained but was never run: its CLI preflight
failed before any worker, then the user extended tuning to AE LR and clipping.
Use the existing general_ml environment; no dependencies are installed.
Data is reused from `_local/feature_shift_digits/prepared`, verified against the
same frozen parent manifest. validation_split.json contains source row indices,
not images. Profiling uses synthetic batches; no held-out test for development.

Calibration command (one worker per GPU, max two; refuses overwrite):

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/capacity_compute_control/controller.py --queue research/capacity_compute_control/validation_queue_v2.json --receipt _local/capacity_compute_control/validation_v2_campaign.json --logs _local/capacity_compute_control/logs/validation_v2
```

After all eighteen runs exit successfully, select_hyperparameters.py writes an immutable selection
from final-round training-derived validation counts. Both classifier LR and
clipping are selected for each method; Fused additionally selects AE LR. Final configurations and
queue are committed before their workers start. Existing Fused results may be
reused only after exact settings/data/scientific-source verification; otherwise
three Fused runs start from their own seed initialization.

The controller preserves stdout/stderr, exit codes, commands, actual dates and
durations privately. runner.py saves progressive JSON, timing JSONL and atomic
checkpoints, including private encoders, shared states, RNG/loader state and
FedAvg budget carry. Explicit --resume requires the same config/source/commit/
device; no implicit restart or overwrite. Old experiments and the manuscript
are never modified.

Plan v2 completed all eighteen workers, with identity/count/compute audits passed.
Validation selected C LR=.02 and clipping=2 for both; Fused AE LR=.0003.
These alter the original updates, so all six final runs start from scratch:
three paired seeds 42–44, full 743/client and 300 rounds. reuse_proof.json records
why no previous Fused checkpoint/result is reused. The decision is frozen in
selection.json; final configs/queue are committed before any final evaluation.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/capacity_compute_control/controller.py --queue research/capacity_compute_control/final_queue.json --receipt _local/capacity_compute_control/final_campaign.json --logs _local/capacity_compute_control/logs/final
```

To reproduce selection from the private raw calibration results, run
select_hyperparameters.py followed by freeze_final_configs.py in a fresh copy
without existing selection/final registry. Both refuse overwrite. The archive
will provide all numerical calibration results without checkpoints/images.

Completed numeric delivery: CAPACITY_COMPUTE_REPORT.md, artifacts/summary.json,
all eighteen calibration results and six definitive results compressed losslessly,
and their timing JSONL. artifact_manifest.json hashes every public payload file.
The independent audit requires only the Python standard library and reads no
images or checkpoints; run from the repository root (the parent Digits partition
manifest is kept in its existing directory):

```bash
python3 research/capacity_compute_control/audit_results.py verify
```

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/schroeder/miniconda3/envs/general_ml/bin/python -B -m pytest tests research/feature_shift_digits/tests research/capacity_compute_control/tests -q
```
