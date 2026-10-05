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

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/schroeder/miniconda3/envs/general_ml/bin/python -B -m pytest tests research/feature_shift_digits/tests research/capacity_compute_control/tests -q
```
