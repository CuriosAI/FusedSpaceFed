# Original-setting PathMNIST: separate FP32 campaign

Five fresh full-method runs (42–46), followed in order by decoder/warm-up replay,
15 paired component ablations, and original/fused gradient diagnostics. No
tuning, head refit, baseline or manuscript change. The previous FP16 campaign,
including the seed43 overflow, remains in its original locations unchanged.

Only precision changes: `use_amp=false`, no autocast or GradScaler, Float32
models/activations/optimizer arithmetic, CUDA matmul and cuDNN TF32 disabled.
All other paper settings are fixed in `plan.json`: ResNet20V2, UNetSmallAE dz16,
50 rounds, ten fully participating clients, batch128, one encoder-only MSE
warm-up epoch and three joint CE epochs, persistent SGD0.01/Adam0.001, native BN,
uniform shared-state aggregation. Full runs restore the exact original round-0
weights, empty optimizers and per-client shuffle generators in
`initial_sources.json`; no previously trained checkpoint is resumed. The
partition is frozen per seed and paired across variants. Seed42 originally
used two workers; process RNG is recorded anew (its CPU coordinator and first
worker CUDA state are restored); there is no dropout or random augmentation,
and its exact client loader generators determine data order.

One final test at round50; primary metric is the uniform mean of ten private
pipelines, each evaluated on the same 7180 test images. Five-seed mean and sample
SD (ddof1) are computed from all seeds. No checkpoint or seed selection. Changes
of precision can change training trajectories; these are precision-controlled
reruns of fixed settings, not a claim of bitwise equality with FP16.

FP32-specific entry points reuse the historical client, data preparation and
audited diagnostic mathematics without editing their scientific sources. Full
data/checkpoints/logs stay in `_local/pathmnist_five_seed/fp32/` on thanos. Each
round writes a resumable complete checkpoint, with shared C/D, ten private
encoders, BN buffers, persistent optimizer states, scaler=None, model/loader/RNG
states, configuration, source hashes and exact partition indices. Results and
timings are archived losslessly, with private checkpoint paths and SHA256.

Use `/home/schroeder/miniconda3/envs/general_ml/bin/python` in all commands:

```bash
python -m research.pathmnist_five_seed.fp32.prepare_full
python -u research/pathmnist_calibrated/controller.py \
  --queue research/pathmnist_five_seed/fp32/full_queue.json \
  --receipt _local/pathmnist_five_seed/fp32/full_campaign.json \
  --logs _local/pathmnist_five_seed/fp32/logs/full
python -m research.pathmnist_five_seed.fp32.archive originals
# After each completed phase, archive/report/commit before the next phase.
python -m research.pathmnist_five_seed.fp32.setup_phases 1
python -u research/pathmnist_calibrated/controller.py \
  --queue research/pathmnist_five_seed/fp32/phase1/queue.json \
  --receipt _local/pathmnist_five_seed/fp32/phase1/campaign.json \
  --logs _local/pathmnist_five_seed/fp32/logs/phase1
python -m research.pathmnist_five_seed.fp32.archive 1
# Repeat the preceding three commands with phase2, then phase3.
```

Existing outputs are protected; continuation requires explicit `--resume` with
identical config/source hashes. Full runs use three slots on GPU1/two on GPU0;
phase queues allow three per GPU and check free VRAM, without signaling external
processes. The fixed training-only probes are identical to the previous plan:
640 examples/client, five128-example batches, initial/final anchors. Phase1
replays one warm-up/three CE epochs on **copies**, restoring persistent
optimizers; it does not recover an actual past within-round trajectory. Phase3
checks unchanged weights/BN/RNG under native-eval and batch-stateless BN, uses
Float32 gradients and Float64 CPU Gram reductions, and audits Γf=Γo+B+Φ.

Ablations: no warm-up; shared encoder weights (local optimizer moments remain
private); classify only decoder output. They change one component at a time,
are paired by original initialization/seed/partition, and are not compute-matched.
Digits comparisons are descriptive: architecture, client heterogeneity, local
epochs, optimizer persistence and calibrated settings differ.
