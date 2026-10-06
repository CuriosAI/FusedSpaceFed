# Original PathMNIST campaign and paired mechanism diagnostics

Current outcome: [numerical block at seed 43](originals/ORIGINALS_REPORT.md).
Four original seeds are complete; no five-seed mean/SD and no mechanism phase
are claimed complete. The fixed-protocol seed43 retry reproduced FP16 overflow.

This campaign implements the user request of 6 October 2026, without changing the
manuscript, previous experiments, or any recovered/tuned/head-refitted checkpoint.
Only FusedSpaceFed and its three component ablations are trained. No baselines.

## Frozen original protocol

`plan.json` records all settings. Ten clients, two classes per client, full
participation, 50 rounds, one encoder-only MSE warm-up epoch and three joint CE
epochs per round. ResNet20V2 (RGB, nine outputs), UNetSmallAE (dz=16), additive
`x + D(E_i(x))`, SGD classifier lr=0.01 (no momentum), Adam AE lr=0.001
(PyTorch defaults). Local optimizer moments and FP16 AMP scalers persist across
rounds. Classifier/decoder states, including native BN buffers, are averaged
uniformly in increasing client order. No clipping, schedule, adaptation, tuning,
BN recalibration, or head refit. AMP-skipped steps are counted rather than
silently treated as executed updates.

We reuse the immutable PathMNIST-64 cache of `research/pathmnist_pathological`
(PIL resize to 32x32 then ToTensor, RGB [0,1], no augmentation). The original
`pathological_partition` uses the run seed for both indices and initialization.
Accordingly, seeds 42–46 have five frozen partitions; **within each seed**, the
full method and all ablations use exactly the same partition and initial model.
Between-seed SD therefore includes partition and initialization variability.
All 89,996 training indices occur once, every client has two classes. Exact
indices remain in private manifests and complete checkpoints; hashes and client
statistics are published in `data_identity.json`.

Seed 42 is reused read-only from `_local/pathmnist_pathological/seed-42`:
39.61977715877437%, 50 rounds. `reused_seed42.json` freezes its artifact hashes.
Seeds 43–46 start from their own initialization. Each new run resides on one
GPU; independent runs share the two GPUs. This changes execution scheduling,
not local epochs, optimizer updates or aggregation. The seed-42 reference used
two worker GPUs; exact cross-device bitwise equality is not asserted.

Test is evaluated **once, after round 50**. Each of the ten private-encoder
pipelines, with current shared classifier/decoder and native BN, predicts all
7,180 official test images in Float32. The reported metric is their uniform
mean (71,800 predictions; not 71,800 distinct test images). No best seed or
checkpoint is selected. Statistics use all five seeds, sample SD (`ddof=1`).
The historical 50.94% in Table 4 is not silently substituted for measured data.

## Reproduction

Python: `/home/schroeder/miniconda3/envs/general_ml/bin/python`.
Reuse the existing verified environment; install nothing.

```bash
python -m research.pathmnist_five_seed.data
python research/pathmnist_calibrated/controller.py \
  --queue research/pathmnist_five_seed/full_queue.json \
  --receipt _local/pathmnist_five_seed/full_campaign.json \
  --logs _local/pathmnist_five_seed/logs/full
```

The runner creates attempt directories and refuses overwrite. To resume an
interrupted attempt, call `runner.py` with the same `--config`, `--output` and
`--device`, plus `--resume`; source/config/partition hashes must match. Initial,
latest and final checkpoints contain shared states, all ten private encoders,
BN buffers, all local optimizers/scalers, loader RNGs, process RNGs and exact
indices. Full logs/checkpoints/cache stay on thanos under `_local/`.

## Three diagnostic phases

1. Training-only fixed probes at initial/final anchors; independent local replay
   on copies with saved optimizers: warm-up 1 epoch then classification 3 epochs.
   This is a hypothetical extra local update, not recovery of historical
   within-round trajectories. Record raw decoder magnitude/reconstruction,
   shared/private state changes and representative training images.
2. Five seed-paired 50-round runs per ablation, no retuning: no warm-up (zero MSE
   epochs), shared encoder (uniformly aggregate encoder weights; keep local
   optimizer moments), decoder-only (classify `D(E_i(x))` without additive x).
   Each changes one component; removing warm-up also changes compute and data
   loader progression, so this is not a FLOPs-matched experiment.
3. Float32 original/fused gradients at the same original full model state,
   using five fixed batches per client. Reuse the Digits decomposition and
   controls: native-eval BN and training-batch BN with buffers restored before
   each branch. Publish Gamma original/fused, B, Phi, gradient norms, normalized
   dispersion/cosines and decoder dispersion. No optimization or test use.

The phase reports will distinguish seed variability from client variability,
and differences in data, model and optimizer persistence from Digits. These
diagnostics do not establish a universal causal explanation of accuracy.
