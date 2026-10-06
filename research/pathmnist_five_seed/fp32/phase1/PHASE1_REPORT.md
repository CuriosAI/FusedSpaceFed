# Phase 1 — PathMNIST decoder and warm-up

Five original seeds, initial/final round-50 anchors. Fixed 640 training images/client; each anchor is copied and independently replayed on the entire local training set: one encoder-only warm-up epoch, then three joint classification epochs. Restore saved persistent SGD/Adam moments, loader RNG; AMP/scaler/TF32 are disabled; no test, tuning, BN recalibration or head refit. Complete per-client intermediate states remain private. This hypothetical extra local update is not a recovered historical trajectory.

Rows summarize client-uniform means within each seed, then five-seed means and sample SD. Clients own different class pairs across seeds, so client identifier is not a stable semantic group. UNet decoder has a linear output; values outside [0,1] are allowed. Probe runs in Float32 eval with unchanged native BN. Visual grids use a fixed [0,1] display, clipping only for rendering; numeric measurements use raw values.

## initialization

| Stage | MSE ± SD | Decoder/input RMS ± SD | Max abs ± SD | Fused training CE ± SD |
|---|---:|---:|---:|---:|
| before | 0.419526 ± 0.027082 | 0.238129 ± 0.025604 | 0.239318 ± 0.030689 | 2.199164 ± 0.007065 |
| after_warmup | 0.188498 ± 0.107534 | 0.706354 ± 0.284328 | 1.194504 ± 0.501225 | 2.199669 ± 0.007364 |
| after_classification | 0.454676 ± 0.054194 | 0.690554 ± 0.248159 | 1.438579 ± 0.614964 | 0.851539 ± 0.206635 |

Mean MSE change: warm-up -0.231028; classification +0.266178.

## final-round50

| Stage | MSE ± SD | Decoder/input RMS ± SD | Max abs ± SD | Fused training CE ± SD |
|---|---:|---:|---:|---:|
| before | 0.610285 ± 0.047908 | 0.230641 ± 0.031032 | 0.345452 ± 0.086122 | 1.838475 ± 0.289404 |
| after_warmup | 0.607762 ± 0.046656 | 0.231070 ± 0.033444 | 0.350222 ± 0.097901 | 1.846959 ± 0.316049 |
| after_classification | 0.614727 ± 0.039365 | 0.241437 ± 0.037610 | 0.387924 ± 0.149168 | 0.553406 ± 0.344935 |

Mean MSE change: warm-up -0.002522; classification +0.006965.

## Controls and limits

All measured values/checkpoints are finite. Shared classifier/decoder states are bitwise unchanged by warm-up; encoder updates are recorded. Classification changes encoder/decoder/classifier; weight distances and BN-buffer distances are separate. Every attempted FP32 optimizer update is executed; no scaler or skipped AMP step exists. A lower MSE does not imply better classification, and CE training need not preserve faithful reconstruction. The final training probe has already been seen during training. No generalization or causal claim follows from a local replay.

Digits used five domains, dz=64, DigitCNN, optimizer reset per round, one classification epoch and calibrated settings; PathMNIST uses ten label-skewed clients, dz=16, ResNet20V2, persistent optimizers and three classification epochs. Raw MSE scales also differ ([−1,1] versus [0,1]). Interpret qualitative patterns without treating these as controlled cross-benchmark effect sizes.

Phase calendar 173.207 s; process sum 747.503 s. Measured per-seed costs and memory are in artifacts/summary.json; executed commands/exit codes in artifacts/campaign.json.

Twenty preselected seed-42 visual grids are in figures/: two anchors × ten clients, two training examples per assigned class. All five seeds contribute to numeric statistics. Configuration/indices/hashes are ../plan.json, ../probe.json and ../anchors.json.

An earlier replay attempt stopped after saving the first client copy because repository-relative output paths were passed to absolute-path metadata serialization. The output path was canonicalized; no model/loss/optimizer change was made. All five diagnostics restarted from the same immutable anchors. Earlier checkpoints, receipt and logs are preserved in phase1_failed_relative_path/ and logs/phase1-relative-path-failure/; their cost is separate from the successful phase.
