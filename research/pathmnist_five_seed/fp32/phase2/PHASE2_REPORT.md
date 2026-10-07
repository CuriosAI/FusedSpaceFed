# Phase 2 — Five-seed paired PathMNIST component ablations

Each ablation starts from zero, restoring the corresponding original full run initial checkpoint, exact frozen partition and loader generators. Fifty rounds, all ten clients, batch 128, SGD 0.01 and Adam 0.001, persistent local optimizer states, FP32 without AMP/scaler/TF32, native BN. One final test, no tuning or adaptation. The full original campaign is reused; no tuned or head-refitted state enters these comparisons.

| Method | Seed 42 | 43 | 44 | 45 | 46 | Mean ± sample SD (%) | Paired delta ± SD (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 26.516713 | 33.013928 | 37.855153 | 30.892758 | 28.873259 | 31.430362 ± 4.323117 | reference |
| no-warmup | 18.303621 | 13.686630 | 12.991643 | 14.770195 | 18.408078 | 15.632033 ± 2.566263 | -15.798329 ± 6.721505 |
| shared-encoder | 26.908078 | 29.791086 | 18.969359 | 29.261838 | 31.824513 | 27.350975 ± 5.001643 | -4.079387 ± 8.593251 |
| decoder-only | 13.774373 | 9.272981 | 15.091922 | 14.548747 | 15.557103 | 13.649025 ± 2.534644 | -17.781337 ± 5.189688 |

## Paired interpretation

- no-warmup: lower than the full method in 5/5 pairs, higher in 0/5; ablation minus full -15.798329 ± 6.721505 percentage points. These are descriptive paired effects at fixed settings, without a significance claim or selection of a winning variant.
- shared-encoder: lower than the full method in 3/5 pairs, higher in 2/5; ablation minus full -4.079387 ± 8.593251 percentage points. These are descriptive paired effects at fixed settings, without a significance claim or selection of a winning variant.
- decoder-only: lower than the full method in 5/5 pairs, higher in 0/5; ablation minus full -17.781337 ± 5.189688 percentage points. These are descriptive paired effects at fixed settings, without a significance claim or selection of a winning variant.

No-warmup removes only the reconstruction phase; classification remains three epochs. This changes compute, Adam history and shuffle progression inherently; it is not compute-matched. Shared-encoder additionally averages encoder model state uniformly after every round, retaining separate persistent local Adam moments. Decoder-only removes the additive raw input from CE and test; warm-up is unchanged. Neither optimizer moments nor BN statistics are specially recalibrated. Full and ablated FP32 runs each use one GPU process, identical original initial weights and loader states. Changing precision can change the trajectory relative to the preserved FP16 campaign.

All accuracies are reconstructed from the ten saved client correct/total counts (7180 images/pipeline). All five seeds are shown, no best-seed selection. Paired deltas are calculated within seed before reporting mean/sample SD; this is descriptive, with five pairs and no significance claim. Between-seed variation includes the run-seeded pathological partition.

Digits component results for context only: full 85.862359 ± 0.846930%, no-warmup 85.714598 ± 0.789950%, shared-encoder 86.178544 ± 0.468069%, decoder-only 84.859627 ± 0.580810%. Different domains/classes, model, preprocessing, local epochs, optimizer persistence and calibrated hyperparameters prevent interpreting between-benchmark differences as a single controlled effect.

New 15-run campaign calendar 10388.717 s; process sum 53524.394 s. Per-run timing logs, actual/attempted optimization steps and CUDA memory are archived. No old run is charged as newly executed. All initial/final complete checkpoints, configs, indices, optimizer/scaler/RNG and full logs remain on thanos, with paths/hashes in artifacts/summary.json.

The original full-method mean belongs to a future Table 4 update. Ablations remain a separate analysis. Manuscript and previous experiments unchanged.
