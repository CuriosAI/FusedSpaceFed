# PathMNIST original settings: five-seed FP32 campaign and mechanism diagnostics

Full FusedSpaceFed: **31.430362 ± 4.323117%** (five fresh seeds 42–46; sample SD, ddof1).

## Scope and exact pairing

All five full runs and all fifteen ablations start at round 0. Historical initial C/D/private-E weights, empty local optimizers and client shuffle-generator states are restored exactly; frozen per-seed partitions are unchanged. Precision is FP32 throughout training, replay, terminal inference and gradient computation: AMP/autocast, GradScaler and matmul/cuDNN TF32 are disabled. Float64 is used only for statistical/Gram reductions. Architecture, data, batch 128, 50 rounds, full participation, native BN, persistent SGD0.01/Adam0.001, warm-up 1 and CE 3 remain fixed. No tuning, baseline, head refit, BN recalibration, checkpoint selection or manuscript edit.

Seed 42 previously used two workers. This campaign uses one process per run; exact model and per-client data-order initialization is retained, while its process RNG mapping is explicitly recorded. Different precision and device scheduling need not reproduce historical trajectories bitwise. The preserved FP16 seed 43 overflow did not justify changing any learning rate or adding clipping.

Each terminal test is performed once at round 50: 7180 official images through each of ten private encoder pipelines; uniform client mean, 71800 predictions on 7180 distinct images. Correct/total counts independently reconstruct all scores. Seed affects both model and frozen pathological partition; every intervention is paired within seed. SD over five seeds is descriptive, not a confidence interval.

## Full method and separate component ablations

| Method | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± SD (%) | Paired delta ± SD (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 26.516713 | 33.013928 | 37.855153 | 30.892758 | 28.873259 | 31.430362 ± 4.323117 | reference |
| no-warmup | 18.303621 | 13.686630 | 12.991643 | 14.770195 | 18.408078 | 15.632033 ± 2.566263 | -15.798329 ± 6.721505 |
| shared-encoder | 26.908078 | 29.791086 | 18.969359 | 29.261838 | 31.824513 | 27.350975 ± 5.001643 | -4.079387 ± 8.593251 |
| decoder-only | 13.774373 | 9.272981 | 15.091922 | 14.548747 | 15.557103 | 13.649025 ± 2.534644 | -17.781337 ± 5.189688 |

For each intervention, paired signs and all five differences are retained:

- no-warmup: below full in 5/5 pairs, above in 0/5. No selection follows from these test results.
- shared-encoder: below full in 3/5 pairs, above in 2/5. No selection follows from these test results.
- decoder-only: below full in 5/5 pairs, above in 0/5. No selection follows from these test results.

The FP32 full-method mean is 19.509638 percentage points below the historical Table 4 value of 50.94%; these runs do not recover that result. The original five Table 4 training artifacts were unavailable: the initial states reused here are from the preceding fixed-setting campaign, not recovered original paper runs. This is new evidence for a future update, separate from the component interventions; the manuscript is unchanged. No-warmup reduces compute and changes Adam history/shuffle progression; it is not FLOP-matched. Shared-encoder averages only encoder model weights, retaining local optimizer moments. Decoder-only removes the raw-input additive branch during CE and test. No ablation hyperparameter calibration was performed.

## Decoder and warm-up: training-only replay

Initial/final complete checkpoints were copied independently per client and replayed for one warm-up epoch followed by three CE epochs on the full local training set. Restored persistent moments and shuffle RNG are retained. Fixed 640 training examples/client are measured before/after each phase. This is a hypothetical extra local update, not a recovered historical within-round trajectory. Warm-up is audited to leave all shared C/D parameters and BN buffers unchanged. Complete before/warm/CE snapshots and twenty predetermined seed 42 visual grids are preserved.

| Anchor | Stage | Reconstruction MSE ± SD | Decoder/input RMS ± SD | Raw max abs ± SD | Fused training CE ± SD |
|---|---|---:|---:|---:|---:|
| initialization | before | 0.419526 ± 0.027082 | 0.238129 ± 0.025604 | 0.239318 ± 0.030689 | 2.199164 ± 0.007065 |
| initialization | after_warmup | 0.188498 ± 0.107534 | 0.706354 ± 0.284328 | 1.194504 ± 0.501225 | 2.199669 ± 0.007364 |
| initialization | after_classification | 0.454676 ± 0.054194 | 0.690554 ± 0.248159 | 1.438579 ± 0.614964 | 0.851539 ± 0.206635 |
| final-round50 | before | 0.610285 ± 0.047908 | 0.230641 ± 0.031032 | 0.345452 ± 0.086122 | 1.838475 ± 0.289404 |
| final-round50 | after_warmup | 0.607762 ± 0.046656 | 0.231070 ± 0.033444 | 0.350222 ± 0.097901 | 1.846959 ± 0.316049 |
| final-round50 | after_classification | 0.614727 ± 0.039365 | 0.241437 ± 0.037610 | 0.387924 ± 0.149168 | 0.553406 ± 0.344935 |

Summaries first average clients uniformly within seed, then average five seeds. Client IDs have different class pairs across seeds. Decoder output is linear, not restricted to[0,1]. Raw amplitudes, out-of-range fractions and input/decoder cosine are retained; only visual display clips to[0,1]. Reduced reconstruction error alone does not establish better classification.

At the final anchor the mean relative MSE is 1.305725, where 1 is the error of returning zeros (an analytic reference, not a trained baseline). The small output RMS and representative grids do not support calling the decoder a faithful image reconstruction. The encoder is exactly unchanged after both replay phases in 3/50 client copies: [{'seed': 42, 'client_id': 0}, {'seed': 42, 'client_id': 1}, {'seed': 44, 'client_id': 3}]. This is a measured absence of parameter updates, not proof that all encoder gradients are zero; phase3 records those gradients separately. All 50 initial encoders did update. BN-buffer distance includes num_batches_tracked, so a large aggregate buffer norm cannot be interpreted as a large normalization-statistic drift.

## Paired gradients at identical model states

Five fixed training batches of 128/client, original x versus x+D(E_i(x)), initial/final anchors. Native-eval BN is the inference-aligned primary probe; batch-stateless BN is a separate batch-conditioned sensitivity probe. All weights, buffers and RNG are audited unchanged. No optimizer step is performed. Client gradients are averaged over batches before computing population dispersion with divisor 10; Γf=Γo+B+Φ is checked to relative tolerance 1e-10. Raw vectors are computed in Float32; CPU means/Gram algebra in Float64.

| Anchor | BN mode | Γf/Γo ± SD | Original normalized Γ | Fused normalized Γ | Original pair cosine | Fused pair cosine |
|---|---|---:|---:|---:|---:|---:|
| initialization | eval | 1.092205 ± 0.033920 | 0.989786 | 0.989559 | -0.100951 | -0.101489 |
| initialization | batch-stateless | 1.001866 ± 0.013251 | 0.917749 | 0.917668 | -0.028905 | -0.029099 |
| final-round50 | eval | 0.940635 ± 1.004737 | 0.700936 | 0.822183 | 0.221775 | 0.093693 |
| final-round50 | batch-stateless | 0.840077 ± 0.160942 | 0.888618 | 0.894538 | -0.017369 | -0.028022 |

A raw Γ decrease can reflect smaller gradients rather than angular alignment. Inspect normalized dispersion, cosines, B, Φ and decoder dispersion together. The trained classifier and its native BN reflect fused inputs; the raw-input branch is a counterfactual at identical weights, not a separately trained classifier. Input-distribution shift or loss calibration can also change gradient scale. Batch-stateless BN is a distinct batch-conditioned objective. Full per-client/per-batch norms, Gram matrices and decomposition identities are archived in phase3/. Two anchors and finite training probes cannot establish a theorem or explain every intermediate round.

At the final anchor raw dispersion falls in 4/5 native-eval seeds, but seed 42 has Γf/Γo 2.697930. Native normalized dispersion increases 0.700936→0.822183 and pairwise cosine falls 0.221775→0.093693. Batch-stateless raw Γ falls in 5/5 seeds, while normalized dispersion also increases 0.888618→0.894538. Thus these probes do not establish improved angular alignment or universal dispersion reduction. At initialization native Γ increases in all 5 seeds.

Encoder fused-CE gradients are exactly zero in all five fixed batches for [{'seed': 42, 'client_id': 0}, {'seed': 42, 'client_id': 1}, {'seed': 44, 'client_id': 3}] under native eval and for [{'seed': 42, 'client_id': 0}, {'seed': 42, 'client_id': 1}, {'seed': 44, 'client_id': 3}] under batch-stateless BN. These are the same three client cases with no encoder parameter change in either replay phase. The observation concerns the measured training probes at the saved state; it does not prove zero gradients on every possible input.

## Comparison with existing Digits diagnostics

Digits full 85.862359±0.846930%, no-warmup 85.714598±0.789950%, shared-encoder 86.178544±0.468069%, decoder-only 84.859627±0.580810%. Final native Γf/Γo 0.791656±0.253408; batch-stateless 0.148367±0.140864. Native normalized dispersion 0.794160→0.799588 and cosine 0.090559→0.043966 demonstrate that smaller raw gradients need not mean improved angular alignment. PathMNIST differs in classifier, latent size 16 versus 64, input scale[0,1] versus[-1,1], label-skew versus domain shift, ten versus five clients, three versus one CE epochs, persistent versus reset optimizers, and fixed original versus calibrated settings. Cross-benchmark comparisons are descriptive. Every PathMNIST result above is a five-seed statistic, separate from older single-seed investigations.

## Measured cost and preservation

| Phase | Campaign elapsed (s) | Sum process elapsed (s) | Maximum process allocated / reserved (MiB) |
|---|---:|---:|---:|---:|
| originals | 3916.973 | 17234.926 | 1422.817 / 1708.000 |
| phase1 | 173.207 | 747.503 | 1469.474 / 1736.000 |
| phase2 | 10388.717 | 53524.394 | 1422.817 / 1710.000 |
| phase3 | 23.185 | 93.378 | 1130.381 / 1164.000 |

Successful first launch→last diagnostic: 14803.821s including intervening verification/archiving/commits. Sum phase campaign elapsed: 14502.081s; sum process elapsed: 71600.201s. Concurrent process times are not exclusive GPU-hour charges. Per-seed training/evaluation/session durations, actual optimizer counts, peak allocated/reserved CUDA bytes and RSS are retained in the corresponding archives.

A Python standard-library name collision caused a failed import-only launch, before any training. It was fixed by renaming the precision module and adding three direct-CLI regression tests. The first decoder replay also stopped on relative-path metadata serialization after updating only client copies. Its output path was canonicalized; all five diagnostics restarted from immutable anchors. Both failed receipts, logs and partial diagnostic checkpoints remain preserved, with measured costs separately recorded in cost_summary.json. Neither was a numerical failure or altered the full trained models. Phase archives require every declared process to complete.

Hash verification confirms all 872 pre-existing tracked files and 35 files of the previous PathMNIST runs unchanged. Frozen partition/probe/source/checkpoint SHA256 manifests are retained. Full checkpoints (initial/latest/final for 20 training runs, plus 100 decoder replay snapshots) remain on thanos in `_local/pathmnist_five_seed/fp32/`; scaler is explicitly None, not missing. Dataset/checkpoints/full logs are not published.

## Reproduction and commits

Commands/configurations: README.md, plan.json, initial_sources.json, full_queue.json and phase1–3/queue.json. Successful commands, physical GPU, exit codes and process durations: each artifacts/campaign.json. Numeric result JSON is gzip-compressed losslessly; timings.jsonl are uncompressed. Source version per process is in each archived result/checkpoint. Tests and exact output are in verification_tests.log, verification_cli_tests.log and verification_diagnostic_tests.log.

Commits for this task before the final report commit:

- 5f6baa76ab4b98b3c3d9aa58dfede0c08038997b experiment: add immutable FP32 PathMNIST five-seed campaign
- 519810435b80b460091277419a24fc4afcda5484 fix: avoid standard-library module collision in FP32 entry points
- 52ed56e8a359106d7a017df8076ef2d7691d772c results: archive five original-setting FP32 PathMNIST runs
- 47dbfa5357f39ffe561d5717845afb65bb35b097 fix: canonicalize FP32 diagnostic checkpoint output paths
- ce36895214762c012811f4902dc80400ea47fcd2 analysis: add five-seed FP32 PathMNIST decoder diagnostics
- 50c7f9a2bd2c5be6b2bd16fdf9a557fd3c6f2ff5 results: add paired five-seed FP32 PathMNIST component ablations
- f486b19aba9c35d074193e2d412ed93b4bf86b2d analysis: add five-seed FP32 PathMNIST gradient diagnostics

Final full repository test suite: 372 passed in 92.55s (0:01:32). Exact output: final_verification_tests.log.

The commit containing this assembled report is identified by Git history; no self-referential commit hash is embedded. All substantive steps are pushed to main. The manuscript remains unchanged.
