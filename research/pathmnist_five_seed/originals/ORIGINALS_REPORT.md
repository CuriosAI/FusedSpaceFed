# Original PathMNIST campaign — numerical block

**The requested task is incomplete.** Four original seeds are complete (42 reused, 44–46 new); seed43 failed reproducibly during round20. No mean or sample SD is reported as if all five runs were valid. The three mechanism phases are implemented/tested but have not been started, respecting the requested order.

## Outcomes

| Seed | Status | Completed rounds | Accuracy (%) | Correct / 71800 | Training (s) |
|---:|---|---:|---:|---:|---:|
| 42 | completed | 50 | 39.619777159 | 28447 | 948.916 |
| 43 | failed FP16 forward | 19 | unavailable | unavailable | 710.798 to last valid checkpoint |
| 44 | completed | 50 | 19.944289694 | 14320 | 1860.742 |
| 45 | completed | 50 | 25.759052925 | 18495 | 1811.233 |
| 46 | completed | 50 | 26.973537604 | 19367 | 1833.684 |

All completed seeds have exactly 50 rounds and one terminal test. Metric: uniform mean over ten private-encoder pipelines on all 7180 official test images; 71800 predictions are repeated use of those same 7180 images. Numerators/denominators reproduce every accuracy. Seed43 has no test evaluation, no final checkpoint and no invented accuracy. Its finite checkpoint at round19 contains all ten encoders, shared states, optimizer/scaler/RNG and indices.

## Reproducible numerical diagnosis

The first attempt and an exact `--resume` from round19 both failed with non-finite phase loss. A training-only replay on copies, using the original FusedSpaceFedClient (not the new subclass), located the first non-finite activation: client8, warm-up epoch1, batch38, autoencoder.bott.2. Exactly one positive infinity appears in Float16, shape [128,16,8,8]; all inputs to that layer are finite. The same convolution with identical weights and inputs converted to Float32 produces finite output with max absolute value 65608.0234375, exceeding the Float16 representable maximum 65504. This supports an activation-range overflow diagnosis; it does not prove that a full Float32 training trajectory would remain stable or improve test accuracy.

No learning rate, clipping, optimizer, method component, partition, seed or BN setting has been changed. The Float32 check is a single forward operation, without backward/optimizer step, tuning or test access. Failed attempts, logs and forensic tensors remain private. `replay_failure.py` and `check_overflow.py` preserve the numerical harness with main/overwrite guards; exact executed script versions and SHA256 are retained privately. Their output filenames belong to this preserved attempt, so reproduce into distinct paths rather than overwrite it.

A correction of numerical precision would change the recorded original implementation; it requires resolving the latest explicit instruction to keep the original documented protocol. The user was asked during execution whether to retain/document the original failure or authorize a numerical correction. No answer is treated as approval. Original successful trajectories and the seed42 checkpoint remain untouched.

## Protocol and paired diagnostics readiness

Frozen original settings: 50 rounds, 10 clients with two pathological classes, all clients each round, batch128, encoder-only warm-up1, joint CE3, additive x+D(E_i(x)), UNetSmallAE dz16, ResNet20V2 RGB9, SGD0.01 without momentum and Adam0.001, persistent local optimizers/scalers, original FP16 AMP, native BN, uniform shared C/D aggregation. No tuning, BN recalibration, head refit or final adaptation. Each seed controls its frozen partition as in the original recipe; each future ablation uses that seed’s full initial checkpoint and indices.

The decoder/warm-up replay and identical-state gradient probes are configured for fixed 640 training images/client, five batches128. Tools to record complete replay states, reconstruction/amplitude statistics and preselected visual grids are implemented; those diagnostic outputs have not been generated. The three ablations (no-warmup/shared-encoder/decoder-only) retain optimizer persistence and change one component. Gradient code reuses the audited Digits BN controls and exact Gamma/B/Phi Gram decomposition, with native-eval and batch-stateless modes. These are planned instruments, not claimed completed analyses. No five-seed diagnostic/ablation results exist from this task.

## Cost, provenance and checks

New campaign calendar 1871.676 s; sum of original-attempt process times 6288.080 s; retry process time 42.015 s. Diagnostic replay 70.244 s. Seed42's original training cost is separate. Failed-round work is included in process receipts, not in the saved checkpoint training duration. No process belonging to another job was interrupted.

Per-run CUDA allocated/reserved peaks, local optimizer step counts, session durations and full source/config/partition identities are retained in numeric artifacts. Source42’s actual optimizer-step counters were not available in its original schema and are not invented. New source manifests and exact private partition paths/hashes: ../data_identity.json. Initial/final/private latest checkpoint hashes: artifacts/summary.json. Executed commands, PIDs, GPUs and exit codes: artifacts/campaign.json and retry_campaign.json.

Verification: the versioned suite passes, including original client update parity, persistent optimizer state, exact checkpoint resume, paired partition coverage/disjointness, component gradients, native BN/RNG immutability, gradient algebra and rejection of four-seed summaries. Exact count/time and output: ../verification_tests.log. Reused seed42 hashes remain exactly those frozen in ../reused_seed42.json. Every completed checkpoint is finite and synchronized, includes all ten private states and exact indices, and matches the stored run identity. Numeric compression is lossless.

The historical 50.94% in Table4 is unchanged in the manuscript. A future replacement mean ± SD is unavailable until all five seeds complete under a declared common protocol. This failure is not hidden by reporting a survivor-only average. There are no new baseline results, ablation results or claims comparing this incomplete campaign with Digits. Dataset, complete checkpoints/logs and forensic input tensors stay under _local on thanos; only code, numeric metadata/results and report are published.
