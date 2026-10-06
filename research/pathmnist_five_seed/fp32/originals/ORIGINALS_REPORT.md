# Five original PathMNIST runs

FusedSpaceFed with original settings, FP32: **31.430362 ± 4.323117%** (sample SD, five seeds 42–46).

| Seed | Accuracy (%) | Correct / predictions | Training (s) | Session (s) |
|---:|---:|---:|---:|---:|
| 42 | 26.516713092 | 19039 / 71800 | 3895.338 | 3903.856 |
| 43 | 33.013927577 | 23704 / 71800 | 2742.658 | 2746.417 |
| 44 | 37.855153203 | 27180 / 71800 | 3901.045 | 3906.588 |
| 45 | 30.892757660 | 22181 / 71800 | 2737.331 | 2742.918 |
| 46 | 28.873259053 | 20731 / 71800 | 3903.837 | 3908.199 |

All seeds 42–46 are rerun from their exact immutable historical round-zero states in FP32; none reuses a trained checkpoint. Three run slots on GPU 1 and two on GPU 0. No tuning, head refit, BN recalibration, adaptation or checkpoint selection. Each seed controls its own frozen pathological partition, as in the original runner; the three ablations will be paired within each seed.

The primary accuracy is the uniform mean of ten private-encoder pipelines; each predicts the same official test set (7180 distinct test images). All raw numerators/denominators reconstruct the reported values. Exactly 50 rounds and one terminal test per run. Full checkpoints contain all private encoders, shared classifier/decoder, BN buffers, local persistent optimizers, RNGs and indices; scaler=None explicitly. AMP and TF32 are disabled everywhere.

Fresh five-run FP32 campaign calendar: 3916.973 s; sum of process wall times: 17234.926 s. Previous FP16 attempts and their costs remain separately archived.

The historical Table 4 contains 50.94%. These five audited runs are the evidence for a future paper update; the manuscript is intentionally unchanged. They do not establish statistical equivalence to the historical number. Between-seed SD includes partition and initialization variation; five seeds are not a confidence interval.

The first launch exited during Python imports because a new module named profile.py shadowed the standard library. The module was renamed before any training update; three direct-CLI regression tests were added. The failed receipt and logs remain separately preserved (failed_launch_campaign.json and logs/import-failure/); successful run times do not include that attempt.

Raw JSON is gzip-compressed without loss; timing logs, source hashes, configuration identity and private checkpoint hashes are in artifacts/. Configuration is ../plan.json. Full data, checkpoints and complete logs remain on thanos. Existing experiments and manuscript are unchanged.
