# Validation-only FEMNIST calibration

Six prespecified candidates; training-derived validation only. The old stabilized
configuration is retained as the reference. No definitive test was accessed by
this search and no baseline was retrained.

## Frozen choice

Selected candidate: `03-classifier-2e-2`.
Classifier LR 0.02; autoencoder LR 0.0003; L2 gradient clipping 1.0.
All other architecture, data and experimental protocol fields remain unchanged.
Receipt frozen at 2026-10-05T00:16:20Z, before the new definitive runs.

Selection uses the mean of two tuning-seed means (141, 142), each on rounds 191–200.
The primary metric is sample-weighted accuracy; client-uniform accuracy is secondary.
All reported values below are validation accuracy percentages.

| Candidate | 80-round screening, seed 141 | Full validation, seed 141 | Full validation, seed 142 | Full two-seed mean |
|---|---:|---:|---:|---:|
| 00-reference | 61.91655204 | 78.25080238 | 61.03392939 | 69.64236589 |
| 01-ae-3e-4 | 67.15038973 | — | — | — |
| 02-ae-1e-4 | 47.74415406 | — | — | — |
| 03-classifier-2e-2 | 72.72581385 | 91.35259055 | 86.18523613 | 88.76891334 |
| 04-classifier-5e-3 | 28.26455754 | — | — | — |
| 05-clip-2 | 74.98395232 | 91.72397983 | 84.75469968 | 88.23933975 |

## Reproduction and contents

The versioned plan and code, numeric attempt snapshots, complete current trial results
and timing records document every attempt. `results.json.gz` is lossless, with a
zero gzip timestamp. `manifest.json` records file hashes; original uncompressed hashes
are also retained. Data, per-example split IDs, checkpoints and complete logs stay local.
The public completed campaign record is explicitly a provenance extract, not a byte
copy of the private campaign file; its required identity fields are unchanged.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python calibrate_femnist_reconstructed.py search \
  --plan configs/femnist_calibration_plan.json \
  --partition _local/femnist_reconstructed/partition \
  --output _local/femnist_reconstructed/calibration-v1 --device cuda:1
```

Resumption used `--resume` on the same output; original identity, settings and checkpoints
were preserved. GPU guards stopped admission of new trials when other work was present.
The initial private queue waited for an idle GPU. After explicit user authorization,
a separate operational wrapper admitted GPU1 with at least 2048 MiB of free memory,
including when other work was present. It replaced only the resource-admission
callable in memory; the five frozen scientific source files and native trial commands
were unchanged. Both queues and their separate costs are recorded; no external
process was interrupted. Complete wrapper provenance and logs remain local.

## Verification-only source transition

After calibration completed, the saved-result verifier was corrected to compare each
evaluation with client participation accumulated through that round, rather than the
final round-200 vector. Synthetic regression tests cover this correction. The selection
receipt and selected configuration retain their original bytes. A separate transition
pins the five original/current source hashes and the completed calibration record,
and requires the module AST outside its two verification functions to remain identical.
The four model/data/training source files are byte-identical to calibration. Neither
optimization nor selection is altered by this transition.

## Limits

A small fixed grid, an 80-round screening stage, two tuning seeds and one validation
partition do not establish a global optimum. Early screening can reject late improvers.
The reconstructed NIST benchmark is not an exact replication or a controlled comparison
with published FedRep baselines. Final results use all original training examples,
which differ in number from the fit-only calibration data. Old attempts and results
are preserved separately. No published-baseline uncertainty is inferred.
