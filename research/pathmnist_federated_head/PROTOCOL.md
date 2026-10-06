# Federated simulated PathMNIST head-only refit

One fixed experiment, seed42, source original round50 SHA256
`fa56da01091debe9031a9acb2e3de0f34ce0f751d2c95d2d0d54c05f15697d1e`.
Centralized reference checkpoint SHA256
`31019b0e1ef8594de1f7da196a6ce4b7e84afdc58fb4ba3d42fcf14b7fd9d4b8`,
accuracy75.33147632311977% (54088/71800 predictions).

## Fixed mathematics and optimizer

Only the585 parameters of the shared64→9 linear head are trainable. Each image
uses its own original private encoder. Frozen Float32 eval representation:
`ResNet20V2_body(x + D(E_i(x)))`; UNetSmallAE dz16, original E/D/body/BN,
no recalibration, attenuation or tuning. All89996 original training examples
are used once in client-private feature caches, preserving index order.

Client i computes full-batch local mean CE L_i(theta) and its raw head gradient.
The server uses L=(1/10)sum L_i and g=(1/10)sum g_i, **not sample weighting**.
The source classifier terminal head is the common initialization. The same
PyTorch L-BFGS settings are retained: LR1, max_iter100, implicit max_eval125,
history20, strong-Wolfe, tolerance_grad1e-7, tolerance_change1e-10, penalty0.
All calculations Float32, including uniform aggregation in client-ID order.
Each closure/line-search evaluation requires an all-client gradient aggregation.
No local SGD steps or representation optimization, validation, sweep, early
stopping on test, choice of checkpoint, or repeated candidates.

## Data locality and roles

Ten spawned processes keep images/labels/features in their own address spaces.
Only their own training indices are materialized; there is no pooled feature
tensor anywhere in this implementation. Each private cache is saved under
`_local/pathmnist_federated_head/seed-42/clients/client-i/`.
The optimizer server has only585 parameters and IPC Connections; requests carry
a trial head, responses only one local loss and its585-dimensional gradient.
Ready/done/error messages are control statuses. The server never reads the
client feature/label files.

An offline report assembler reads scalar error/count metadata written by the
clients, not logits/features/labels. Test auditing begins only after the final
optimized head and complete checkpoint are frozen. Each pipeline uses the
same7180 official test images and native BN. The metric is the uniform mean of
all10 pipeline accuracies (71800 predictions on7180 distinct images).
Client-side comparisons cover train/test logits against the stored centralized
head and reproduce both original and centralized archived correct counts.

The original full checkpoint is preinstalled for this simulated exercise; its
unchanged private/shared states and all old optimizer/scaler/RNG/indices are
preserved in the complete final checkpoint. New L-BFGS state and phase RNGs are
separate. This is address-space/data-flow separation on one host with ordinary
pipes and files, not a cryptographic privacy or real-network deployment claim.

## Comparison without centralizing private data

The old centralized optimizer stores `prev_flat_grad`, `prev_loss`, final
direction d and accepted step t. Its gradient is **before** the last accepted
update, not the final gradient. Reconstruct theta_pre=theta_final−t*d and
aggregate clients' local loss/gradient at that point. Validate the reference
accepted-step history first. Reconstruction entails Float32 roundoff.
Compare the initial CE to the archived initial CE and CE at the stored central
final head to the archived final CE. Also compare gradients/CE at the two
different terminal heads; those measurements include optimization-trajectory
differences, not solely reduction roundoff. No centralized refit is rerun.

Tolerances fixed before execution in `config.json`:

| Comparison | atol | rtol |
|---|---:|---:|
| Same-point/initial CE | 2e-6 | 2e-6 |
| Stored central pre-last gradient, each entry | 1e-5 | 1e-4 |
| CE at the two optimized terminal heads | 2e-4 | 1e-4 |
| Gradients at the two optimized terminal heads | 2e-4 | 1e-2 |
| Train/test logits at the terminal heads | 0.02 | 0.002 |

Absolute accuracy tolerance0.05 percentage points; test prediction disagreement
fraction≤0.001. These are declared empirical equivalence checks, not a theorem
that truncated Float32 L-BFGS trajectories must coincide. A failure will be
reported as such, without changing tolerances or reopening training after test.
No SD between seeds is inferred from this one-seed diagnostic.

## Communication accounting

Request: one opcode byte plus585 Float32 parameters =2341 bytes. Response:
one opcode plus one Float32 loss and585 Float32 gradients =2345 bytes.
One ten-client aggregation:46860 application bytes;46940 including the4-byte
Unix Pipe length prefix on each of20 messages. Count every actual closure,
including strong-Wolfe trials; iterations are not communication rounds.
Three fixed post-optimization loss/gradient audits are counted separately.
Final inference-audit broadcast carries two heads; only done status comes back.
Ready/done statuses, protocol opcodes and Pipe framing are accounted separately.

Preinstalled dataset/body/checkpoints, operating-system process startup,
shared-file audit/checkpoint writes and network/TCP/TLS overhead are excluded.
Report both payload and payload+Pipe-framing totals; these are measured
simulated head-protocol costs, not end-to-end original FusedSpaceFed training
communication. All complete logs/data/cache/checkpoints remain under `_local/`.
Only code/config/numeric metadata/results/report are committed. Manuscript and
the original campaign are unchanged.

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -u \
  research/pathmnist_federated_head/run.py \
  --config research/pathmnist_federated_head/config.json \
  --output _local/pathmnist_federated_head/seed-42 --device cuda:1
```
