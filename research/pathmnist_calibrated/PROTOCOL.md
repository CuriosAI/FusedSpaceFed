# PathMNIST pathological calibration and final campaign

This task suspends the requested component ablations and decoder/gradient
diagnostics. Following the user's budget reduction it completes the existing
24-candidate screening, freezes the best single-seed round20 validation score,
and executes ONE fresh final seed42 run for50 rounds. Confirmations, extensions
and the five-seed campaign described in the original frozen plan below are
cancelled, not executed. See BUDGET_REDUCTION.md and budget_reduction.json;
selection ties prefer native BN then original candidate order. The frozen
scientific plan/code remain unchanged during screening. The manuscript and
old experiments and the fresh paper-settings
seed-42 run remain unchanged. No packages or environments are installed.

## Data, objective and information boundary

Reuse the verified PathMNIST-64 source and exact `Resize(32,32); ToTensor`
float32 cache from `research/pathmnist_pathological/`. Keep the original
two-class, ten-client partition (data seed 42). All 89,996 official training
examples retain their original unique indices. A fixed stratified 10%
holdout within each client/class uses seed 20261006 and integer floor:
**81,002 fit images and 8,994 validation images**. Both fit/validation retain
two classes/client, are globally disjoint and cover the complete training
split. Exact indices and six hundred forty BN-calibration fit indices per
client are frozen losslessly in `partition.json.gz`.

Validation mirrors the paper's global test metric: every private encoder,
with the same shared decoder/classifier, predicts all pooled holdout images.
The primary score is the uniform mean of ten pipeline accuracies. It is
neither local two-class accuracy nor the best encoder's accuracy. All ten
pipeline correct/total counts are retained and reconstruct the score.

The old full-training seed-42 test accuracy (39.619777%) was known before this
task. No new candidate test score is accessed before the final freeze. Thus
this is a validation-selected exploratory improvement on a previously seen
benchmark, not a pristine blind test of the dataset. The original official
validation split is not used. New final runs retrain on all 89,996 training
images, including the former holdout, after selection. Test is evaluated once
at the selected terminal round; no best seed/checkpoint or post-test tuning.

## Original search plan and superseded budget

`search_plan.json` predeclares 24 candidates. It includes the exact GPU
paper-reference updates (FP16 AMP, no clipping, SGD LR .01, Adam LR .001,
one warm-up epoch, three CE epochs, persistent optimizer states), a matched
FP32 reference, a classifier-LR {0.003,0.01,0.03,0.1} × AE-LR
{0.0001,0.0003,0.001} BF16/clipping-5 grid, clipping 1/10 controls, a
BF16 unclipped control, a matched FP32 clipped control, cosine schedules,
one-local-CE-epoch alternatives, a higher .3 LR with one CE epoch and a
separate lower warm-up LR. All candidates keep at least one warm-up epoch.

1. **Active:** fresh seed-142 screening, 20 rounds. Constant or declared 100-round
   cosine schedule; stopping at round 20 does not compress the schedule.
2. **Cancelled:** top three distinct training candidates by their best declared inference
   mode, plus the paper reference if absent. Fresh confirmations on seeds
   142/143, 50 rounds.
3. **Cancelled:** extend the two best distinct confirmed training candidates to 100 rounds
   by exact resume of both 50-round checkpoints. Hyperparameters and cosine
   horizon stay fixed, as do private states, optimizer moments, loader
   generators and RNG. Preserve the 50-round results/checkpoints.
4. **Cancelled:** freeze the highest two-seed mean terminal validation score among all
   50-round confirmations and the 100-round extensions. Ties prefer fewer
   rounds, native BN, then candidate ID. Five fresh full-training runs on
   model seeds 41–45 use exactly one selected configuration and inference
   mode. The data partition remains seed 42 for all runs; seed SD measures
   initialization/optimization variation, not partition variation.

Active final rule: highest validation score among the finite, completed
original profiles at 20 rounds on seed 142, comparing native and
train-recalibrated BN; all 24 declared profiles are attempted.
no confirmations or extension. Freeze ONE configuration, retain its schedule
horizon100 if applicable, and run seed42 from zero for50 rounds on full
training. The result is one seed, with no SD across seeds. Use the existing
single-GPU runner; no new parallelism implementation.

No numerical-failure candidate is silently repaired or substituted. Its
logs and initial/last complete checkpoints remain available; other declared
candidates continue. Scores are compared at declared terminal rounds, not
picked from per-round validation maxima. Finite loss/model/optimizer states
are checked at each checkpoint. The final selection is saved before
launching the definitive run. Source snapshots and hashes are
retained with every stage. No shortlist or confirmation stage is executed
under the reduced budget; selection is frozen directly after screening.

## Method and declared optimization differences

Architecture stays ResNet20-v2 (nine logits) and UNetSmallAE (`dz=16`),
with private persistent encoders, shared decoder/classifier, additive
`x + D(E_i(x))` fusion, uniform server averaging and full participation.
Encoder warm-up uses MSE with shared components frozen. Joint training uses
CE only and updates all components. No auxiliary loss or encoder sharing is
introduced. SGD remains without momentum/weight decay; Adam remains
betas .9/.999, epsilon 1e-8, weight decay zero. Both persist across rounds.

Differences from the paper are explicit candidate hyperparameters: precision
(native FP16 AMP, BF16 without scaling or FP32), optional global norm clipping
separately for autoencoder and classifier, LRs, optional separate warm-up
LR, cosine schedule with 0.1 minimum multiplier, one versus three local CE
epochs. The reduced final run always uses 50 communication rounds; a selected
cosine schedule still retains its original 100-round horizon. Clipping preserves
the two objectives and component ownership. The exact reference routes
through the existing core implementation. Synthetic tests verify exact
FP32 unclipped update/state parity, component gradient flow and persistence.

Two predeclared inference modes are compared only on validation:

- `native`: unchanged, uniformly averaged global BatchNorm buffers.
- `train-recalibrated`: **a variant relative to the paper**. After training,
  reset only the shared classifier's BN running means/variances/counters,
  then re-estimate them by cumulative averaging on a fixed shuffled mixture
  of 640 fit-only images/client (6,400 total, 50 batches of 128). Each
  training image is fused by its corresponding private encoder and final
  shared decoder. No labels, parameter gradients, optimizer steps or test
  images are used. Encoder/decoder parameters and all classifier weights
  are identical. BN buffers remain shared, not private. This variant adds
  training-only forward computation and changes inference normalization;
  it is not an exact reproduction of the paper protocol.

BN calibration is reproducible from the complete native tuning checkpoint,
the fixed indices and `recalibrate_bn`; the final selected checkpoint stores
the actual selected shared BN buffers and all local copies. Both native and
selected final checkpoints are retained. The unselected BN mode is never
tested during the final campaign. The runner currently computes the fixed
training-only BN candidate even when native is selected; that unused forward
cost is included in recorded wall time and does not alter the native state.

## Execution, complete checkpoints and audit

Use the existing `/home/schroeder/miniconda3/envs/general_ml/bin/python`.
`controller.py` runs up to three independent workers/GPU, checks free memory
before launching, saves commands/PIDs/exit codes and never signals unrelated
processes. Settings, partition, scientific sources and queues are frozen.
Measured wall time, summed worker wall time, step/image counts and allocator
peaks are reported separately; summed worker wall time under concurrency is
not physical GPU compute time.

Private storage: `_local/pathmnist_calibrated/`. Every run saves complete
`initial.pt`, atomically replaced `latest.pt` after each complete round,
terminal `round-NNN.pt`, numeric results, progress and timing records.
Final runs also save `native-final.pt` and `final.pt` before the only test.
Files contain shared classifier/decoder, ten private encoders, all local
model dictionaries and BN buffers, both optimizer dictionaries/client,
AMP scaler state when applicable, loader generators, CPU/Python/NumPy/CUDA
RNG, exact data indices, settings, source/version hashes and history.
BF16/FP32 need no GradScaler; SGD has an empty momentum state by design.
CUDA worker `cuda:0` is mapped to its physical GPU via `CUDA_VISIBLE_DEVICES`.

The ordinary CLI refuses output overwrite; explicit `--resume` checks
scientific identity and restores persistent states. Resume may extend the
declared execution budget only on training-derived validation; the final
frozen identity cannot change. Previous run and calibration attempts stay
on thanos. Numeric archives, manifests, configuration, selection and reports
are versioned in a calibration/freeze commit followed by a separate final-results
commit. Images, complete checkpoints, local reviews and private inputs are
excluded. Diagnostics and ablations remain suspended at the final report.
