# Minimal capacity and compute control

One existing scenario: balanced FedBN Digits, five domains, frozen part0, 743
training/client, 300 rounds, same preprocessing, all clients and uniform
aggregation, all BN shared, fixed final test and no adaptation. Chosen for low
cost and reuse of completed runs, not a favorable accuracy/domain. The preceding
five Fused tests were already observed: this is a retrospective control. No
claim of an unseen-test preregistration or coverage of label-skew settings.

FusedSpaceFed is unchanged: DigitCNN, private persistent encoder, shared decoder,
additive fusion, one E-only MSE warm-up and one E/D/C CE epoch. FedAvg widens only
fc1 from 2048 to 2065 (fc2 input and BN4 follow). 14,334,589 FedAvg parameters
versus 14,336,285 E+D+C parameters/client, difference -0.01183%. Global stored
parameter counts differ from active/client counts; all five private E states
are counted separately in the report. Common-shape CNN initialization uses the
same initial draws. Full models cannot have identical weights because shapes
and personalization differ. Run seeds and data/evaluation are paired.

Training cost convention: Torch 2.5.1 FlopCounterMode on executed forward and
backward, FMA=2 for convolution/transposed convolution/matrix products, including
input-gradient propagation through the frozen decoder during warm-up. Add
semantic scalar costs per step: 5P for SGD+L2 norm/clipping, 16P for Adam+norm/
clipping (square/multiply/add/divide/sqrt convention). Norm accumulation is
Float64 but one scalar operation is still one FLOP; this is not hardware cost.
BN, ReLU, pooling, losses, finite checks, aggregation, copies, kernel overhead
are excluded from the matching metric, disclosed rather than presented as
exhaustive hardware instructions. GPU spot checks confirm dense operation
signatures. Report dense and combined counted totals, actual steps/exposures,
wall times and allocator memory; no claim of equal time/energy/communication.

For each client/round Fused's full warm-up+CE counted budget is the target.
FedAvg first sees all training examples once, then extra shuffled minibatches.
Integer arithmetic chooses batches of 2..32 (BN forbids size one) and carries
the unused remainder across rounds. Cumulative deficit is less than the cost of
a two-example update, tested below 0.001% after 300 rounds. Counts are derived
from actual executed batch sizes, not just nominal epochs. This modified FedAvg
control is not the standard one-epoch published FedAvg baseline.

Training-only calibration: source-row ID SHA order within labels; proportional
largest-remainder class quotas, 149 validation and 594 fit/client, frozen seed
20261005 and manifest. Same split, seed 142, SGD LR [.005,.01,.02], clipping
[.5,1,2], and Fused AE LR [.0001,.0003,.001]. Following the user's extension,
plan v2 uses nine balanced L9 rows c=(a+b+2)%3 over these three factors for
Fused, including the original [.01,.0003,1] first. Projected classifierLR/clip
pairs cover all nine combinations once: FedAvg uses the same nine pairs.
The initial three-candidate plan was never run and is retained as history.
60 rounds, same training-FLOP target/round for each method. Only final round 60
validation uniform-domain accuracy selects settings; ties choose lower classifier
LR, then lower clipping, then lower AE LR. Other AE settings remain fixed.
Equal effort means nine candidates each, same seed, rounds and compute convention.
This is a balanced screening design, not exhaustive 27-combination Fused tuning;
interactions/AE effects cannot be isolated. No second search stage, test tuning,
intermediate checkpoint selection, early stop or seed selection.

Freeze both selected configurations before final tests. Exactly three paired
seeds 42,43,44, full training 743/client, 300 rounds and one test at round 300.
Reuse preceding Fused results only if all training settings/source bytes and
data match exactly. Otherwise run all three Fused seeds fresh. Always retain
previous attempts. Report every paired seed, mean/sample SD ddof=1 and paired
difference. Three seeds are a small descriptive control, not proof of universal
causality. Matching capacity+compute jointly does not separate their individual
effects and does not match model inductive bias/private parameters/BN histories.

All files are isolated here. Images, checkpoints and complete process logs stay
in `_local/capacity_compute_control/`; existing experiments/manuscript unchanged.
