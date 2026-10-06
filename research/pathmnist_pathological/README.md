# PathMNIST pathological diagnostic run

One fresh FusedSpaceFed run, seed 42, with the fixed medical settings in
`paper/aistats_2027.tex` (experimental evaluation and `tab:pathological`).
This produces new checkpoints; it does not recover the original checkpoints
or partition indices underlying the paper's five-run mean of 50.94%.
See `PATHMNIST_REPORT.md` for the verified outcome.

## Fixed protocol

`config.json` fixes 10 clients, exactly two assigned classes per client,
all clients in each of 50 rounds, one encoder-only reconstruction warm-up
epoch, three classification epochs, batch 128, ResNet20-v2, UNetSmallAE with
`dz=16`, SGD classifier LR 0.01 and Adam autoencoder LR 0.001. Decoder and
classifier are aggregated uniformly; private encoders and each client's
optimizer states persist. Optimizer defaults remain those of the checked-in
client (SGD momentum/weight decay zero; Adam betas 0.9/0.999, epsilon 1e-8,
weight decay zero). There is no clipping, schedule, adaptation, early stopping,
checkpoint selection or tuning. CUDA AMP follows the existing MedMNIST
client; precision is an implementation detail not specified in the paper.

The official PathMNIST-64 training split is resized using torchvision's PIL
`Resize((32,32))` followed by `ToTensor()`. No validation examples are added
to training. The fixed seed is passed to the existing `pathological_partition`.
Exact official-array row indices are retained, with complete coverage and
no repeated training rows. The official test split is evaluated only at the
end, using each private encoder and the final shared classifier/decoder.
The primary metric is the uniform average of the ten client-conditioned
accuracies on the entire official test split, as specified in the paper.
This is not an evaluation on ten disjoint local test sets.

## Parallel execution and identity

Two persistent worker processes perform independent local updates on the
two physical GPUs. Clients are assigned once by a deterministic greedy
balance of their number of mini-batches. The server waits for both workers
and averages shared states in increasing client-ID order. Floating BatchNorm
buffers are averaged uniformly; integer counters follow the existing core
rule, which selects client 0 for equal weights. Parallelism does not change
participation or the number/order of each client's local updates.

A preprocessed float32 resident cache replaces repeated PIL transformations
and per-image CPU-to-GPU copies. Unit tests compare native preprocessing,
DataLoader ordering and generator states. There is no augmentation, dropout
or other stochastic operation in the model. Each client's loader generator,
scaler and optimizer remain independent. GPU workers have separate recorded
Python, NumPy, CPU Torch and CUDA RNG states. GPU `cuda:0` inside a worker is
mapped to its recorded physical GPU using `CUDA_VISIBLE_DEVICES`.

Each artifact records the base Git commit and SHA256 of the exact core,
original MedMNIST runner, new runner and configuration. The new runner is
uncommitted at execution time and identified by these hashes, then archived
with this dedicated results commit. The existing scientific code and paper
are unchanged.

## Commands and private storage

From `/mnt/data/codex/FusedSpaceFed`, using the existing `general_ml` Python:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_pathological/run.py prepare
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_pathological/run.py run
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_pathological/run.py verify
/home/schroeder/miniconda3/envs/general_ml/bin/python -u research/pathmnist_pathological/run.py archive
```

The official source is `https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1`,
verified against MedMNIST 3.0.2's MD5 `55aa9c1e0525abe5a6b9d8343a507616`.
`artifacts/data_manifest.json` records its SHA256, processed-cache hashes,
partition hash, counts and software version. Data remain in
`_local/pathmnist_pathological/data/` and `prepared/`.

The runner creates `_local/pathmnist_pathological/seed-42/` and rejects an
existing output directory. Stdout/stderr and execution receipts are retained
separately in `_local/pathmnist_pathological/logs/`. If interrupted, the same
command with `--resume` restores `latest.pt`, refuses changed scientific
sources/configuration/partition, and keeps completed rounds. Existing outputs
are never replaced by a fresh run.

## Complete checkpoints and later diagnostics

Private, CPU-loadable files on thanos:

- `_local/pathmnist_pathological/seed-42/initial.pt`: round 0 before training.
- `_local/pathmnist_pathological/seed-42/final.pt`: round 50, synchronized
  shared states, saved before the only final test evaluation.
- `_local/pathmnist_pathological/seed-42/latest.pt`: last complete round for
  recovery; client classifier/decoder copies may be pre-aggregation.

Both diagnostic checkpoints contain `classifier`, `decoder`, all ten
`encoders`, complete per-client classifier and autoencoder state dictionaries,
all BatchNorm buffers, SGD/Adam optimizer dictionaries, AMP scaler states,
training/evaluation modes and parameter `requires_grad` flags. Empty initial
optimizer dictionaries, and empty SGD state dictionaries without momentum,
are normal; their parameter groups and settings are still saved. They also
contain exact partitions, seed, configuration, source hashes, loader generator
states, all worker RNG states and the coordinator CPU RNG state. CUDA RNG
states belong to the worker processes actually executing model operations.

For trusted local files:

```python
import torch
from fusedspacefed_core import ResNet20V2, UNetSmallAE

saved = torch.load("_local/pathmnist_pathological/seed-42/final.pt",
                   map_location="cpu", weights_only=False)
classifier = ResNet20V2(9, 3)
classifier.load_state_dict(saved["classifier"], strict=True)
autoencoder = UNetSmallAE(3, 16)
autoencoder.load_state_dict({**saved["encoders"]["0"], **saved["decoder"]}, strict=True)
classifier.eval()
autoencoder.eval()
```

`restore_client` restores complete local model, optimizer, mode and loader
states; on CUDA it also restores the AMP scaler. `restore_rng` restores the
appropriate worker's recorded random states. These files support controlled
decoder/warm-up replays and fixed-state gradient probes. They do not contain
the historical within-round warm-up boundaries; those would need a replay
from a saved anchor, with its scope stated explicitly.

Public `artifacts/` contains numeric results, timing records, partition indices
compressed losslessly, identity, runtime, checkpoint verification and file
hashes, plus the run log. Images and complete checkpoints stay private.
