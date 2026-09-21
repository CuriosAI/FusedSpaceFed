# FusedSpaceFed

Reference implementation for **FusedSpaceFed: Enhanced Federated Learning
Through Dual-Space Data Fusion**.

The code follows the two-phase protocol in the manuscript:

1. each client keeps a private encoder;
2. the server broadcasts the shared decoder and classifier;
3. the client freezes decoder and classifier and warms up only its encoder with
   pixel-wise reconstruction loss;
4. the client unfreezes the full pipeline and trains it using classification
   loss only on `x + D(E_i(x))`;
5. only decoder and classifier parameters are sent to the server and averaged.

FedAvg, FedProx and SCAFFOLD use the same ResNet20-v2 classifier and initial
classifier parameters.

## Repository layout

| File | Purpose |
|---|---|
| `fusedspacefed_core.py` | Models, clients, partitions, aggregation, metrics and gradient dissimilarity |
| `train_medmnist.py` | All 12 MedMNIST datasets, Dirichlet and pathological experiments |
| `train_femnist.py` | Natural writer-level clients from the LEAF FEMNIST JSON files |
| `tests/test_core.py` | Partition, aggregation, architecture and gradient-flow checks |
| `REPRODUCIBILITY.md` | Exact mapping from manuscript statements to code and documented defaults |
| `.github/workflows/tests.yml` | Automatic syntax and unit tests after every GitHub push |


## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/CuriosAI/FusedSpaceFed.git
cd FusedSpaceFed
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install and test:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

## MedMNIST

The script downloads the official splits through the `medmnist` package. It
loads the 64x64 release and resizes inputs to 32x32, as recorded in the output
configuration.

Paper configuration for PathMNIST at alpha 0.05:

```bash
python train_medmnist.py \
  --dataset pathmnist \
  --partition dirichlet \
  --alpha 0.05 \
  --clients 10 \
  --rounds 50 \
  --local-epochs 3 \
  --warmup-epochs 1 \
  --methods fedavg fedprox scaffold fusedspacefed
```

The default is five independent runs with seeds `41 42 43 44 45`. The default
`dz` is read from the manuscript's “Best dz” column for each dataset. Override
it only for a pre-declared validation sweep:

```bash
python train_medmnist.py --dataset retinamnist --alpha 0.50 --dz 64
```

Pathological two-class partition:

```bash
python train_medmnist.py \
  --dataset dermamnist \
  --partition pathological \
  --classes-per-client 2
```

The script supports:

`pathmnist`, `chestmnist`, `dermamnist`, `octmnist`, `pneumoniamnist`,
`retinamnist`, `breastmnist`, `bloodmnist`, `tissuemnist`, `organamnist`,
`organcmnist`, and `organsmnist`.

## FEMNIST

Prepare FEMNIST with the official LEAF preprocessing pipeline. Pass the LEAF
FEMNIST directory whose structure is:

```text
femnist/
└── data/
    ├── train/*.json
    └── test/*.json
```

Then run:

```bash
python train_femnist.py \
  --leaf-root /path/to/leaf/data/femnist \
  --max-clients 3400 \
  --clients-per-round 340 \
  --rounds 50 \
  --local-epochs 3 \
  --methods fedavg fedprox scaffold fusedspacefed
```

Every LEAF writer remains a client. The script does not pool writers and does
not manufacture a second synthetic Dirichlet partition.

## Outputs and evaluation safeguards

Each command writes a JSON file under `results/` containing:

- every CLI option and seed;
- Python, package, CUDA/cuDNN and device versions;
- a partition audit;
- per-round training losses;
- final accuracy, macro F1, balanced accuracy and gradient dissimilarity;
- mean and sample standard deviation across independent runs.

The official test split is evaluated only once, after the final communication
round. FusedSpaceFed's FEMNIST test prediction uses each writer's own private encoder
together with the final shared decoder and classifier.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation

```bibtex
@article{dicecco2025fusedspacefed,
  title  = {FusedSpaceFed: Enhanced Federated Learning Through Dual-Space Data Fusion},
  author = {Di Cecco, Antonio and Metta, Carlo and Bianchi, Luigi Amedeo and
            Veglio, Michelangelo and Parton, Maurizio},
  year   = {2026}
}
```

