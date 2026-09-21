# Reproducibility

This document describes the experimental protocol implemented in the repository
and the information recorded for each run.

## Implementation overview

FusedSpaceFed uses three model components:

- a private encoder for each client;
- a decoder shared across clients;
- a classifier shared across clients.

At the beginning of every communication round, the server broadcasts the shared
decoder and classifier. Local optimization then proceeds in two phases:

1. **Encoder warm-up.** The decoder and classifier are frozen. The client trains
   only its encoder by minimizing mean squared reconstruction error.
2. **Joint training.** The complete local pipeline is optimized using only the
   classification loss. Predictions are computed from the fused input
   `x + D(E_i(x))`, and gradients propagate through the classifier, decoder and
   private encoder.

After local training, clients transmit only decoder and classifier parameters.
Private encoders remain local and are never aggregated.

FedAvg, FedProx, SCAFFOLD and FusedSpaceFed use the same ResNet20-v2 classifier
architecture and the same initial classifier parameters for a given run.

## Default experimental configuration

| Setting | MedMNIST | FEMNIST |
|---|---:|---:|
| Communication rounds | 50 | 50 |
| Local epochs | 3 | 3 |
| Warm-up epochs | 1 | 1 |
| Batch size | 128 | 256 |
| Classifier optimizer | SGD | SGD |
| Classifier learning rate | 0.01 | 0.01 |
| Autoencoder optimizer | Adam | Adam |
| Autoencoder learning rate | 0.001 | 0.001 |
| FedProx coefficient | 0.01 | 0.01 |
| Aggregation | Uniform client average | Uniform client average |
| Independent seeds | 41, 42, 43, 44, 45 | 41, 42, 43, 44, 45 |

All parameters can be changed through command-line arguments. The complete
configuration is stored in the output file of every experiment.

## MedMNIST protocol

Experiments use the official MedMNIST training and test splits. Images are
loaded from the 64x64 release and resized to 32x32.

The training set is distributed across 10 clients. Two partitioning strategies
are available:

- **Dirichlet:** class-wise allocation controlled by concentration parameter
  `alpha`; every training example is assigned to exactly one client.
- **Pathological:** every client receives examples from a fixed number of
  classes, set with `--classes-per-client`.

All medical clients participate in every communication round. Client sample
counts, total assigned samples and unique assigned samples are included in the
partition audit saved with each run.

ChestMNIST is treated as a multi-label task and optimized with binary
cross-entropy. A single proxy stratum is required only for partitioning: the
first positive finding is used, while samples without positive findings receive
a separate stratum. Original multi-label targets are retained for training and
evaluation.

The default bottleneck widths are:

| Dataset | `dz` |
|---|---:|
| PathMNIST | 16 |
| ChestMNIST | 32 |
| DermaMNIST | 16 |
| OCTMNIST | 8 |
| PneumoniaMNIST | 16 |
| RetinaMNIST | 64 |
| BreastMNIST | 32 |
| BloodMNIST | 16 |
| TissueMNIST | 32 |
| OrganAMNIST | 16 |
| OrganCMNIST | 32 |
| OrganSMNIST | 32 |

## FEMNIST protocol

FEMNIST experiments use the writer-level train and test files produced by the
LEAF preprocessing pipeline. Each writer is treated as one federated client.
Writer data are not pooled or repartitioned.

By default, up to 3,400 writers are included and 340 writers are sampled without
replacement in each communication round. Private encoder states and SCAFFOLD
client control variates persist between client participations. FEMNIST images
are resized from 28x28 to 32x32, and the default bottleneck width is `dz = 64`.

During evaluation, every test writer uses its corresponding private encoder and
the final shared decoder and classifier.

## Evaluation protocol

For every independent run:

1. the dataset partition, model initialization and random seed are fixed;
2. all methods receive the same classifier initialization and data partition;
3. training runs for the configured number of communication rounds;
4. the official test split is evaluated after the final round;
5. accuracy, macro F1, balanced accuracy and gradient dissimilarity are stored;
6. results are summarized using the mean and sample standard deviation across
   independent seeds.

Gradient dissimilarity is computed as the mean squared distance between each
client classifier gradient and the mean client gradient. The default estimate
uses one deterministic batch from every MedMNIST client and one deterministic
batch from 50 FEMNIST writers.

For MedMNIST, the FusedSpaceFed global-test result is the uniform mean of the
client-specific pipelines evaluated with the final shared decoder and
classifier. For FEMNIST, predictions are computed with the private encoder of
the corresponding writer.

## Recorded outputs

Each experiment produces a JSON file containing:

- all command-line options;
- random seeds;
- software, CUDA, cuDNN and device information;
- partition statistics;
- per-round training losses;
- run-level evaluation metrics;
- mean and sample standard deviation across runs.

Generated data and result directories are excluded from version control. Raw
JSON outputs should be retained together with the tables and figures derived
from them.
