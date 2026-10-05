# Balanced FedBN Digits: FusedSpaceFed protocol

This directory is isolated from all existing experiments. Only FusedSpaceFed
is run; FedBN/FedAvg/FedProx values are transcribed published references.

## Sources and sampling

Primary references: [FedBN repository](https://github.com/med-air/FedBN),
[appendix](https://michaelkamp.org/wp-content/uploads/2021/05/FedBN_appendix.pdf),
and [paper](https://arxiv.org/abs/2102.07623), section 5.1.
Appendix D.2, Tables 3 and 8 specify the classifier and balanced experiment;
Table 11, section E.2, gives domain-specific mean (SD). The paper describes
five trials; original trial seeds and precise statistic round are unavailable.

Repository version: `2fa38adf627a8c8ba71c5fb515b1f2ba00aa8812`.
Relevant functions: `utils/data_preprocess.py::stratified_split/split`,
`utils/data_utils.py::DigitsDataset`, `nets/models.py::DigitModel`, and
`federated/fed_digits.py::prepare_data/train/test/communication`.
The preprocessing combines the original train/test data and applies a
stratified 80/20 split with random_state=0; it does not simply use torchvision's
standard test splits. Its balanced partition uses `part_len=743.8`; part0 is
the first 743 rows. We use the distributed `train_part0.pkl` directly for each
domain and each full distributed `test.pkl`, with no resampling or new split.
Training shuffle varies with run seed; prepared data is frozen across seeds.
Dataset class histograms and identifiers are retained. SVHN's class distribution
differs from other domains; equal client counts do not imply identical labels.

The current README swaps the Digits data/model hyperlinks. We use the actual
`digit_dataset.zip` from the linked mirror, not pretrained models. HF dataset
version `0b6cd64d780662b683a373ddb23aa25d1d968cf8`; file 278200677 bytes, SHA256
`6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221`,
verified against the LFS metadata. It is an author-linked third-party mirror
published in 2025, not a certified archived artifact of the 2021 table.

Preprocessing exactly follows `prepare_data`: SVHN/SynthDigits/USPS PIL bilinear
resize to 28x28; MNIST/USPS grayscale expanded to three channels; MNIST-M
already 28x28 RGB. ToTensor and channelwise Normalize(mean=.5, std=.5): [-1,1].
No augmentation. Cache stores the same resized uint8 pixels; normalization
is performed in Float32. Tests compare both pipelines exactly.
Row IDs identify archive member/index, not missing original raw-image IDs.
Train/test source-row IDs are disjoint; raw provenance before author resplit
cannot be recovered. MNIST-M is derived from MNIST; domains are not claimed
to be independent source identities.

## Training fixed before the test

Five clients, all participating every round, 743 training images each,
300 communication rounds, batch 32 (last batch 7, retained), one local CE
epoch. The six-layer CNN has three convolutions (3→64→64→128), BN/ReLU and
two max-pools, followed by 6272→2048→512→10 FC layers and BN/ReLU before the
last logits. The duplicate third-convolution row in Table 3 is a typesetting
ambiguity; the executable model has three convolutions. Our model has exact
state/output parity with the author's model under common weights.

Classifier optimizer: SGD lr=.01, no momentum or weight decay, recreated each
round as in `fed_digits.py`. Appendix specifies 300 rounds; the current CLI
default is 100, which we explicitly override. CE uses logits, no Softmax.

FusedSpaceFed-specific settings are chosen exogenously, without a Digits sweep:
UNetSmallAE, dz=64, base=16, Adam lr=.0003, betas=(.9,.999), eps=1e-8,
weight_decay=0, L2 clipping=1 per active optimizer. AE LR/clipping come from the
previous training-validation-selected FusedSpaceFed configuration; the CNN SGD
lr=.01 also matches the original FusedSpaceFed default and this benchmark.
Classifier momentum/decay follow FedBN, not the FEMNIST-specific optimizer.
Float32, no AMP/TF32, four threads, deterministic kernels. Double auxiliary
gradient norm prevents overflow without changing model precision.

Warm-up is one additional MSE epoch updating only the private encoder, through
the frozen shared decoder. Classification is one CE epoch updating E/D/C on
`x + D(E(x))`. No MSE term in phase two. Private E persists across rounds;
D/C are uniformly aggregated, including all classifier BN parameters and
running buffers. Integer BN counters copy the first client; equal training
batch counts make them identical. We do not implement FedBN-local BN inside
FusedSpaceFed. Optimizers reset per participation, with Adam state continuous
between the two phases of that participation. Thus the requested one epoch
is the local classification epoch; warm-up adds one pass and extra resources.
This is explicitly a budget difference from the published baselines.

## Evaluation and reproducibility

Only the fixed final round 300 is tested, on every full domain test set,
with current shared D/C and persistent domain E, eval mode, no adaptation or
BN recalibration. No best round/seed/checkpoint, no early stopping and no
tuning after test. The original code tests every round and saves the last
model; testing frequency is a documented difference. Exact round used to
form the published Table 11 statistics is not explicitly specified.
Per-domain accuracy is correct/total. Also record uniform-domain and
sample-weighted overall means. Domain accuracy is the primary comparison.
For ours, mean and sample SD ddof=1 use exactly two run values, not rounds or
domains. Published Table 11 SD remains the author's SD; its ddof is unknown.

Fixed seeds 42/43 map to cuda:1/cuda:0. One process per GPU, two in parallel;
no external process actions. Outputs/checkpoints/logs/dataset are private in
`_local/feature_shift_digits/`. Round checkpoints preserve all private
encoders, shared states, loader generators and Python/NumPy/Torch/device RNGs;
explicit resume refuses changed scientific identity. No overwrite.
Early timing estimates at round 5 exclude final test/IO and are preliminary.

The comparison is ours versus published values, not common baseline reruns.
Same CNN does not imply equal capacity or cost: E/D and warm-up add resources.
Different BN policy, mirror provenance, unspecified published seeds/statistic
round, modern runtime and two versus five trials limit controlled inference.
No significance claim is justified by this descriptive comparison.
