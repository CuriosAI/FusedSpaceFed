# FusedSpaceFed on balanced FedBN Digits

See [PROTOCOL.md](PROTOCOL.md) and the final `FEATURE_SHIFT_REPORT.md`.
No baseline runner is provided. Keep datasets, checkpoints and full logs in
`_local/feature_shift_digits/`, outside Git.

Use the existing general_ml interpreter; no installations are required:

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python -m research.feature_shift_digits.data \
  --archive _local/feature_shift_digits/reference/digit_dataset.zip \
  --output _local/feature_shift_digits/prepared

/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits/run_digits.py \
  --config research/feature_shift_digits/config.json \
  --partition _local/feature_shift_digits/prepared \
  --output _local/feature_shift_digits/runs/seed-42 --seed 42 --device cuda:1

/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits/run_digits.py \
  --config research/feature_shift_digits/config.json \
  --partition _local/feature_shift_digits/prepared \
  --output _local/feature_shift_digits/runs/seed-43 --seed 43 --device cuda:0
```

The runner creates each output directory, refuses overwrite, and allows
explicit `--resume` only for the same identity. These are commands for the two
authorized definitive runs, not additional trials.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/schroeder/miniconda3/envs/general_ml/bin/python -m pytest \
  tests research/feature_shift_digits/tests -q
```
