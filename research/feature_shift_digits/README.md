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
explicit `--resume` only for the same identity. The first pair was executed by
`launch_pair.py`; the authorized expansion was executed by `launch_extension.py`,
after the first pair finished. Both scripts use the interpreter above, retain
stdout/stderr privately and preserve process exit codes. They refuse existing
outputs and do not interrupt external jobs.

The additional three runs use the unchanged scientific runner through a
registry-only wrapper. For example (45 uses cuda:0; 46 uses cuda:1):

```bash
/home/schroeder/miniconda3/envs/general_ml/bin/python research/feature_shift_digits/run_registered_seed.py \
  --config research/feature_shift_digits/config_five.json \
  --partition _local/feature_shift_digits/prepared \
  --output _local/feature_shift_digits/runs/seed-44 --seed 44 --device cuda:1
```

These document the five completed runs, not additional trials. Full config
hashes differ only because the registry was expanded; the common scientific
hash is recorded in `five_run_authorization.json`. Resume additionally requires
the original run's Git commit, device and source identity; no run was resumed.

Pinned dataset download, if needed on another machine:

```bash
curl --fail --location --continue-at - \
  --output _local/feature_shift_digits/reference/digit_dataset.zip \
  https://huggingface.co/datasets/Jemary/FedBN_Dataset/resolve/0b6cd64d780662b683a373ddb23aa25d1d968cf8/digit_dataset.zip
```

Create only the parent reference directory beforehand. The preparer verifies
SHA256 `6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221`
before loading trusted author-linked pickle payloads. It reuses verified caches.
The archive and cache are never versioned. See the provenance limits in the report.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/schroeder/miniconda3/envs/general_ml/bin/python -m pytest \
  tests research/feature_shift_digits/tests -q
```

The numerical audit needs only Python's standard library. It neither evaluates
models nor loads images/checkpoints. Results are losslessly compressed, with
per-round timing records, counts, confusion matrices and all five seeds:

```bash
python3 research/feature_shift_digits/audit_and_package.py verify \
  --directory research/feature_shift_digits
```

`FEATURE_SHIFT_HANDOFF.zip` contains this directory and the exact core source,
excluding datasets/checkpoints/private logs. After extracting at a repository
root, append `--payload-only` to verify extracted files without the original ZIP
beside the report. Keep the original ZIP beside the report for full ZIP CRC/SHA
verification. Both checks work without Git, CUDA or ML packages. Training requires
the existing ML environment, a Git checkout and checksum-pinned prepared data.
