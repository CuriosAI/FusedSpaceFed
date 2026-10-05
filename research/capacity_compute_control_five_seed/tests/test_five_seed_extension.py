"""No scientific changes: exact source/config parity and synthetic resume."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from research.capacity_compute_control import runner as original
from research.capacity_compute_control_five_seed import runner as extension
from research.capacity_compute_control_five_seed.protocol import (
    ORIGINAL_RUNNER_SHA256, REFERENCE, REPO, validate_final_config,
)
from research.feature_shift_digits.data import DOMAINS, DIRECTORIES, atomic_json, canonical_hash


def test_only_metadata_registration_changes_in_copied_runner():
    path = REPO / 'research/capacity_compute_control/runner.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_RUNNER_SHA256
    expected = path.read_text().replace(
        '\n\ndef utc():',
        "\n\nfrom research.capacity_compute_control_five_seed.protocol import validate_final_config\n"
        "SOURCES += ('research/capacity_compute_control_five_seed/runner.py',\n"
        "            'research/capacity_compute_control_five_seed/protocol.py')\n\n\ndef utc():",
    ).replace("    if config['phase']=='final':\n",
              "    if config['phase']=='final':\n        validate_final_config(config)\n"
    ).replace('seed not in (42,43,44)', 'seed not in (45,46)')
    assert Path(extension.__file__).read_text() == expected


@pytest.mark.parametrize('seed', [45, 46])
def test_new_configs_differ_only_in_seed(seed):
    config = json.loads((REPO / f'research/capacity_compute_control_five_seed/configs/FedAvg-seed-{seed}.json').read_text())
    reference = validate_final_config(config)
    assert reference == json.loads((REPO / REFERENCE).read_text())
    assert {**config, 'seed': 42} == reference


@pytest.mark.parametrize('field,value', [('classifier_lr', .1), ('gradient_clip_norm', 10),
                                         ('rounds', 299), ('batch_size', 16)])
def test_reject_training_protocol_changes(field, value):
    config = json.loads((REPO / REFERENCE).read_text())
    config['seed'] = 45
    config['training'][field] = value
    with pytest.raises(ValueError, match='except for seed'):
        validate_final_config(config)


def test_reject_other_method_phase_seed_or_data():
    reference = {**json.loads((REPO / REFERENCE).read_text()), 'seed': 45}
    for key, value in [('seed', 47), ('seed', True), ('method', 'FusedSpaceFed'),
                       ('phase', 'validation'), ('parent_partition_sha256', 'different')]:
        config = {**reference, key: value}
        with pytest.raises(ValueError):
            validate_final_config(config)


def test_direct_cli_has_no_import_shadowing():
    result = subprocess.run([sys.executable, '-B', str(Path(extension.__file__)), '--help'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 10)

    def forward(self, inputs):
        return self.fc(inputs.mean((2, 3)))


def test_original_driver_parity_and_exact_extension_resume(tmp_path, monkeypatch):
    partition = tmp_path / 'partition'
    partition.mkdir()
    for domain in DOMAINS:
        prefix = DIRECTORIES.get(domain, domain) + '-train'
        np.save(partition / (prefix + '-images.npy'),
                np.random.default_rng(7).integers(0, 256, (6, 3, 28, 28), dtype=np.uint8))
        np.save(partition / (prefix + '-labels.npy'), np.arange(6, dtype=np.int64))
    for module in (original, extension):
        monkeypatch.setattr(module, 'EnlargedDigitCNN', TinyCNN)
        monkeypatch.setattr(module, 'verify', lambda path: {'partition_sha256': 'synthetic'})
    split = {'domains': {d: {'fit_indices': [0, 1, 2, 3], 'validation_indices': [4, 5]}
                         for d in DOMAINS}}
    split['validation_sha256'] = canonical_hash(split)
    split_path = tmp_path / 'split.json'
    atomic_json(split_path, split)
    profile = json.loads((REPO / 'research/capacity_compute_control/flop_profile.json').read_text())
    profile_path = tmp_path / 'profile.json'
    atomic_json(profile_path, profile)
    config = {'phase': 'validation', 'method': 'FedAvg', 'seed': 142,
              'parent_partition_sha256': 'synthetic', 'profile_file': str(profile_path),
              'profile_sha256': profile['profile_sha256'], 'validation_file': str(split_path),
              'validation_sha256': split['validation_sha256'],
              'training': {**profile['settings'], 'rounds': 2, 'torch_threads': 2}}
    old_threads = torch.get_num_threads()
    try:
        original.run(copy.deepcopy(config), partition, tmp_path / 'original', torch.device('cpu'))
        extension.run(copy.deepcopy(config), partition, tmp_path / 'extended', torch.device('cpu'))
        partial = extension.run(config, partition, tmp_path / 'resumed', torch.device('cpu'), stop_after=1)
        assert partial['evaluations'] == []
        extension.run(config, partition, tmp_path / 'resumed', torch.device('cpu'), resume=True)
        checkpoints = [torch.load(tmp_path / name / 'checkpoint.pt', weights_only=False)
                       for name in ('original', 'extended', 'resumed')]
        reference = checkpoints[0]
        for current in checkpoints[1:]:
            assert all(torch.equal(value, current['classifier'][key])
                       for key, value in reference['classifier'].items())
            assert reference['carries'] == current['carries']
            assert torch.equal(reference['rng']['torch'], current['rng']['torch'])
            assert all(torch.equal(reference['rng']['loaders'][d], current['rng']['loaders'][d])
                       for d in DOMAINS)
            assert reference['results']['evaluations'][0]['domains'] == current['results']['evaluations'][0]['domains']
            assert [r['clients'] for r in reference['results']['history']] == [r['clients'] for r in current['results']['history']]
        with pytest.raises(FileExistsError):
            extension.run(config, partition, tmp_path / 'extended', torch.device('cpu'))
    finally:
        torch.set_num_threads(old_threads)
