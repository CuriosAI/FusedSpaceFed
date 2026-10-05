"""Synthetic CPU tests: no benchmark test use, exact private/RNG resume, freeze."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from research.feature_shift_digits import run_digits as previous
from research.feature_shift_digits.data import DOMAINS, DIRECTORIES, atomic_json, canonical_hash
from research.feature_shift_digits_calibrated import runner
from research.feature_shift_digits_calibrated.audit import check_result, finite_tree, TEST_COUNTS
from research.feature_shift_digits_calibrated.select_stage import rank_candidates


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('entry', ['runner.py', 'controller.py', 'select_stage.py', 'audit.py', 'prepare_plan.py'])
def test_direct_cli_imports(entry):
    result = subprocess.run([sys.executable, '-B', str(Path('research/feature_shift_digits_calibrated') / entry), '--help'],
                            capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, timeout=30)
    assert result.returncode == 0, result.stderr


def test_plan_preserves_protocol_references_and_balanced_grid():
    plan = json.loads(Path('research/feature_shift_digits_calibrated/search_plan.json').read_text())
    original = json.loads(Path('research/feature_shift_digits/config.json').read_text())
    assert plan['fixed_training'] == original['training']
    assert plan['final'] == {'rounds': 300, 'seeds': [42, 43, 44, 45, 46], 'initialization': 'fresh, full 743/domain',
                             'evaluation': 'one test at round300, no adaptation, no checkpoint/seed selection'}
    assert len(plan['candidates']) == 10
    expanded = plan['candidates'][1:]
    assert {(r['training']['classifier_lr'], r['training']['autoencoder_lr']) for r in expanded} == {(lr, ae) for lr in (.02, .05, .1) for ae in (.0001, .0003, .001)}
    assert sorted(r['training']['gradient_clip_norm'] for r in expanded) == [2.] * 3 + [5.] * 3 + [10.] * 3
    pilot, selected = [r for r in plan['candidates'] if r['id'] in plan['mandatory_confirmation_references']]
    keys = ('classifier_lr', 'autoencoder_lr', 'gradient_clip_norm')
    assert tuple(pilot['training'][key] for key in keys) == (.01, .0003, 1.)
    assert tuple(selected['training'][key] for key in keys) == (.02, .0003, 2.)


def test_validation_ranking_uses_two_seed_mean_and_predeclared_ties():
    candidates = [{'id': 'a', 'training': {'classifier_lr': .1, 'autoencoder_lr': .001, 'gradient_clip_norm': 10.}},
                  {'id': 'b', 'training': {'classifier_lr': .02, 'autoencoder_lr': .0003, 'gradient_clip_norm': 2.}}]
    assert rank_candidates(candidates, {'a': [80., 60.], 'b': [72., 72.]})[0]['id'] == 'b'
    assert rank_candidates(candidates, {'a': [72., 72.], 'b': [72., 72.]})[0]['id'] == 'b'


def test_unregistered_updates_or_seeds_and_unfrozen_final_are_rejected():
    path = Path('research/feature_shift_digits_calibrated/screening_queue.json')
    config = json.loads(Path(json.loads(path.read_text())['jobs'][0]['config']).read_text())
    runner.validate_config(config)
    changed = copy.deepcopy(config)
    changed['training']['classification_epochs'] = 2
    with pytest.raises(ValueError, match='protocol'):
        runner.validate_config(changed)
    changed = copy.deepcopy(config)
    changed['training']['classifier_lr'] = .123
    with pytest.raises(ValueError, match='registered'):
        runner.validate_config(changed)
    changed = copy.deepcopy(config)
    changed['seed'] = 42
    with pytest.raises(ValueError, match='seed'):
        runner.validate_config(changed)
    changed = copy.deepcopy(config)
    changed.update(phase='final', seed=42)
    changed['training']['rounds'] = 300
    with pytest.raises(KeyError, match='selection_file'):
        runner.validate_config(changed)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 10)
    def forward(self, x):
        return self.fc(x.mean((2, 3)))


def test_training_only_validation_exact_resume_private_states_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'DigitCNN', TinyCNN)
    monkeypatch.setattr(previous, 'DigitCNN', TinyCNN)
    root = tmp_path / 'data'
    root.mkdir()
    for domain in DOMAINS:
        prefix = DIRECTORIES.get(domain, domain) + '-train'
        np.save(root / (prefix + '-images.npy'), np.random.default_rng(7).integers(0, 256, (6, 3, 28, 28), dtype=np.uint8))
        np.save(root / (prefix + '-labels.npy'), np.arange(6, dtype=np.int64))
    original_dataset = runner.PreparedDigits
    calls = []
    def training_only(path, domain, split):
        calls.append(split)
        assert split == 'train'
        return original_dataset(path, domain, split)
    monkeypatch.setattr(runner, 'PreparedDigits', training_only)
    monkeypatch.setattr(runner, 'verify', lambda path: {'partition_sha256': 'synthetic'})
    profile = json.loads(Path('research/capacity_compute_control/flop_profile.json').read_text())
    settings = {**profile['settings'], 'rounds': 2, 'torch_threads': 2}
    plan = {'plan_sha256': 'synthetic-plan', 'fit_samples_per_client': 4}
    split = {'validation_sha256': 'synthetic-split', 'domains': {d: {'fit_indices': [0, 1, 2, 3], 'validation_indices': [4, 5]} for d in DOMAINS}}
    plan['validation_samples_per_client'] = 2
    monkeypatch.setattr(runner, 'validate_config', lambda config: (plan, split, profile))
    config = {'phase': 'screening', 'method': 'FusedSpaceFed', 'candidate_id': 'synthetic', 'seed': 142,
              'parent_partition_sha256': 'synthetic', 'training': settings}
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    a_result = runner.run(config, root, full, torch.device('cpu'))
    partial = runner.run(config, root, resumed, torch.device('cpu'), stop_after=1)
    assert partial['evaluations'] == []
    b_result = runner.run(config, root, resumed, torch.device('cpu'), resume=True)
    a = torch.load(full / 'checkpoint.pt', weights_only=False)
    b = torch.load(resumed / 'checkpoint.pt', weights_only=False)
    for component in ('classifier', 'decoder'):
        assert all(torch.equal(value, b[component][key]) for key, value in a[component].items())
    for domain in DOMAINS:
        assert all(torch.equal(value, b['encoders'][domain][key]) for key, value in a['encoders'][domain].items())
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    assert all(torch.equal(a['rng']['loaders'][domain], b['rng']['loaders'][domain]) for domain in DOMAINS)
    assert a_result['evaluations'][0]['domains'] == b_result['evaluations'][0]['domains']
    assert a_result['evaluations'][0]['split'] == 'validation'
    assert len(a_result['sessions']) == 1 and len(b_result['sessions']) == 2
    assert b_result['total_session_wall_seconds'] == sum(s['wall_seconds'] for s in b_result['sessions'])
    assert calls and set(calls) == {'train'}
    for result in (a_result, b_result):
        assert all(row['warmup_steps'] == row['classification_steps'] == 1 for record in result['history'] for row in record['clients'].values())
    with pytest.raises(FileExistsError):
        runner.run(config, root, full, torch.device('cpu'))


def synthetic_final():
    config = {'method': 'FusedSpaceFed', 'seed': 42, 'phase': 'final', 'parent_partition_sha256': 'p',
              'validation_sha256': 'v', 'plan_sha256': 's', 'profile_sha256': 'f', 'training': {'rounds': 300}}
    identity = {'config_sha256': canonical_hash(config), 'partition_sha256': 'p', **{k: config[k] for k in ('seed', 'phase', 'method', 'validation_sha256', 'plan_sha256', 'profile_sha256')}}
    metric = {'warmup_steps': 24, 'classification_steps': 24, 'encoder_steps': 48, 'decoder_steps': 24, 'classifier_steps': 24,
              'warmup_samples': 743, 'classification_samples': 743, 'counted_training_flops': 4, 'dense_training_flops': 3}
    domains = {}
    for domain, count in TEST_COUNTS.items():
        matrix = [[0] * 10 for _ in range(10)]
        matrix[0][0] = count
        domains[domain] = {'confusion_matrix': matrix, 'correct': count, 'total': count, 'accuracy_percent': 100.}
    return config, {'configuration': config, 'identity': identity, 'status': 'completed', 'completed_rounds': 300,
                    'history': [{'round': i, 'clients': {d: metric.copy() for d in DOMAINS}} for i in range(1, 301)],
                    'evaluations': [{'round': 300, 'split': 'test', 'domains': domains, 'uniform_domain_accuracy_percent': 100., 'sample_weighted_accuracy_percent': 100.}],
                    'sessions': [{'wall_seconds': 2.}, {'wall_seconds': 3.}], 'total_session_wall_seconds': 5.,
                    'total_counted_training_flops': 6000, 'total_dense_training_flops': 4500}


def test_audit_reconstructs_confusions_final_window_and_whole_resume_duration():
    config, result = synthetic_final()
    check_result(result, config)
    broken = copy.deepcopy(result)
    broken['evaluations'][0]['domains']['SVHN']['accuracy_percent'] = 99.
    with pytest.raises(ValueError, match='Accuracy'):
        check_result(broken, config)
    broken = copy.deepcopy(result)
    broken['evaluations'][0]['round'] = 299
    with pytest.raises(ValueError, match='evaluation'):
        check_result(broken, config)
    broken = copy.deepcopy(result)
    broken['total_session_wall_seconds'] = 3.
    with pytest.raises(ValueError, match='Whole-run'):
        check_result(broken, config)
    with pytest.raises(ValueError, match='Non-finite'):
        finite_tree({'loss': [float('nan')]})
