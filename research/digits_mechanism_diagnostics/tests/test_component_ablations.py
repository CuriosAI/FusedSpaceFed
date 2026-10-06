"""Matched initialization, exact original updates, pathway and resume checks."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from research.digits_mechanism_diagnostics import common
from research.digits_mechanism_diagnostics.phase2 import runner
from research.digits_mechanism_diagnostics.phase2.client import VariantClient
from research.feature_shift_digits import run_digits
from research.feature_shift_digits_calibrated import runner as original
from research.feature_shift_digits.data import DOMAINS, canonical_hash


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(3, 10)
    def forward(self, x):
        return self.fc(x.mean((2, 3)))


def settings():
    return {**json.loads(Path('research/feature_shift_digits_calibrated/configs/final/expanded-2-1-seed-42.json').read_text())['training'],
            'rounds': 2, 'torch_threads': 2}


@pytest.mark.parametrize('additive,warmup', [(True, 0), (True, 1), (False, 1)])
def test_intervention_pathway_and_expected_parameter_updates(monkeypatch, additive, warmup):
    monkeypatch.setattr(run_digits, 'DigitCNN', TinyCNN)
    torch.manual_seed(8)
    dataset = TensorDataset(torch.randn(2, 3, 28, 28), torch.tensor([0, 1]))
    loader = DataLoader(dataset, batch_size=2)
    client = VariantClient('synthetic', loader, settings(), torch.device('cpu'), additive=additive)
    before = [client.encoder_state(), client.decoder_state(), client.classifier_state()]
    captured = {}
    def ae_hook(module, args, output):
        captured['x'] = args[0].detach().clone(); captured['d'] = output[0].detach().clone()
    def classifier_hook(module, args):
        captured['classification_input'] = args[0].detach().clone()
    client.autoencoder.register_forward_hook(ae_hook)
    client.classifier.register_forward_pre_hook(classifier_hook)
    metric = client.train_round(warmup, 1)
    expected = captured['x'] + captured['d'] if additive else captured['d']
    assert torch.equal(expected, captured['classification_input'])
    assert metric['warmup_steps'] == warmup and metric['classification_steps'] == 1
    assert metric['warmup_batch_mean_loss'] is None if not warmup else metric['warmup_batch_mean_loss'] > 0
    for a, b in zip(before, [client.encoder_state(), client.decoder_state(), client.classifier_state()]):
        assert common.state_distance(a, b) > 0


@pytest.mark.parametrize('variant', ['no-warmup', 'shared-encoder', 'decoder-only'])
def test_variant_resume_rng_private_or_shared_states_and_no_overwrite(tmp_path, monkeypatch, variant):
    monkeypatch.setattr(run_digits, 'DigitCNN', TinyCNN)
    monkeypatch.setattr(common, 'DigitCNN', TinyCNN)
    spec = json.loads(Path('research/digits_mechanism_diagnostics/phase2/plan.json').read_text())['variants'][variant]
    plan = {'variants': {variant: spec}, 'partition_sha256': 'synthetic', 'plan_sha256': 'synthetic',
            'reference_configuration': {'profile_sha256': json.loads(Path('research/capacity_compute_control/flop_profile.json').read_text())['profile_sha256']}}
    monkeypatch.setattr(runner, 'validate_config', lambda c: plan)
    monkeypatch.setattr(runner, 'verify', lambda p: {'partition_sha256': 'synthetic'})
    datasets = {domain: TensorDataset(torch.rand(4, 3, 28, 28, generator=torch.Generator().manual_seed(7 + i)) * 2 - 1,
                                    torch.tensor([0, 1, 2, 3])) for i, domain in enumerate(DOMAINS)}
    def loaders(seed, config):
        return {d: DataLoader(ds, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(seed * 10000 + i))
                for i, (d, ds) in enumerate(datasets.items())}
    monkeypatch.setattr(runner, 'loaders_for', loaders)
    monkeypatch.setattr(runner, 'PreparedDigits', lambda p, d, s: datasets[d])
    config = {'training': settings(), 'seed': 42, 'variant': variant}
    runner.run(config, tmp_path / 'full', torch.device('cpu'))
    partial = runner.run(config, tmp_path / 'resumed', torch.device('cpu'), stop_after=1)
    assert partial['evaluations'] == []
    runner.run(config, tmp_path / 'resumed', torch.device('cpu'), resume=True)
    a = torch.load(tmp_path / 'full/checkpoint.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=False)
    for component in ('classifier', 'decoder'):
        assert all(torch.equal(value, b[component][key]) for key, value in a[component].items())
    for d in DOMAINS:
        assert all(torch.equal(value, b['encoders'][d][key]) for key, value in a['encoders'][d].items())
        assert torch.equal(a['rng']['loaders'][d], b['rng']['loaders'][d])
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    assert a['results']['evaluations'][0]['domains'] == b['results']['evaluations'][0]['domains']
    if variant == 'shared-encoder':
        assert all(common.state_distance(a['encoders'][DOMAINS[0]], a['encoders'][d]) == 0 for d in DOMAINS)
    else:
        assert any(common.state_distance(a['encoders'][DOMAINS[0]], a['encoders'][d]) > 0 for d in DOMAINS[1:])
    with pytest.raises(FileExistsError):
        runner.run(config, tmp_path / 'full', torch.device('cpu'))


def test_common_full_path_reproduces_original_update_bit_for_bit(monkeypatch):
    monkeypatch.setattr(run_digits, 'DigitCNN', TinyCNN)
    dataset = TensorDataset(torch.randn(4, 3, 28, 28), torch.tensor([0, 1, 2, 3]))
    def one(client_class):
        torch.manual_seed(7)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
        client = client_class('synthetic', loader, settings(), torch.device('cpu'))
        for _ in range(2):
            client.train_round(1, 1)
        return [client.encoder_state(), client.decoder_state(), client.classifier_state()], loader.generator.get_state(), torch.get_rng_state()
    a, ag, ar = one(run_digits.DigitsClient)
    b, bg, br = one(VariantClient)
    assert all(common.state_distance(x, y) == 0 for x, y in zip(a, b))
    assert torch.equal(ag, bg) and torch.equal(ar, br)


def test_changed_training_or_unregistered_seed_is_rejected():
    config = json.loads(Path('research/digits_mechanism_diagnostics/phase2/configs/no-warmup-seed-42.json').read_text())
    runner.validate_config(config)
    for key, value in [('seed', 47), ('variant', 'new-variant')]:
        with pytest.raises(ValueError):
            runner.validate_config({**config, key: value})
    config['training']['classifier_lr'] = .02
    with pytest.raises(ValueError):
        runner.validate_config(config)
