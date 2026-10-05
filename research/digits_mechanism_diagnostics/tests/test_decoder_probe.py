"""Measurement preserves states; synthetic phase boundaries use the original method."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from research.feature_shift_digits import run_digits
from research.digits_mechanism_diagnostics.common import probe_metrics, state_distance, state_hash, tensor_metrics


def test_tensor_metrics_reconstruction_and_amplitude():
    x = torch.tensor([-1., 1.]).reshape(1, 1, 1, 2)
    zero = tensor_metrics(x, torch.zeros_like(x))
    exact = tensor_metrics(x, x)
    assert zero['reconstruction_mse'] == 1 and zero['decoder_rms'] == 0
    assert exact['reconstruction_mse'] == 0 and exact['input_decoder_cosine'] == 1
    assert exact['fused_to_input_rms_ratio'] == 2
    with pytest.raises(FloatingPointError):
        tensor_metrics(x, x * float('nan'))


def test_state_hash_and_distance_include_buffers():
    a = {'weight': torch.tensor([1., 2.]), 'counter': torch.tensor(1)}
    b = {'weight': torch.tensor([1., 2.]), 'counter': torch.tensor(2)}
    assert state_distance(a, a) == 0 and state_distance(a, b) == 1
    assert state_hash(a) == state_hash(a) and state_hash(a) != state_hash(b)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(3, 10)
    def forward(self, x):
        return self.fc(x.mean((2, 3)))


def test_measurement_does_not_update_bn_states_rng_or_modes(monkeypatch):
    monkeypatch.setattr(run_digits, 'DigitCNN', TinyCNN)
    old = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        dataset = TensorDataset(torch.randn(4, 3, 28, 28), torch.arange(4))
        loader = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(7))
        settings = {'dz': 64, 'classifier_lr': .1, 'autoencoder_lr': .0003,
                    'classifier_momentum': 0, 'classifier_weight_decay': 0,
                    'autoencoder_betas': [.9, .999], 'autoencoder_eps': 1e-8,
                    'autoencoder_weight_decay': 0, 'gradient_clip_norm': 10}
        client = run_digits.DigitsClient('synthetic', loader, settings, torch.device('cpu'))
        before = client.classifier_state(); decoder = client.decoder_state(); encoder = client.encoder_state()
        rng = torch.get_rng_state(); loader_rng = loader.generator.get_state()
        probe_metrics(client, dataset, list(range(4)))
        assert client.autoencoder.training and client.classifier.training
        assert state_distance(before, client.classifier_state()) == 0
        assert state_distance(decoder, client.decoder_state()) == 0
        assert state_distance(encoder, client.encoder_state()) == 0
        assert torch.equal(rng, torch.get_rng_state()) and torch.equal(loader_rng, loader.generator.get_state())
        client._warmup(1)
        assert state_distance(before, client.classifier_state()) == 0
        assert state_distance(decoder, client.decoder_state()) == 0
        assert state_distance(encoder, client.encoder_state()) > 0
        client.phase = 'classification'; client._joint_train(1)
        assert state_distance(before, client.classifier_state()) > 0
        assert state_distance(decoder, client.decoder_state()) > 0
    finally:
        torch.set_num_threads(old)
