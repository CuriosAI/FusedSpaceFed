"""Algebra, batch weighting, raw derivatives, and immutable BN/RNG."""
import torch
from torch import nn
import pytest

from fusedspacefed_core import UNetSmallAE
from research.digits_mechanism_diagnostics.common import state_hash
from research.digits_mechanism_diagnostics.phase3.gradient_stats import decompose, dispersion, weighted_mean
from research.digits_mechanism_diagnostics.phase3.probe_gradients import paired_gradients


def test_exact_identity_cross_covariance_and_population_client_divisor():
    o = [torch.tensor([1., 0.]), torch.tensor([-1., 0.])]
    f = [torch.zeros(2), torch.zeros(2)]
    value = decompose(o, f, chunk_size=1)
    assert value['Gamma_original'] == 1 and value['Gamma_fused'] == 0
    assert value['B'] == 1 and value['Phi'] == -2 and value['identity_residual'] == 0
    same = decompose(o, o)
    assert same['B'] == 0 and same['Phi'] == 0
    assert dispersion(o)['Gamma_decoder'] == 1


def test_random_identity_chunk_independence_and_unequal_batch_weights():
    generator = torch.Generator().manual_seed(41)
    o = [torch.randn(17, generator=generator) for _ in range(5)]
    f = [torch.randn(17, generator=generator) for _ in range(5)]
    a, b = decompose(o, f, 3), decompose(o, f, 17)
    for name in ('Gamma_original', 'Gamma_fused', 'B', 'Phi'):
        assert abs(a[name] - b[name]) < 1e-12
    assert torch.equal(weighted_mean([torch.tensor([1.]), torch.tensor([4.])], [2, 1]), torch.tensor([2.], dtype=torch.float64))
    with pytest.raises(FloatingPointError):
        decompose([torch.tensor([float('nan')]), torch.zeros(1)], [torch.zeros(1), torch.zeros(1)])


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__(); self.bn = nn.BatchNorm2d(3); self.fc = nn.Linear(3, 10)
    def forward(self, x):
        return self.fc(self.bn(x).mean((2, 3)))


@pytest.mark.parametrize('mode', ['eval', 'batch-stateless'])
def test_paired_gradient_measurement_preserves_parameters_buffers_and_rng(mode):
    old = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        classifier = TinyBN(); ae = UNetSmallAE(3, 64)
        x = torch.randn(4, 3, 28, 28); y = torch.arange(4)
        before = (state_hash(classifier.state_dict()), state_hash(ae.state_dict()), torch.get_rng_state())
        original, fused, decoder, metric = paired_gradients(classifier, ae, x, y, batch_norm_mode=mode)
        assert len(original) == len(fused) == sum(p.numel() for p in classifier.parameters())
        assert len(decoder) == 44995 and metric['decoder_fused_norm'] > 0
        assert state_hash(classifier.state_dict()) == before[0]
        assert state_hash(ae.state_dict()) == before[1] and torch.equal(torch.get_rng_state(), before[2])
        assert all(p.grad is None for p in [*classifier.parameters(), *ae.parameters()])
    finally:
        torch.set_num_threads(old)
