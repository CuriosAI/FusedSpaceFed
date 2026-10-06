"""Independent reconstruction from scalar Gram entries, without autograd."""
import pytest
from research.digits_mechanism_diagnostics.phase3.archive import reconstructed, check_decomposition


def fixture():
    values = [0., 1., 2., 3., 4., 0., .5, 1., 1.5, 2.]
    gram = [[a * b for b in values] for a in values]
    return {'client_count': 5, 'parameter_count': 14219210, 'gram_original_then_fused': gram,
            'Gamma_original': 2., 'Gamma_fused': .5, 'B': .5, 'Phi': -2.,
            'original': {'client_gradient_norms': values[:5]}, 'fused': {'client_gradient_norms': values[5:]}}


def test_gram_reconstruction_covers_cross_covariance_and_five_client_divisor():
    data = fixture()
    assert reconstructed(data['gram_original_then_fused']) == {'Gamma_original': 2., 'Gamma_fused': .5, 'B': .5, 'Phi': -2.}
    check_decomposition(data)


def test_saved_decomposition_cannot_diverge_from_gram_or_dimensions():
    data = fixture(); data['B'] = .6
    with pytest.raises(ValueError):
        check_decomposition(data)
    data = fixture(); data['parameter_count'] -= 1
    with pytest.raises(ValueError):
        check_decomposition(data)
