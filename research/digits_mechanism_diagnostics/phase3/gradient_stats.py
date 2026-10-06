"""Exact fixed-state decomposition; chunked Float64 reductions and audit Gram."""
import math
import torch


def decompose(original, fused, chunk_size=65536):
    if len(original) != len(fused) or len(original) < 2:
        raise ValueError('Need matched client gradients')
    count = len(original); dimension = original[0].numel()
    if any(v.ndim != 1 or v.numel() != dimension for v in [*original, *fused]):
        raise ValueError('Gradient dimensions differ')
    gram = torch.zeros(2 * count, 2 * count, dtype=torch.float64)
    values = torch.zeros(4, dtype=torch.float64)
    for start in range(0, dimension, chunk_size):
        o = torch.stack([v[start:start + chunk_size].double() for v in original])
        f = torch.stack([v[start:start + chunk_size].double() for v in fused])
        if not bool(torch.isfinite(o).all() & torch.isfinite(f).all()):
            raise FloatingPointError('Non-finite gradient')
        oc = o - o.mean(0); fc = f - f.mean(0); bc = fc - oc
        values += torch.stack([oc.square().sum(), fc.square().sum(), bc.square().sum(), 2 * (oc * bc).sum()]) / count
        both = torch.cat([o, f]); gram += both @ both.T
    go, gf, b, phi = [float(v) for v in values]
    residual = gf - go - b - phi
    scale = max(go, gf, b, abs(phi), 1e-30)
    if abs(residual) > 1e-10 * scale:
        raise ArithmeticError('Gradient dispersion identity failed')
    result = {'Gamma_original': go, 'Gamma_fused': gf, 'B': b, 'Phi': phi,
              'B_plus_Phi': b + phi, 'identity_residual': residual,
              'relative_identity_residual': residual / scale,
              'fused_original_ratio': gf / go if go else None,
              'client_count': count, 'parameter_count': dimension, 'gram_original_then_fused': gram.tolist()}
    for label, begin, gamma in (('original', 0, go), ('fused', count, gf)):
        g = gram[begin:begin + count, begin:begin + count]
        energies = g.diag().clamp(min=0)
        average_energy = float(energies.mean())
        cosines = [float(g[i, j] / torch.sqrt(energies[i] * energies[j]))
                   for i in range(count) for j in range(i + 1, count) if energies[i] > 0 and energies[j] > 0]
        result[label] = {'client_gradient_norms': energies.sqrt().tolist(),
                         'mean_client_squared_gradient_norm': average_energy,
                         'mean_gradient_norm': math.sqrt(max(0, float(g.sum()) / count**2)),
                         'normalized_dispersion': gamma / average_energy if average_energy else None,
                         'mean_pairwise_cosine': sum(cosines) / len(cosines) if cosines else None,
                         'defined_cosine_pairs': len(cosines)}
    return result


def dispersion(vectors):
    zero = [torch.zeros_like(v) for v in vectors]
    result = decompose(zero, vectors)
    count = len(vectors)
    return {'Gamma_decoder': result['Gamma_fused'], 'parameter_count': result['parameter_count'],
            **result['fused'], 'gram': [row[count:] for row in result['gram_original_then_fused'][count:]]}


def weighted_mean(vectors, counts):
    if len(vectors) != len(counts) or not counts or any(n <= 0 for n in counts):
        raise ValueError('Invalid batch counts')
    total = sum(counts)
    result = torch.zeros_like(vectors[0], dtype=torch.float64)
    for vector, count in zip(vectors, counts):
        result.add_(vector.double(), alpha=count / total)
    return result
