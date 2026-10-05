"""Training-only probe and state-preserving numerical diagnostics."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.utils.data import DataLoader

from fusedspacefed_core import UNetSmallAE, clone_state_dict
from research.feature_shift_digits.data import DOMAINS, PreparedDigits, canonical_hash, file_hash
from research.feature_shift_digits.model import DigitCNN
from research.feature_shift_digits.run_digits import DigitsClient, rng_state, set_seed

REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / 'research/digits_mechanism_diagnostics'
PARTITION = REPO / '_local/feature_shift_digits/prepared'
SOURCES = ('fusedspacefed_core.py', 'research/feature_shift_digits/data.py',
           'research/feature_shift_digits/model.py', 'research/feature_shift_digits/run_digits.py',
           'research/digits_mechanism_diagnostics/common.py')


def hashed(path, field):
    value = json.loads(Path(path).read_text())
    if canonical_hash({k: v for k, v in value.items() if k != field}) != value[field]:
        raise ValueError('Changed ' + field)
    return value


def source_identity(extra=()):
    return {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'source_sha256': {p: file_hash(REPO / p) for p in SOURCES + extra},
            'note': 'New diagnostic files identified by SHA256 before the per-phase archive commit'}


def loaders_for(seed, settings):
    return {d: DataLoader(PreparedDigits(PARTITION, d, 'train'), batch_size=32, shuffle=True,
                          num_workers=0, generator=torch.Generator().manual_seed(seed * 10000 + i))
            for i, d in enumerate(DOMAINS)}


def initial_state(seed, settings, device, loaders):
    set_seed(seed, device, settings['torch_threads'])
    model = DigitCNN()
    classifier = clone_state_dict(model.state_dict())
    del model
    ae = UNetSmallAE(3, settings['dz'])
    state = {'classifier': classifier, 'decoder': ae.decoder_state(),
             'encoders': {d: ae.encoder_state() for d in DOMAINS}, 'rng': rng_state(device, loaders)}
    return state


def load_client(domain, loader, settings, device, state):
    client = DigitsClient(domain, loader, settings, device)
    client.set_classifier_state(state['classifier'])
    client.set_decoder_state(state['decoder'])
    client.set_encoder_state(state['encoders'][domain])
    return client


def state_distance(before, after):
    if set(before) != set(after):
        raise ValueError('Different state names')
    return math.sqrt(sum(float((before[k].double() - after[k].double()).square().sum()) for k in before))


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_metrics(inputs, outputs):
    x, d = inputs.detach().double(), outputs.detach().double()
    if x.shape != d.shape:
        raise ValueError('Reconstruction shape differs')
    if not bool(torch.isfinite(x).all() & torch.isfinite(d).all()):
        raise FloatingPointError('Non-finite diagnostic output')
    signal = float(x.square().mean())
    mse = float((d - x).square().mean())
    dot = float((x * d).mean())
    output_energy = float(d.square().mean())
    fused = x + d
    return {'reconstruction_mse': mse, 'zero_reconstruction_mse': signal,
            'relative_reconstruction_mse': mse / signal,
            'input_rms': math.sqrt(signal), 'decoder_rms': math.sqrt(output_energy),
            'decoder_to_input_rms_ratio': math.sqrt(output_energy / signal),
            'decoder_mean': float(d.mean()), 'decoder_std_population': float(d.std(unbiased=False)),
            'decoder_min': float(d.min()), 'decoder_max': float(d.max()),
            'decoder_max_abs': float(d.abs().max()),
            'decoder_fraction_outside_input_range': float((d.abs() > 1).double().mean()),
            'fused_to_input_rms_ratio': math.sqrt(float(fused.square().mean()) / signal),
            'fused_fraction_outside_input_range': float((fused.abs() > 1).double().mean()),
            'input_decoder_cosine': dot / math.sqrt(signal * output_energy) if output_energy else 0.0,
            'finite': True}


@torch.no_grad()
def probe_metrics(client, dataset, indices):
    modes = (client.autoencoder.training, client.classifier.training)
    client.autoencoder.eval()
    client.classifier.eval()
    inputs, outputs, targets = [], [], []
    original_loss = fused_loss = 0.0
    try:
        for start in range(0, len(indices), 32):
            examples = [dataset[i] for i in indices[start:start + 32]]
            x = torch.stack([e[0] for e in examples]).to(client.device)
            y = torch.tensor([e[1] for e in examples], device=client.device)
            d, _ = client.autoencoder(x)
            logits_original, logits_fused = client.classifier(x), client.classifier(x + d)
            if not bool(torch.isfinite(logits_original).all() & torch.isfinite(logits_fused).all()):
                raise FloatingPointError('Non-finite probe logits')
            original_loss += float(nn.functional.cross_entropy(logits_original, y, reduction='sum'))
            fused_loss += float(nn.functional.cross_entropy(logits_fused, y, reduction='sum'))
            inputs.append(x.cpu()); outputs.append(d.cpu()); targets.extend([e[1] for e in examples])
        metric = tensor_metrics(torch.cat(inputs), torch.cat(outputs))
        metric.update(samples=len(indices), original_cross_entropy=original_loss / len(indices),
                      fused_cross_entropy=fused_loss / len(indices))
        return metric
    finally:
        client.autoencoder.train(modes[0]); client.classifier.train(modes[1])
