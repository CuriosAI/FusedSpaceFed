"""Original/fused gradient probes on exactly the same model and training batches."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from fusedspacefed_core import UNetSmallAE
from research.feature_shift_digits.data import DOMAINS, PreparedDigits, atomic_json, file_hash, verify
from research.feature_shift_digits.model import DigitCNN
from research.feature_shift_digits.run_digits import runtime, set_seed, synchronize
from research.digits_mechanism_diagnostics.common import PUBLIC, PARTITION, hashed, source_identity, state_hash
from research.digits_mechanism_diagnostics.phase3.gradient_stats import decompose, dispersion


def flatten(gradients):
    return torch.cat([g.detach().reshape(-1).cpu() for g in gradients])


def paired_gradients(classifier, autoencoder, inputs, labels, *, batch_norm_mode):
    classifier.train(batch_norm_mode == 'batch-stateless'); autoencoder.eval()
    cparams = list(classifier.parameters()); aparams = list(autoencoder.parameters())
    names = [name for name, _ in autoencoder.named_parameters()]
    buffers = {name: value.detach().clone() for name, value in classifier.named_buffers()}
    def reset():
        with torch.no_grad():
            for name, value in classifier.named_buffers():
                value.copy_(buffers[name])
    try:
        logits = classifier(inputs)
        original_loss = nn.functional.cross_entropy(logits, labels)
        if not bool(torch.isfinite(logits).all() & torch.isfinite(original_loss)):
            raise FloatingPointError('Non-finite original prediction/loss')
        original = flatten(torch.autograd.grad(original_loss, cparams))
        reset()
        reconstruction, _ = autoencoder(inputs)
        logits = classifier(inputs + reconstruction)
        fused_loss = nn.functional.cross_entropy(logits, labels)
        if not bool(torch.isfinite(reconstruction).all() & torch.isfinite(logits).all() & torch.isfinite(fused_loss)):
            raise FloatingPointError('Non-finite fused prediction/loss')
        grads = torch.autograd.grad(fused_loss, cparams + aparams)
        fused = flatten(grads[:len(cparams)])
        ae_grads = grads[len(cparams):]
        decoder = flatten([g for name, g in zip(names, ae_grads) if name.startswith(UNetSmallAE.DECODER_PREFIXES)])
        encoder = flatten([g for name, g in zip(names, ae_grads) if name.startswith(UNetSmallAE.ENCODER_PREFIXES)])
        norm = lambda v: float(torch.linalg.vector_norm(v.double()))
        result = {'original_cross_entropy': float(original_loss), 'fused_cross_entropy': float(fused_loss),
                  'classifier_original_norm': norm(original), 'classifier_fused_norm': norm(fused),
                  'decoder_fused_norm': norm(decoder), 'encoder_fused_norm': norm(encoder),
                  'autoencoder_joint_fused_norm': (norm(decoder)**2 + norm(encoder)**2)**.5}
        return original, fused, decoder, result
    finally:
        reset()


def measure_state(state, batches, mode, device):
    classifier = DigitCNN().to(device); classifier.load_state_dict(state['classifier'])
    ae = UNetSmallAE(3, 64).to(device); ae.load_decoder_state(state['decoder'])
    before_classifier = state_hash(classifier.state_dict()); before_decoder = state_hash(ae.decoder_state())
    before_rng = torch.get_rng_state(); before_cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    cdim = sum(p.numel() for p in classifier.parameters()); ddim = sum(p.numel() for p in ae.decoder_parameters())
    original_means = [torch.zeros(cdim, dtype=torch.float64) for _ in DOMAINS]
    fused_means = [torch.zeros(cdim, dtype=torch.float64) for _ in DOMAINS]
    decoder_means = [torch.zeros(ddim, dtype=torch.float64) for _ in DOMAINS]
    per_batch = []; per_client = {d: [] for d in DOMAINS}
    for index in range(5):
        originals = []; fused = []; decoders = []
        for n, domain in enumerate(DOMAINS):
            ae.load_encoder_state(state['encoders'][domain])
            x, y = batches[domain][index]
            o, f, d, metrics = paired_gradients(classifier, ae, x, y, batch_norm_mode=mode)
            originals.append(o); fused.append(f); decoders.append(d)
            original_means[n].add_(o.double(), alpha=.2)
            fused_means[n].add_(f.double(), alpha=.2)
            decoder_means[n].add_(d.double(), alpha=.2)
            metrics.update(batch=index + 1, samples=len(y)); per_client[domain].append(metrics)
        per_batch.append({'batch': index + 1, 'classifier': decompose(originals, fused), 'decoder': dispersion(decoders)})
        del originals, fused, decoders
    result = {'classifier': decompose(original_means, fused_means), 'decoder': dispersion(decoder_means),
              'per_batch': per_batch, 'per_client_batches': per_client, 'batch_norm_mode': mode,
              'gradient_definition': 'Mean of five equally sized batch mean-loss gradients/client; then uniform population dispersion over five client means',
              'state_before': {'classifier': before_classifier, 'decoder': before_decoder},
              'state_after': {'classifier': state_hash(classifier.state_dict()), 'decoder': state_hash(ae.decoder_state())},
              'rng_unchanged': torch.equal(before_rng, torch.get_rng_state()) and
                  (device.type != 'cuda' or torch.equal(before_cuda_rng, torch.cuda.get_rng_state(device)))}
    if result['state_before'] != result['state_after'] or not result['rng_unchanged']:
        raise AssertionError('Gradient measurement mutated state or RNG')
    return result


def run(seed, device, output):
    if output.exists():
        raise FileExistsError('Existing gradient probe output')
    config = hashed(PUBLIC / 'phase3/config.json', 'config_sha256')
    probe = hashed(REPO / config['probe_file'], 'probe_sha256')
    if seed not in config['seeds'] or probe['probe_sha256'] != config['probe_sha256']:
        raise ValueError('Wrong seed/probe')
    parent = verify(PARTITION)
    if parent['partition_sha256'] != probe['parent_partition_sha256']:
        raise ValueError('Changed data')
    set_seed(seed, device, 4); output.mkdir(parents=True); started = time.perf_counter()
    batches = {}
    for domain in DOMAINS:
        dataset = PreparedDigits(PARTITION, domain, 'train'); indices = probe['domains'][domain]['indices']
        examples = [dataset[i] for i in indices]
        x = torch.stack([e[0] for e in examples]).to(device)
        y = torch.tensor([e[1] for e in examples], device=device)
        batches[domain] = [(x[i:i + 32], y[i:i + 32]) for i in range(0, 160, 32)]
    result = {'seed': seed, 'device': str(device), 'configuration': config,
              'source': source_identity(('research/digits_mechanism_diagnostics/phase3/gradient_stats.py',
                                        'research/digits_mechanism_diagnostics/phase3/probe_gradients.py')),
              'runtime': runtime(device), 'status': 'running', 'anchors': {}}
    for anchor in config['anchors']:
        spec = config['checkpoint_hashes'][str(seed)][anchor]; path = REPO / spec['path']
        if file_hash(path) != spec['sha256']:
            raise ValueError('Anchor checkpoint changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        result['anchors'][anchor] = {}
        for mode in config['batch_norm_modes']:
            result['anchors'][anchor][mode] = measure_state(state, batches, mode, device)
            atomic_json(output / 'results.json', result)
            print(json.dumps({'seed': seed, 'anchor': anchor, 'mode': mode,
                              'Gamma_original': result['anchors'][anchor][mode]['classifier']['Gamma_original'],
                              'Gamma_fused': result['anchors'][anchor][mode]['classifier']['Gamma_fused']}), flush=True)
        del state
    synchronize(device)
    result.update(status='completed', wall_seconds=time.perf_counter() - started,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                  peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    atomic_json(output / 'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True); parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.seed, torch.device(args.device), args.output)
