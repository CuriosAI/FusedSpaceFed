"""Decoder/warm-up probes on copies of initial and final Digits states."""
import argparse
import json
from pathlib import Path
import resource
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from research.feature_shift_digits.data import DOMAINS, atomic_json, file_hash, verify
from research.feature_shift_digits.run_digits import restore_rng, rng_state, runtime, synchronize
from research.digits_mechanism_diagnostics.common import (
    PUBLIC, PARTITION, hashed, initial_state, load_client, loaders_for,
    probe_metrics, source_identity, state_distance, state_hash,
)


@torch.no_grad()
def visual_tensors(client, dataset, indices):
    old = client.autoencoder.training
    client.autoencoder.eval()
    x = torch.stack([dataset[i][0] for i in indices]).to(client.device)
    d, _ = client.autoencoder(x)
    client.autoencoder.train(old)
    return {'input': x.cpu(), 'decoder': d.cpu(), 'fused': (x + d).cpu()}


def figure(path, domain, anchor, stages, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [('before', 'input')] + [(stage, signal) for stage in ('before', 'after_warmup', 'after_classification') for signal in ('decoder', 'fused')]
    fig, axes = plt.subplots(len(rows), len(labels), figsize=(len(labels) * 2, len(rows) * 1.7))
    for r, (stage, signal) in enumerate(rows):
        for c, label in enumerate(labels):
            raw = stages[stage][signal][c]
            axes[r, c].imshow(((raw.permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f'label {label}')
            if c == 0:
                axes[r, c].set_ylabel(f'{stage}\n{signal}', fontsize=9)
    fig.suptitle(f'{domain}: {anchor}, seed 42\nFixed [-1,1] display; raw values recorded, no per-panel rescaling', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(path, dpi=120); plt.close(fig)


def run(seed, device, output):
    if output.exists():
        raise FileExistsError('Diagnostic output already exists')
    config = hashed(PUBLIC / 'phase1/config.json', 'config_sha256')
    probe = hashed(REPO / config['probe_file'], 'probe_sha256')
    if seed not in config['seeds'] or probe['probe_sha256'] != config['probe_sha256']:
        raise ValueError('Wrong seed/probe')
    parent = verify(PARTITION)
    if parent['partition_sha256'] != config['partition_sha256']:
        raise ValueError('Changed partition')
    settings = config['training']
    source = config['checkpoints'][str(seed)]
    checkpoint_path = REPO / source['path']
    if file_hash(checkpoint_path) != source['sha256']:
        raise ValueError('Frozen checkpoint changed')
    output.mkdir(parents=True)
    started = time.perf_counter()
    loaders = loaders_for(seed, settings)
    first = initial_state(seed, settings, device, loaders)
    final = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if final['results']['status'] != 'completed' or final['results']['identity']['seed'] != seed:
        raise ValueError('Wrong completed source checkpoint')
    result = {'seed': seed, 'device': str(device), 'configuration': config,
              'source': source_identity(('research/digits_mechanism_diagnostics/phase1/diagnose.py',)),
              'runtime': runtime(device), 'source_checkpoint_sha256': source['sha256'],
              'interpretation': config['interpretation'], 'status': 'running', 'anchors': {}}
    for anchor, state in (('initialization', first), ('final-round300', final)):
        restore_rng(state['rng'], device, loaders)
        anchor_output = output / anchor
        anchor_output.mkdir()
        # Preserve shared anchor and private encoders for the later fixed-state gradient diagnostic.
        anchor_path = anchor_output / 'anchor.pt'
        torch.save({k: state[k] for k in ('classifier', 'decoder', 'encoders', 'rng')}, anchor_path)
        row = {'checkpoint_sha256': file_hash(anchor_path), 'classifier_state_sha256': state_hash(state['classifier']),
               'decoder_state_sha256': state_hash(state['decoder']), 'domains': {}}
        for domain in DOMAINS:
            client = load_client(domain, loaders[domain], settings, device, state)
            dataset = loaders[domain].dataset
            spec = probe['domains'][domain]
            before = {'classifier': client.classifier_state(), 'decoder': client.decoder_state(), 'encoder': client.encoder_state()}
            metrics = {'before': probe_metrics(client, dataset, spec['indices'])}
            visuals = {'before': visual_tensors(client, dataset, spec['visual_indices'])} if seed == 42 else {}
            client.statistics = {}; client.phase = 'warmup'
            warmup = client._warmup(1)
            warm = {'classifier': client.classifier_state(), 'decoder': client.decoder_state(), 'encoder': client.encoder_state()}
            metrics['after_warmup'] = probe_metrics(client, dataset, spec['indices'])
            if seed == 42:
                visuals['after_warmup'] = visual_tensors(client, dataset, spec['visual_indices'])
            client.phase = 'classification'
            classification = client._joint_train(1)
            after = {'classifier': client.classifier_state(), 'decoder': client.decoder_state(), 'encoder': client.encoder_state()}
            metrics['after_classification'] = probe_metrics(client, dataset, spec['indices'])
            delta = {stage: {component: state_distance(a[component], b[component]) for component in before}
                     for stage, a, b in (('warmup', before, warm), ('classification', warm, after))}
            if delta['warmup']['classifier'] or delta['warmup']['decoder']:
                raise AssertionError('Warm-up modified a shared component')
            phases_path = anchor_output / (domain + '-phases.pt')
            torch.save({'before_encoder': before['encoder'], 'after_warmup_encoder': warm['encoder'],
                        'after_classification': after, 'rng_after': rng_state(device, loaders)}, phases_path)
            metrics.update(parameter_state_l2_changes=delta, clipping=client.statistics,
                           warmup_loss=statistics.mean(warmup), classification_loss=statistics.mean(classification),
                           warmup_steps=len(warmup), classification_steps=len(classification),
                           local_replay_checkpoint_sha256=file_hash(phases_path))
            if seed == 42:
                visuals['after_classification'] = visual_tensors(client, dataset, spec['visual_indices'])
                figure(output / f'{anchor}-{domain}.png', domain, anchor, visuals, spec['visual_labels'])
            row['domains'][domain] = metrics
            del client, before, warm, after
        result['anchors'][anchor] = row
        atomic_json(output / 'results.json', result)
        print(json.dumps({'seed': seed, 'anchor': anchor, 'completed_clients': 5}), flush=True)
    synchronize(device)
    result.update(status='completed', wall_seconds=time.perf_counter() - started,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                  peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    atomic_json(output / 'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, torch.device(args.device), args.output)
