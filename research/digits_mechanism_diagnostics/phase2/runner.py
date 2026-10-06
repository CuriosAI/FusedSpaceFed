"""Frozen Digits component ablations: final test only at fixed round300."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import DataLoader
from fusedspacefed_core import clone_state_dict, weighted_average_states
from research.feature_shift_digits.data import DOMAINS, PreparedDigits, atomic_json, canonical_hash, verify
from research.feature_shift_digits.run_digits import aggregate, restore_rng, rng_state, runtime, synchronize, set_seed
from research.capacity_compute_control.audit_results import cost
from research.capacity_compute_control.audit_results import epoch_batches
from research.capacity_compute_control.runner import evaluate_fedavg
from research.digits_mechanism_diagnostics.common import PUBLIC, PARTITION, hashed, initial_state, loaders_for, source_identity
from research.digits_mechanism_diagnostics.phase2.client import VariantClient


def validate_config(config):
    plan = hashed(REPO / config['plan_file'], 'plan_sha256')
    if config['plan_sha256'] != plan['plan_sha256'] or config['variant'] not in plan['variants'] or config['seed'] not in plan['seeds'] or config['training'] != plan['training']:
        raise ValueError('Unregistered variant, seed or changed training settings')
    if config != {'variant': config['variant'], 'seed': config['seed'], 'training': plan['training'],
                  'plan_file': config['plan_file'], 'plan_sha256': plan['plan_sha256']}:
        raise ValueError('Unexpected configuration keys')
    return plan


@torch.no_grad()
def evaluate(client, loader):
    client.autoencoder.eval(); client.classifier.eval()
    class Pathway(torch.nn.Module):
        def forward(self, x):
            d, _ = client.autoencoder(x)
            return client.classifier(x + d if client.additive else d)
    return evaluate_fedavg(Pathway(), loader, client.device)


def run(config, output, device, *, resume=False, stop_after=None):
    started = time.perf_counter(); plan = validate_config(config)
    parent = verify(PARTITION)
    if parent['partition_sha256'] != plan['partition_sha256']:
        raise ValueError('Changed frozen partition')
    if output.exists() and not resume:
        raise FileExistsError('Existing attempt; explicit resume required')
    if resume and not output.is_dir():
        raise FileNotFoundError('Missing resume output')
    output.mkdir(parents=True, exist_ok=resume)
    lock = (output / '.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = config['training']; seed = config['seed']; variant = plan['variants'][config['variant']]
    loaders = loaders_for(seed, settings)
    state = initial_state(seed, settings, device, loaders)
    classifier, decoder, encoders = [state[k] for k in ('classifier', 'decoder', 'encoders')]
    profile = hashed(REPO / 'research/capacity_compute_control/flop_profile.json', 'profile_sha256')
    if profile['profile_sha256'] != plan['reference_configuration']['profile_sha256']:
        raise ValueError('Changed frozen FLOP profile')
    identity = {'seed': seed, 'variant': config['variant'], 'config_sha256': canonical_hash(config),
                'partition_sha256': plan['partition_sha256'], 'plan_sha256': plan['plan_sha256'], 'profile_sha256': profile['profile_sha256'],
                'device': str(device), 'code': source_identity(('research/digits_mechanism_diagnostics/phase2/client.py',
                    'research/digits_mechanism_diagnostics/phase2/runner.py', 'research/capacity_compute_control/audit_results.py',
                    'research/capacity_compute_control/runner.py'))}
    results = {'status': 'running', 'completed_rounds': 0, 'identity': identity, 'configuration': config,
               'runtime': runtime(device), 'history': [], 'evaluations': [], 'sessions': [],
               'parameters': profile['parameters'], 'counted_flop_convention': profile['metric'],
               'excluded_flop_operations': profile['excluded']}
    first = 1
    if resume:
        checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if checkpoint['results']['identity'] != identity:
            raise ValueError('Resume source/config identity differs')
        results = checkpoint['results']
        if results['status'] == 'completed':
            return results
        classifier, decoder, encoders = [checkpoint[k] for k in ('classifier', 'decoder', 'encoders')]
        restore_rng(checkpoint['rng'], device, loaders); first = results['completed_rounds'] + 1
        (output / 'timings.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in results['history']))
    session = {'start_round': first}; results['sessions'].append(session)
    def save():
        session.update(end_round=results['completed_rounds'], wall_seconds=time.perf_counter() - started)
        results['total_session_wall_seconds'] = sum(s['wall_seconds'] for s in results['sessions'])
        temporary = output / 'checkpoint.pt.tmp'
        torch.save({'results': results, 'classifier': classifier, 'decoder': decoder,
                    'encoders': encoders, 'rng': rng_state(device, loaders)}, temporary)
        temporary.replace(output / 'checkpoint.pt'); atomic_json(output / 'results.json', results)
    save()
    for number in range(first, settings['rounds'] + 1):
        synchronize(device); began = time.perf_counter(); classifiers = []; decoders = []; local_encoders = []; clients = {}
        for domain in DOMAINS:
            client = VariantClient(domain, loaders[domain], settings, device, additive=variant['additive_fusion'])
            client.set_classifier_state(classifier); client.set_decoder_state(decoder); client.set_encoder_state(encoders[domain])
            metric = client.train_round(variant['warmup_epochs'], 1)
            classifiers.append(client.classifier_state()); decoders.append(client.decoder_state())
            encoder = client.encoder_state(); local_encoders.append(encoder); encoders[domain] = encoder
            phases = ('warmup', 'classification') if variant['warmup_epochs'] else ('classification',)
            metric['counted_training_flops'] = sum(cost(profile, phase, batch) for phase in phases for batch in epoch_batches(len(loaders[domain].dataset)))
            metric['dense_training_flops'] = sum(cost(profile, phase, batch, True) for phase in phases for batch in epoch_batches(len(loaders[domain].dataset)))
            clients[domain] = metric; del client
        classifier, decoder = aggregate(classifiers, decoders)
        if not variant['private_encoder']:
            shared = weighted_average_states(local_encoders, [1] * len(DOMAINS))
            encoders = {domain: clone_state_dict(shared) for domain in DOMAINS}
        del classifiers, decoders, local_encoders
        synchronize(device)
        row = {'round': number, 'seconds': time.perf_counter() - began, 'clients': clients}
        results['history'].append(row); results['completed_rounds'] = number
        with (output / 'timings.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if number <= 5 or number % 10 == 0 or number == settings['rounds'] or number == stop_after:
            save()
            print(json.dumps({'variant': config['variant'], 'seed': seed, 'round': number,
                              'seconds': row['seconds'], 'remaining_seconds_estimate': statistics.median(r['seconds'] for r in results['history'][-10:]) * (settings['rounds'] - number)}), flush=True)
        if number == stop_after and number < settings['rounds']:
            return results
    saved_rng = rng_state(device, loaders); evaluation = {'round': settings['rounds'], 'split': 'test', 'domains': {}}
    began = time.perf_counter()
    for domain in DOMAINS:
        client = VariantClient(domain, loaders[domain], settings, device, additive=variant['additive_fusion'])
        client.set_classifier_state(classifier); client.set_decoder_state(decoder); client.set_encoder_state(encoders[domain])
        dataset = PreparedDigits(PARTITION, domain, 'test')
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(seed * 10000 + 999))
        evaluation['domains'][domain] = evaluate(client, loader); del client
    restore_rng(saved_rng, device, loaders)
    evaluation.update(seconds=time.perf_counter() - began,
                      uniform_domain_accuracy_percent=statistics.mean(m['accuracy_percent'] for m in evaluation['domains'].values()),
                      sample_weighted_accuracy_percent=100 * sum(m['correct'] for m in evaluation['domains'].values()) / sum(m['total'] for m in evaluation['domains'].values()))
    results.update(evaluations=[evaluation], status='completed',
                   total_counted_training_flops=sum(c['counted_training_flops'] for r in results['history'] for c in r['clients'].values()),
                   total_dense_training_flops=sum(c['dense_training_flops'] for r in results['history'] for c in r['clients'].values()),
                   peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0,
                   peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    save(); print(json.dumps({'status': 'completed', 'variant': config['variant'], 'seed': seed,
                             'uniform_accuracy_percent': evaluation['uniform_domain_accuracy_percent']}), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, torch.device(args.device), resume=args.resume)
