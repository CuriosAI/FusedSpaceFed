"""Fused-only Digits runner: training validation, frozen final tests, exact resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.utils.data import DataLoader, Subset
from fusedspacefed_core import UNetSmallAE, clone_state_dict
from research.feature_shift_digits.data import DOMAINS, PreparedDigits, atomic_json, canonical_hash, file_hash, verify
from research.feature_shift_digits.model import DigitCNN
from research.feature_shift_digits.run_digits import (DigitsClient, aggregate, evaluate, restore_rng, rng_state,
                                                     runtime, set_seed, state_bytes, synchronize)
from research.capacity_compute_control.compute import epoch_batches, fused_round_cost, step_cost

SOURCES = ('fusedspacefed_core.py', 'research/feature_shift_digits/data.py',
           'research/feature_shift_digits/model.py', 'research/feature_shift_digits/run_digits.py',
           'research/capacity_compute_control/compute.py', 'research/feature_shift_digits_calibrated/runner.py')
TUNABLE = ('classifier_lr', 'autoencoder_lr', 'gradient_clip_norm')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_hashed(path, field):
    value = json.loads(Path(path).read_text())
    content = value.copy()
    if canonical_hash({key: item for key, item in content.items() if key != field}) != content[field]:
        raise ValueError(f'Changed {field}')
    return value


def validate_config(config):
    plan = read_hashed(REPO / config['plan_file'], 'plan_sha256')
    validation = read_hashed(REPO / config['validation_file'], 'validation_sha256')
    profile = read_hashed(REPO / config['profile_file'], 'profile_sha256')
    for name, value in (('plan', plan), ('validation', validation), ('profile', profile)):
        if value[name + '_sha256'] != config[name + '_sha256']:
            raise ValueError(f'Changed {name} identity')
    if config['method'] != 'FusedSpaceFed' or config['parent_partition_sha256'] != plan['parent_partition_sha256']:
        raise ValueError('Wrong method/partition')
    if validation['parent_partition_sha256'] != config['parent_partition_sha256']:
        raise ValueError('Validation belongs to another partition')
    settings = config['training']
    expected = plan['fixed_training'].copy()
    expected.update({key: settings[key] for key in TUNABLE}, rounds=settings['rounds'])
    if settings != expected:
        raise ValueError('Changed architecture/optimizer/experimental protocol')
    candidates = {row['id']: row['training'] for row in plan['candidates']}
    selected_settings = {key: settings[key] for key in TUNABLE}
    if selected_settings != candidates.get(config['candidate_id']):
        raise ValueError('Hyperparameters not registered before calibration')
    phase = config['phase']
    if phase not in ('screening', 'confirmation', 'final'):
        raise ValueError('Unknown campaign phase')
    phase_plan = plan[phase]
    if settings['rounds'] != phase_plan['rounds'] or config['seed'] not in phase_plan['seeds']:
        raise ValueError('Wrong seed/horizon')
    if phase == 'confirmation':
        shortlist = read_hashed(REPO / config['shortlist_file'], 'shortlist_sha256')
        if shortlist['shortlist_sha256'] != config['shortlist_sha256'] or config['candidate_id'] not in shortlist['candidate_ids']:
            raise ValueError('Confirmation not registered by screening')
    if phase == 'final':
        selection = read_hashed(REPO / config['selection_file'], 'selection_sha256')
        if selection['selection_sha256'] != config['selection_sha256'] or selection['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('Final configuration is not frozen')
        if selected_settings != selection['selected_training_settings'] or config['candidate_id'] != selection['selected_candidate_id']:
            raise ValueError('Final hyperparameters differ from validation selection')
        if settings['rounds'] != 300 or config['seed'] not in (42, 43, 44, 45, 46):
            raise ValueError('Wrong final protocol')
    return plan, validation, profile


def run(config, partition, output, device, *, resume=False, stop_after=None):
    started = time.perf_counter()
    plan, validation, profile = validate_config(config)
    parent = verify(partition)
    if parent['partition_sha256'] != config['parent_partition_sha256']:
        raise ValueError('Frozen parent data changed')
    if output.exists() and not resume:
        raise FileExistsError('Refusing to overwrite a previous attempt')
    if resume and not output.is_dir():
        raise FileNotFoundError('Resume directory missing')
    output.mkdir(parents=True, exist_ok=resume)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings, seed = config['training'], config['seed']
    identity = {'method': 'FusedSpaceFed', 'phase': config['phase'], 'seed': seed, 'device': str(device),
                'config_sha256': canonical_hash(config), 'partition_sha256': parent['partition_sha256'],
                'validation_sha256': validation['validation_sha256'], 'plan_sha256': plan['plan_sha256'],
                'profile_sha256': profile['profile_sha256'],
                'code': {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                         'source_sha256': {name: file_hash(REPO / name) for name in SOURCES}}}
    set_seed(seed, device, settings['torch_threads'])
    datasets = {domain: PreparedDigits(partition, domain, 'train') for domain in DOMAINS}
    final = config['phase'] == 'final'
    if not final:
        datasets = {domain: Subset(dataset, validation['domains'][domain]['fit_indices']) for domain, dataset in datasets.items()}
    count = plan['full_samples_per_client'] if final else plan['fit_samples_per_client']
    if any(len(dataset) != count for dataset in datasets.values()):
        raise ValueError('Wrong training counts')
    loaders = {domain: DataLoader(datasets[domain], batch_size=32, shuffle=True, num_workers=0,
                                 generator=torch.Generator().manual_seed(seed * 10000 + index))
               for index, domain in enumerate(DOMAINS)}
    classifier_model = DigitCNN()
    ae = UNetSmallAE(3, settings['dz'])
    classifier, decoder = clone_state_dict(classifier_model.state_dict()), ae.decoder_state()
    encoders = {domain: ae.encoder_state() for domain in DOMAINS}
    costs = {'parameters': {'classifier': sum(p.numel() for p in classifier_model.parameters()),
                             'private_encoder_per_client': sum(p.numel() for p in ae.encoder_parameters()),
                             'shared_decoder': sum(p.numel() for p in ae.decoder_parameters())},
             'logical_communication_bytes_per_round': 10 * (state_bytes(classifier) + state_bytes(decoder)),
             'counted_flop_metric': profile['metric'], 'excluded_flop_operations': profile['excluded']}
    del classifier_model, ae
    results = {'identity': identity, 'configuration': config, 'runtime': runtime(device), 'result_origin': 'ours',
               'status': 'running', 'completed_rounds': 0, 'history': [], 'evaluations': [], 'sessions': [], 'costs': costs}
    first = 1
    if resume:
        snapshot = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if snapshot['results']['identity'] != identity:
            raise ValueError('Resume source/commit/seed/device/configuration/partition identity changed')
        results = snapshot['results']
        if results['status'] == 'completed':
            return results
        classifier, decoder, encoders = snapshot['classifier'], snapshot['decoder'], snapshot['encoders']
        restore_rng(snapshot['rng'], device, loaders)
        first = results['completed_rounds'] + 1
        (output / 'timings.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results['history']))
    session = {'started_utc': utc(), 'start_round': first}
    results['sessions'].append(session)

    def save():
        session.update(ended_utc=utc(), end_round=results['completed_rounds'], wall_seconds=time.perf_counter() - started)
        results['total_session_wall_seconds'] = sum(row['wall_seconds'] for row in results['sessions'])
        temporary = output / 'checkpoint.pt.tmp'
        torch.save({'results': results, 'classifier': classifier, 'decoder': decoder, 'encoders': encoders,
                    'rng': rng_state(device, loaders)}, temporary)
        temporary.replace(output / 'checkpoint.pt')
        atomic_json(output / 'results.json', results)

    save()
    for number in range(first, settings['rounds'] + 1):
        synchronize(device)
        began = time.perf_counter()
        classifiers, decoders, metrics = [], [], {}
        for domain in DOMAINS:
            client = DigitsClient(domain, loaders[domain], settings, device)
            client.set_classifier_state(classifier)
            client.set_decoder_state(decoder)
            client.set_encoder_state(encoders[domain])
            metric = client.train_round(1, 1)
            classifiers.append(client.classifier_state())
            decoders.append(client.decoder_state())
            encoders[domain] = client.encoder_state()
            del client
            metric['counted_training_flops'] = fused_round_cost(profile, count)
            metric['dense_training_flops'] = sum(step_cost(profile, phase, batch, dense_only=True)
                                                for phase in ('warmup', 'classification') for batch in epoch_batches(count))
            metrics[domain] = metric
        classifier, decoder = aggregate(classifiers, decoders)
        del classifiers, decoders
        synchronize(device)
        row = {'round': number, 'clients': metrics, 'seconds': time.perf_counter() - began}
        results['history'].append(row)
        results['completed_rounds'] = number
        with (output / 'timings.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if number <= 5 or number % 10 == 0 or number == settings['rounds'] or number == stop_after:
            save()
            print(json.dumps({'phase': config['phase'], 'candidate': config['candidate_id'], 'seed': seed,
                              'round': number, 'round_seconds': row['seconds'],
                              'remaining_training_seconds_estimate': statistics.median(r['seconds'] for r in results['history'][-10:]) * (settings['rounds'] - number)}), flush=True)
        if number == stop_after and number < settings['rounds']:
            return results
    preserved = rng_state(device, loaders)
    evaluation = {'round': settings['rounds'], 'split': 'test' if final else 'validation', 'domains': {}}
    began = time.perf_counter()
    for domain in DOMAINS:
        dataset = PreparedDigits(partition, domain, 'test' if final else 'train')
        if not final:
            dataset = Subset(dataset, validation['domains'][domain]['validation_indices'])
            if len(dataset) != plan['validation_samples_per_client']:
                raise ValueError('Wrong validation counts')
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(seed * 10000 + 999))
        client = DigitsClient(domain, loaders[domain], settings, device)
        client.set_classifier_state(classifier)
        client.set_decoder_state(decoder)
        client.set_encoder_state(encoders[domain])
        evaluation['domains'][domain] = evaluate(client, loader)
        del client
    restore_rng(preserved, device, loaders)
    evaluation['seconds'] = time.perf_counter() - began
    evaluation['uniform_domain_accuracy_percent'] = statistics.mean(row['accuracy_percent'] for row in evaluation['domains'].values())
    evaluation['sample_weighted_accuracy_percent'] = 100 * sum(row['correct'] for row in evaluation['domains'].values()) / sum(row['total'] for row in evaluation['domains'].values())
    results['evaluations'] = [evaluation]
    results['status'] = 'completed'
    results['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0
    results['peak_cuda_reserved_mib'] = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0
    results['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results['total_counted_training_flops'] = sum(metric['counted_training_flops'] for row in results['history'] for metric in row['clients'].values())
    results['total_dense_training_flops'] = sum(metric['dense_training_flops'] for row in results['history'] for metric in row['clients'].values())
    results['costs']['total_logical_communication_bytes'] = len(results['history']) * costs['logical_communication_bytes_per_round']
    save()
    print(json.dumps({'status': 'completed', 'seed': seed, 'evaluation': evaluation}), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.partition, args.output, torch.device(args.device), resume=args.resume)
