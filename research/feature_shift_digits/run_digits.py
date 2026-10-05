"""Only FusedSpaceFed: balanced FedBN Digits, fixed final-round evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time
import weakref

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fusedspacefed_core import FusedSpaceFedClient, UNetSmallAE, clone_state_dict, weighted_average_states
from research.feature_shift_digits.data import DOMAINS, PreparedDigits, atomic_json, canonical_hash, file_hash, verify
from research.feature_shift_digits.model import DigitCNN

SOURCE_FILES = ('fusedspacefed_core.py', 'research/feature_shift_digits/model.py',
                'research/feature_shift_digits/data.py', 'research/feature_shift_digits/run_digits.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def source_identity():
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'source_sha256': {name: file_hash(REPO / name) for name in SOURCE_FILES}}


def set_seed(seed, device, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(device)


def rng_state(device, loaders):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'loaders': {name: loader.generator.get_state() for name, loader in loaders.items()}}


def restore_rng(state, device, loaders):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)
    for name, loader in loaders.items():
        loader.generator.set_state(state['loaders'][name])


def state_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())


class DigitsClient(FusedSpaceFedClient):
    def __init__(self, domain, loader, settings, device):
        super().__init__(domain, loader, 10, 3, settings['dz'], 'multiclass', device,
                         settings['classifier_lr'], settings['autoencoder_lr'], classifier=DigitCNN(), use_amp=False)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=settings['classifier_lr'],
                                                   momentum=settings['classifier_momentum'], weight_decay=settings['classifier_weight_decay'])
        self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=settings['autoencoder_lr'],
                                             betas=tuple(settings['autoencoder_betas']), eps=settings['autoencoder_eps'],
                                             weight_decay=settings['autoencoder_weight_decay'])
        self.clip = settings['gradient_clip_norm']
        self.phase = 'warmup'
        self.statistics = {}
        owner = weakref.proxy(self)

        def clip_hook(optimizer, args, kwargs):
            owner.clip_gradients(optimizer)

        def finite_forward(module, args, output):
            values = output if isinstance(output, tuple) else (output,)
            if any(not bool(torch.isfinite(t).all()) for t in values):
                raise FloatingPointError(f'Non-finite forward: {owner.client_id}/{owner.phase}')

        self.ae_optimizer.register_step_pre_hook(clip_hook)
        self.classifier_optimizer.register_step_pre_hook(clip_hook)
        self.autoencoder.register_forward_hook(finite_forward)
        self.classifier.register_forward_hook(finite_forward)

    def clip_gradients(self, optimizer):
        parameters = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad and p.grad is not None]
        if not parameters:
            raise RuntimeError('Optimizer without active gradients')
        # Double auxiliary norm avoids overflow; model, gradients, updates remain Float32.
        norms = torch.stack([torch.linalg.vector_norm(p.grad.detach().double()) for p in parameters])
        norm = torch.linalg.vector_norm(norms)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError(f'Non-finite gradient: {self.client_id}/{self.phase}')
        coefficient = (self.clip / (norm + 1e-6)).clamp(max=1.0)
        for parameter in parameters:
            parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
        value = float(norm)
        key = self.phase + ('_autoencoder' if optimizer is self.ae_optimizer else '_classifier')
        stat = self.statistics.setdefault(key, {'steps': 0, 'clipped_steps': 0, 'sum_norm_before_clip': 0.0, 'max_norm_before_clip': 0.0})
        stat['steps'] += 1
        stat['clipped_steps'] += int(value > self.clip)
        stat['sum_norm_before_clip'] += value
        stat['max_norm_before_clip'] = max(stat['max_norm_before_clip'], value)

    def train_round(self, warmup_epochs, local_epochs):
        self.statistics = {}
        synchronize(self.device)
        started = time.perf_counter()
        self.phase = 'warmup'
        warmup = self._warmup(warmup_epochs)
        synchronize(self.device)
        middle = time.perf_counter()
        self.phase = 'classification'
        classification = self._joint_train(local_epochs)
        synchronize(self.device)
        ended = time.perf_counter()
        if not warmup or not classification or not all(math.isfinite(x) for x in warmup + classification):
            raise FloatingPointError('Non-finite or empty phase loss')
        return {'warmup_batch_mean_loss': statistics.mean(warmup),
                'classification_batch_mean_loss': statistics.mean(classification),
                'warmup_steps': len(warmup), 'classification_steps': len(classification),
                'encoder_steps': len(warmup) + len(classification),
                'decoder_steps': len(classification), 'classifier_steps': len(classification),
                'warmup_samples': len(self.loader.dataset) * warmup_epochs,
                'classification_samples': len(self.loader.dataset) * local_epochs,
                'warmup_seconds': middle - started, 'classification_seconds': ended - middle,
                'clipping': self.statistics}


def aggregate(classifiers, decoders):
    if len(classifiers) != len(decoders) or not classifiers:
        raise ValueError('Shared-state aggregation lists differ or are empty')
    if any(not name.startswith(UNetSmallAE.DECODER_PREFIXES) for state in decoders for name in state):
        raise ValueError('Private encoder entered aggregation')
    states = weighted_average_states(classifiers, [1] * len(classifiers)), weighted_average_states(decoders, [1] * len(decoders))
    if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for state in states for t in state.values()):
        raise FloatingPointError('Non-finite aggregate')
    return states


@torch.no_grad()
def evaluate(client, loader):
    client.autoencoder.eval()
    client.classifier.eval()
    correct = total = 0
    loss_sum = 0.0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for inputs, labels in loader:
        inputs, labels = inputs.to(client.device), labels.to(client.device)
        reconstruction, _ = client.autoencoder(inputs)
        logits = client.classifier(inputs + reconstruction)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Non-finite evaluation')
        prediction = logits.argmax(1)
        correct += int((prediction == labels).sum())
        total += len(labels)
        loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction='sum'))
        confusion += torch.bincount((labels * 10 + prediction).cpu(), minlength=100).reshape(10, 10)
    if not total:
        raise ValueError('Empty test set')
    return {'correct': correct, 'total': total, 'accuracy_percent': 100 * correct / total,
            'sample_mean_cross_entropy': loss_sum / total, 'confusion_matrix': confusion.tolist()}


def runtime(device):
    return {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'torchvision': __import__('torchvision').__version__, 'pillow': __import__('PIL').__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'amp': False, 'tf32': False, 'precision': 'float32'}


def run(config, partition, output, seed, device, *, resume=False, stop_after_round=None):
    started = time.perf_counter()
    manifest = verify(partition)
    if manifest['partition_sha256'] != config['partition_sha256']:
        raise ValueError('Frozen partition differs from configuration')
    if seed not in config['run_seeds']:
        raise ValueError('Unregistered seed')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device unavailable')
    if output.exists() and not resume:
        raise FileExistsError('Refusing to overwrite existing run; use explicit --resume')
    if resume and not output.is_dir():
        raise FileNotFoundError('Resume directory absent')
    output.mkdir(parents=True, exist_ok=resume)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'benchmark': config['benchmark'], 'method': 'FusedSpaceFed', 'seed': seed,
                'device': str(device), 'config_sha256': canonical_hash(config),
                'partition_sha256': manifest['partition_sha256'], 'code': source_identity()}
    settings = config['training']
    set_seed(seed, device, settings['torch_threads'])
    loaders = {domain: DataLoader(PreparedDigits(partition, domain, 'train'), batch_size=settings['batch_size'],
                                  shuffle=True, drop_last=False, num_workers=0,
                                  generator=torch.Generator().manual_seed(seed * 10000 + index))
               for index, domain in enumerate(DOMAINS)}
    classifier = DigitCNN()
    autoencoder = UNetSmallAE(3, settings['dz'])
    shared_classifier, shared_decoder = clone_state_dict(classifier.state_dict()), autoencoder.decoder_state()
    encoders = {domain: autoencoder.encoder_state() for domain in DOMAINS}
    costs = {'parameters': {'classifier': sum(p.numel() for p in classifier.parameters()),
                            'private_encoder_per_client': sum(p.numel() for p in autoencoder.encoder_parameters()),
                            'shared_decoder': sum(p.numel() for p in autoencoder.decoder_parameters())},
             'shared_state_bytes': state_bytes(shared_classifier) + state_bytes(shared_decoder),
             'logical_communication_bytes_per_round': 2 * len(DOMAINS) * (state_bytes(shared_classifier) + state_bytes(shared_decoder))}
    del classifier, autoencoder
    results = {'identity': identity, 'configuration': config, 'runtime': runtime(device),
               'status': 'running', 'completed_rounds': 0, 'history': [], 'evaluations': [],
               'sessions': [], 'costs': costs, 'result_origin': 'ours'}
    first_round = 1
    if resume:
        saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if saved['results']['identity'] != identity:
            raise ValueError('Resume seed, device, source, commit, configuration or partition differs')
        results = saved['results']
        if results['status'] == 'completed':
            print(json.dumps({'already_completed': True, 'seed': seed}), flush=True)
            return results
        shared_classifier, shared_decoder, encoders = saved['classifier'], saved['decoder'], saved['encoders']
        restore_rng(saved['rng'], device, loaders)
        first_round = results['completed_rounds'] + 1
        # A crash can leave a JSONL suffix newer than the committed checkpoint.
        (output / 'timings.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results['history']))
    session = {'started_utc': utc(), 'start_round': first_round, 'ended_utc': None, 'wall_seconds': None}
    results['sessions'].append(session)
    atomic_json(output / 'results.json', results)

    def checkpoint():
        snapshot = {'results': results, 'classifier': shared_classifier, 'decoder': shared_decoder,
                    'encoders': encoders, 'rng': rng_state(device, loaders)}
        temporary = output / 'checkpoint.pt.tmp'
        torch.save(snapshot, temporary)
        temporary.replace(output / 'checkpoint.pt')
        atomic_json(output / 'results.json', results)

    for round_number in range(first_round, settings['rounds'] + 1):
        synchronize(device)
        round_started = time.perf_counter()
        classifiers, decoders, clients = [], [], {}
        for domain in DOMAINS:
            client = DigitsClient(domain, loaders[domain], settings, device)
            client.set_classifier_state(shared_classifier)
            client.set_decoder_state(shared_decoder)
            client.set_encoder_state(encoders[domain])
            clients[domain] = client.train_round(settings['warmup_epochs'], settings['classification_epochs'])
            classifiers.append(client.classifier_state())
            decoders.append(client.decoder_state())
            encoders[domain] = client.encoder_state()
            del client
        aggregate_started = time.perf_counter()
        shared_classifier, shared_decoder = aggregate(classifiers, decoders)
        del classifiers, decoders
        synchronize(device)
        row = {'round': round_number, 'clients': clients, 'seconds': time.perf_counter() - round_started,
               'aggregation_seconds': time.perf_counter() - aggregate_started}
        results['history'].append(row)
        results['completed_rounds'] = round_number
        session['end_round'] = round_number
        session['wall_seconds'] = time.perf_counter() - started
        session['ended_utc'] = utc()
        with (output / 'timings.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if round_number <= 5 or round_number % config['checkpoint_every'] == 0 or round_number == settings['rounds'] or round_number == stop_after_round:
            checkpoint()
        if round_number <= 5 or round_number % 10 == 0:
            median = statistics.median(r['seconds'] for r in results['history'][-min(10, round_number):])
            progress = {'seed': seed, 'device': str(device), 'round': round_number, 'round_seconds': row['seconds'],
                        'median_recent_round_seconds': median, 'remaining_training_estimate_seconds': median * (settings['rounds'] - round_number),
                        'estimate_excludes_final_test': True, 'elapsed_session_seconds': session['wall_seconds']}
            if round_number == 5:
                results['early_estimate'] = {'observed_utc': utc(), **progress}
                checkpoint()
            print(json.dumps(progress), flush=True)
        if stop_after_round == round_number and round_number < settings['rounds']:
            return results

    # The final round is fixed before training; no intermediate test selection.
    preserved_rng = rng_state(device, loaders)
    evaluation_started = time.perf_counter()
    evaluation = {'round': settings['rounds'], 'domains': {}}
    for domain in DOMAINS:
        client = DigitsClient(domain, loaders[domain], settings, device)
        client.set_classifier_state(shared_classifier)
        client.set_decoder_state(shared_decoder)
        client.set_encoder_state(encoders[domain])
        test_loader = DataLoader(PreparedDigits(partition, domain, 'test'), batch_size=settings['batch_size'],
                                 shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(seed * 10000 + 999))
        evaluation['domains'][domain] = evaluate(client, test_loader)
        del client, test_loader
    restore_rng(preserved_rng, device, loaders)
    evaluation['seconds'] = time.perf_counter() - evaluation_started
    evaluation['uniform_domain_accuracy_percent'] = statistics.mean(v['accuracy_percent'] for v in evaluation['domains'].values())
    evaluation['sample_weighted_accuracy_percent'] = 100 * sum(v['correct'] for v in evaluation['domains'].values()) / sum(v['total'] for v in evaluation['domains'].values())
    results['evaluations'] = [evaluation]
    results['status'] = 'completed'
    session['ended_utc'] = utc()
    session['wall_seconds'] = time.perf_counter() - started
    results['total_session_wall_seconds'] = sum(item['wall_seconds'] or 0 for item in results['sessions'])
    results['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0
    results['peak_cuda_reserved_mib'] = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0
    results['costs']['logical_communication_bytes_total'] = settings['rounds'] * costs['logical_communication_bytes_per_round']
    checkpoint()
    print(json.dumps({'seed': seed, 'status': 'completed', 'test_round': settings['rounds'],
                      'domains': evaluation['domains'], 'total_wall_seconds': results['total_session_wall_seconds']}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['training']['rounds'] != 300 or config['training']['classification_epochs'] != 1 or config['training']['batch_size'] != 32:
        raise ValueError('Definitive Digits protocol must remain 300 rounds / 1 CE epoch / batch 32')
    if config['evaluation'] != {'round': 300, 'adaptation_epochs': 0, 'checkpoint_selection': 'fixed_final'}:
        raise ValueError('Only the fixed final evaluation is authorized')
    if args.device != config['per_seed_device'][str(args.seed)]:
        raise ValueError('Definitive device differs from the frozen per-seed map')
    run(config, args.partition, args.output, args.seed, torch.device(args.device), resume=args.resume)


if __name__ == '__main__':
    main()
