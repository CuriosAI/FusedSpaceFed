"""Frozen-stage calibration/final runner with complete persistent checkpoints."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from fusedspacefed_core import ResNet20V2, UNetSmallAE, clone_state_dict, seed_everything, weighted_average_states, evaluate_fused
from research.pathmnist_pathological.run import (atomic_save, assert_finite, client_snapshot, restore_client,
    rng_state, restore_rng, file_hash, json_hash, write_json, resident_loader)
from research.pathmnist_calibrated.client import CalibratedClient
from research.pathmnist_calibrated.data import PUBLIC, PRIVATE, DATA, verify, cache, loaders

SOURCES = ('fusedspacefed_core.py', 'research/pathmnist_pathological/run.py',
           'research/pathmnist_calibrated/data.py', 'research/pathmnist_calibrated/client.py',
           'research/pathmnist_calibrated/runner.py', 'research/pathmnist_calibrated/search_plan.json',
           'research/pathmnist_calibrated/partition.json.gz')


def identity():
    return {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {p: file_hash(ROOT / p) for p in SOURCES}}


@torch.no_grad()
def evaluate(classifier_state, decoder_state, encoders, images, labels, indices, settings, device):
    model = ResNet20V2(9, 3).to(device); model.load_state_dict(classifier_state); model.eval()
    ae = UNetSmallAE(3, 16).to(device); ae.load_decoder_state(decoder_state); ae.eval()
    config = {'seed': 20261006, 'batch_size': 128}
    loader = resident_loader(images, labels, indices, config, 99, shuffle=False)
    metrics = []
    for cid in range(10):
        ae.load_encoder_state(encoders[str(cid)])
        correct = 0
        for x, y in loader:
            d, _ = ae(x)
            logits = model(x + d)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits')
            correct += int((logits.argmax(1) == y).sum())
        metrics.append({'client_id': cid, 'correct': correct, 'total': len(indices),
                        'accuracy_percent': 100 * correct / len(indices)})
    value = {'pipeline_metrics': metrics,
             'uniform_pipeline_accuracy_percent': float(np.mean([r['accuracy_percent'] for r in metrics])),
             'correct_total': sum(r['correct'] for r in metrics), 'predictions_total': 10 * len(indices)}
    assert_finite(value)
    return value


@torch.no_grad()
def recalibrate_bn(classifier_state, decoder_state, encoders, images, partition, device):
    """Inference-only shared BN buffer variant; fit images, no target labels."""
    saved_rng = rng_state(cuda=device.type == 'cuda')
    model = ResNet20V2(9, 3).to(device); model.load_state_dict(classifier_state)
    ae = UNetSmallAE(3, 16).to(device); ae.load_decoder_state(decoder_state); ae.eval()
    fused = []
    for cid in range(10):
        ae.load_encoder_state(encoders[str(cid)])
        ids = partition['bn_calibration'][str(cid)]
        for begin in range(0, len(ids), 128):
            x = images[torch.tensor(ids[begin:begin+128], device=device)]
            d, _ = ae(x); fused.append(x + d)
    fused = torch.cat(fused)
    permutation = np.random.default_rng(20261006).permutation(len(fused))
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats(); module.momentum = None
    model.train()
    for begin in range(0, len(fused), 128):
        model(fused[torch.tensor(permutation[begin:begin+128], device=device)])
    result = clone_state_dict(model.state_dict())
    for key, value in classifier_state.items():
        if not key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            if not torch.equal(value, result[key]):
                raise AssertionError('BN calibration changed weights')
    assert_finite(result)
    restore_rng(saved_rng)
    return result


def validate_configuration(config, partition):
    plan = json.loads((PUBLIC / 'search_plan.json').read_text())
    if config['candidate_id'] not in plan['candidates'] or config['settings'] != plan['candidates'][config['candidate_id']]:
        raise ValueError('Unregistered/changed candidate')
    if config['partition_sha256'] != partition['partition_sha256']:
        raise ValueError('Changed partition')
    if config['stage'] == 'final':
        selected = json.loads((PUBLIC / 'selection.json').read_text())
        if config['candidate_id'] != selected['selected_candidate_id'] or config['rounds'] != selected['rounds'] or config['bn_mode'] != selected['bn_mode'] or config['seed'] not in selected['final_seeds']:
            raise ValueError('Final run differs from frozen selection')
        if config['selection_sha256'] != file_hash(PUBLIC / 'selection.json'):
            raise ValueError('Changed selection')
    elif config['stage'] not in ('screening', 'confirmation') or config['seed'] not in (142, 143) or config['rounds'] not in (20, 50, 100):
        raise ValueError('Unregistered tuning budget/seed')
    return plan


def run(config, output, physical_device, resume=False):
    started = time.perf_counter()
    os.environ['CUDA_VISIBLE_DEVICES'] = physical_device.split(':')[-1]
    device = torch.device('cuda:0')
    torch.cuda.set_device(device); torch.set_num_threads(2)
    seed_everything(config['seed'])
    partition = verify(); validate_configuration(config, partition)
    if output.exists() and not resume:
        raise FileExistsError('Output exists; preserve previous attempt')
    output.mkdir(parents=True, exist_ok=resume)
    lock = (output / '.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = config['settings']; images, labels = cache(device)
    split = partition['full' if config['stage'] == 'final' else 'fit']
    client_loaders = loaders(images, labels, split, settings, config['seed'])
    classifier = clone_state_dict(ResNet20V2(9, 3).state_dict())
    seed_everything(config['seed'] + 1000000)
    ae = UNetSmallAE(3, 16)
    ae_initial = clone_state_dict(ae.state_dict()); decoder = ae.decoder_state()
    clients = []
    for i in range(10):
        c = CalibratedClient(i, client_loaders[str(i)], settings, device)
        c.set_classifier_state(classifier); c.set_full_autoencoder_state(ae_initial)
        clients.append(c)
    code = identity(); history = []; milestones = []; first = 1; prior_seconds = 0
    if resume:
        old = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        for key in ('settings', 'seed', 'candidate_id', 'partition_sha256', 'stage'):
            if old['configuration'][key] != config[key]:
                raise ValueError('Changed resume scientific identity')
        if old['code']['source_sha256'] != code['source_sha256']:
            raise ValueError('Changed resume sources')
        code = old['code']; classifier, decoder = old['classifier'], old['decoder']
        history, milestones = old['history'], old['milestones']
        prior_seconds = old['training_seconds']
        first = old['round'] + 1
        for client in clients:
            restore_client(client, old['clients'][str(client.client_id)])
        restore_rng(old['rng'])
        if config['stage'] == 'final' and (output / 'results.json').exists():
            raise FileExistsError('Final run already complete')
    encoders = {str(c.client_id): c.encoder_state() for c in clients}
    began = time.perf_counter()
    training_end_seconds = None
    def saved(round_index):
        value = {'format': 'pathmnist-calibrated-v1', 'seed': config['seed'], 'round': round_index,
                 'configuration': config, 'config_sha256': json_hash(config), 'code': code,
                 'partition': partition, 'training_indices': split,
                 'classifier': classifier, 'decoder': decoder, 'encoders': encoders,
                 'clients': {str(c.client_id): client_snapshot(c) for c in clients},
                 'rng': rng_state(cuda=True), 'history': history, 'milestones': milestones,
                 'training_seconds': training_end_seconds if training_end_seconds is not None else prior_seconds + time.perf_counter() - began,
                 'physical_device': physical_device, 'visible_device': 'cuda:0'}
        assert_finite(value); return value
    if not resume:
        atomic_save(output / 'initial.pt', saved(0))
    for round_index in range(first, config['rounds'] + 1):
        if identity()['source_sha256'] != code['source_sha256']:
            raise ValueError('Sources changed during training')
        before = time.perf_counter(); records = []
        for client in clients:
            client.set_classifier_state(classifier); client.set_decoder_state(decoder)
            record = client.train_round_at(round_index)
            record['client_id'] = client.client_id
            records.append(record)
        classifier = weighted_average_states([c.classifier_state() for c in clients], [1] * 10)
        decoder = weighted_average_states([c.decoder_state() for c in clients], [1] * 10)
        encoders = {str(c.client_id): c.encoder_state() for c in clients}
        torch.cuda.synchronize(device)
        row = {'round': round_index, 'seconds': time.perf_counter() - before, 'clients': records,
               'warmup_reconstruction_loss': float(np.mean([r['warmup_reconstruction_loss'] for r in records])),
               'classification_loss': float(np.mean([r['classification_loss'] for r in records]))}
        assert_finite(row); history.append(row)
        atomic_save(output / 'latest.pt', saved(round_index))
        write_json(output / 'progress.json', {'round': round_index, 'target': config['rounds'],
                                            'candidate_id': config['candidate_id'], 'seed': config['seed'],
                                            'wall_seconds': time.perf_counter() - started,
                                            'estimated_remaining_seconds': (time.perf_counter() - began) / (round_index-first+1) * (config['rounds']-round_index)})
        with (output / 'timings.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'round': round_index, 'candidate': config['candidate_id'], 'seed': config['seed'],
                          'classification_loss': row['classification_loss'], 'seconds': row['seconds']}), flush=True)
    for client in clients:
        client.set_classifier_state(classifier); client.set_decoder_state(decoder)
    training_end_seconds = prior_seconds + time.perf_counter() - began
    native = classifier
    selected_classifier = native
    calibrator_start = time.perf_counter()
    bn = recalibrate_bn(native, decoder, encoders, images, partition, device)
    bn_seconds = time.perf_counter() - calibrator_start
    evaluation_started = time.perf_counter()
    if config['stage'] == 'final':
        atomic_save(output / 'native-final.pt', {**saved(config['rounds']),
                                               'note': 'native server BN buffers; final.pt contains frozen selected inference mode'})
        selected_classifier = bn if config['bn_mode'] == 'train-recalibrated' else native
        classifier = selected_classifier
        for c in clients:
            c.set_classifier_state(classifier)
        atomic_save(output / 'final.pt', saved(config['rounds']))
        test_images = torch.from_numpy(np.load(DATA / 'test-images.npy')).to(device)
        test_labels = torch.from_numpy(np.load(DATA / 'test-labels.npy')).to(device)
        evaluations = {config['bn_mode']: evaluate(selected_classifier, decoder, encoders, test_images, test_labels, list(range(7180)), settings, device)}
        evaluation_split = 'test'
    else:
        validation = [i for cid in range(10) for i in partition['validation'][str(cid)]]
        evaluations = {mode: evaluate(state, decoder, encoders, images, labels, validation, settings, device)
                       for mode, state in (('native', native), ('train-recalibrated', bn))}
        evaluation_split = 'training-holdout'
    record = {'round': config['rounds'], 'split': evaluation_split, 'evaluations': evaluations,
              'bn_calibration_seconds': bn_seconds, 'evaluation_seconds': time.perf_counter()-evaluation_started}
    milestones.append(record)
    atomic_save(output / f'round-{config["rounds"]:03d}.pt', saved(config['rounds']))
    atomic_save(output / 'latest.pt', saved(config['rounds']))
    result = {'status': 'completed', 'configuration': config, 'code': code, 'seed': config['seed'],
              'completed_rounds': config['rounds'], 'history': history, 'milestones': milestones,
              'training_seconds': training_end_seconds,
              'session_seconds': time.perf_counter()-started, 'resumed': resume,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'runtime': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(device),
                          'cudnn_deterministic': torch.backends.cudnn.deterministic,
                          'cudnn_benchmark': torch.backends.cudnn.benchmark},
              'checkpoint_sha256': {p.name:file_hash(p) for p in output.glob('*.pt') if p.name!='latest.pt'}}
    assert_finite(result)
    write_json(output / f'results-round-{config["rounds"]:03d}.json', result)
    write_json(output / 'results.json', result)
    print(json.dumps({'status':'completed','candidate':config['candidate_id'],'seed':config['seed'],
                      'rounds':config['rounds'],'evaluations':{k:v['uniform_pipeline_accuracy_percent'] for k,v in evaluations.items()}}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.device, args.resume)
