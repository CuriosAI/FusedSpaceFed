"""Frozen positive-fusion/BN inference on an immutable complete checkpoint."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from fusedspacefed_core import seed_everything
from research.pathmnist_pathological.run import file_hash, write_json, atomic_save, assert_finite
from research.pathmnist_calibrated.data import verify, cache, DATA
from research.pathmnist_calibrated.runner import evaluate, recalibrate_bn
from research.pathmnist_recovery.normalization import calibrate
from research.pathmnist_recovery.fusion_probe import scaled_decoder


def inference_decoder(checkpoint):
    """Apply the stored gain without changing trained weights/optimizer states.

    The top-level decoder is always the original trained state. Consumers of
    a recovery checkpoint must use this function (or multiply its output by
    recovery.config.fusion_gain) to reproduce its declared inference.
    """
    gain = float(checkpoint.get('recovery', {}).get('config', {}).get('fusion_gain', 1.0))
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('A FusedSpaceFed inference gain must be finite and positive')
    return scaled_decoder(checkpoint['decoder'], gain)


def substitute_shared_bn(checkpoint, classifier):
    """Copy complete state; changing classifier weights is an error."""
    original = checkpoint['classifier']
    if set(original) != set(classifier):
        raise ValueError('Different classifier architecture')
    for key in original:
        if not key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            if not torch.equal(original[key], classifier[key]):
                raise ValueError('Inference calibration changed a weight')
    result = copy.deepcopy(checkpoint)
    result['classifier'] = classifier
    for client in result['clients'].values():
        client['classifier'] = copy.deepcopy(classifier)
    assert_finite(result)
    return result


def main(config_path, output, physical_device):
    started = time.perf_counter()
    os.environ['CUDA_VISIBLE_DEVICES'] = physical_device.split(':')[-1]
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    config = json.loads(config_path.read_text())
    source = ROOT / config['checkpoint']
    if file_hash(source) != config['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed')
    partition = verify()
    if partition['partition_sha256'] != config['partition_sha256']:
        raise ValueError('Partition changed')
    if config['normalization'] not in ('owner-cumulative-shared-bn','native',
            'owner-cumulative','owner-layerwise','cross-cumulative','cross-layerwise'):
        raise ValueError('Unregistered inference mode')
    if config['test_access'] != 'one frozen candidate; known benchmark; exploratory threshold stop':
        raise ValueError('Undeclared test access')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    assert_finite(checkpoint)
    if checkpoint['seed'] != 42 or checkpoint['round'] != 50:
        raise ValueError('Wrong seed/round')
    if set(checkpoint['encoders']) != {str(i) for i in range(10)}:
        raise ValueError('Incomplete private states')
    decoder = inference_decoder({**checkpoint, 'recovery': {'config': config}})
    seed_everything(checkpoint['seed'])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / 'precalibration.pt')
    images, labels = cache(device)
    before = time.perf_counter()
    if config['normalization']=='owner-cumulative-shared-bn':
        classifier = recalibrate_bn(checkpoint['classifier'], decoder,
                                    checkpoint['encoders'], images, partition, device)
    else:
        classifier = calibrate(checkpoint['classifier'], decoder,
                               checkpoint['encoders'], images, partition, device,
                               config['normalization'])
    calibrated = substitute_shared_bn(checkpoint, classifier)
    code = {'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256': {name: file_hash(ROOT/name) for name in
                ('research/pathmnist_recovery/inference.py', 'research/pathmnist_recovery/normalization.py',
                 'research/pathmnist_recovery/fusion_probe.py', 'research/pathmnist_recovery/training.py',
                 'research/pathmnist_recovery/method.py',
                 'research/pathmnist_calibrated/runner.py',
                 'research/pathmnist_calibrated/data.py', 'fusedspacefed_core.py')},
            'config_sha256': file_hash(config_path)}
    calibrated['recovery'] = {'config': config, 'code': code,
                              'original_checkpoint_sha256': file_hash(source),
                              'training_weights_and_optimizer_states_unchanged': True,
                              'decoder_representation': 'original trained weights; use inference_decoder() for configured positive gain',
                              'exact_training_resume_source': str(output / 'precalibration.pt')}
    atomic_save(output / 'final.pt', calibrated)
    calibration_seconds = time.perf_counter()-before
    test_images = torch.from_numpy(np.load(DATA/'test-images.npy')).to(device)
    test_labels = torch.from_numpy(np.load(DATA/'test-labels.npy')).to(device)
    evaluated = time.perf_counter()
    metric = evaluate(classifier, inference_decoder(calibrated), checkpoint['encoders'],
                      test_images, test_labels, list(range(7180)), {}, device)
    result = {'status': 'completed', 'kind': 'inference-only variant; no new training',
              'seed': 42, 'round': 50, 'test_evaluations': 1, 'config': config, 'code': code,
              'metrics': metric, 'above_target': metric['uniform_pipeline_accuracy_percent'] > 50.94,
              'normalization_seconds': calibration_seconds,
              'evaluation_seconds': time.perf_counter()-evaluated,
              'session_seconds': time.perf_counter()-started,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'physical_device': physical_device, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_sha256': {p.name:file_hash(p) for p in output.glob('*.pt')}}
    assert_finite(result)
    write_json(output/'results.json', result)
    print(json.dumps({'status': 'completed', 'accuracy_percent': metric['uniform_pipeline_accuracy_percent'],
                      'above_target':result['above_target'], 'session_seconds':result['session_seconds']}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True)
    a=p.parse_args();main(a.config,a.output,a.device)
