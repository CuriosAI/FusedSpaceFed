"""Fit-only last-layer refit; a declared extra shared-classifier phase."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from fusedspacefed_core import UNetSmallAE, seed_everything
from research.pathmnist_pathological.run import file_hash, write_json, atomic_save, assert_finite, cpu_copy
from research.pathmnist_calibrated.data import verify, cache
from research.pathmnist_calibrated import runner
from research.pathmnist_recovery.method import build_classifier
from research.pathmnist_recovery.training import adapted
from research.pathmnist_recovery.fusion_probe import scaled_decoder

SOURCES = ('research/pathmnist_recovery/head_calibration.py',
           'research/pathmnist_recovery/method.py', 'research/pathmnist_recovery/training.py',
           'research/pathmnist_recovery/fusion_probe.py', 'research/pathmnist_calibrated/runner.py',
           'research/pathmnist_calibrated/data.py', 'fusedspacefed_core.py')


def weighted_ce(logits, labels, client_ids):
    """Exact mean of ten local full-batch CE objectives (uniform clients)."""
    losses = F.cross_entropy(logits, labels, reduction='none')
    return torch.stack([losses[client_ids == cid].mean()
                        for cid in torch.unique(client_ids)]).mean()


@torch.no_grad()
def features(classifier, decoder, encoders, images, labels, split, settings, device):
    """Each fit image uses its owner's encoder; no foreign-client training."""
    model = build_classifier(settings).to(device)
    model.load_state_dict(classifier); model.fc = nn.Identity(); model.eval()
    ae = UNetSmallAE(3, 16).to(device); ae.load_decoder_state(decoder); ae.eval()
    xx, yy, owners = [], [], []
    for cid in range(10):
        ae.load_encoder_state(encoders[str(cid)])
        ids = split[str(cid)]
        for begin in range(0, len(ids), 128):
            batch = torch.tensor(ids[begin:begin+128], device=device)
            x = images[batch]; d, _ = ae(x)
            xx.append(model(x+d)); yy.append(labels[batch])
            owners.append(torch.full((len(batch),), cid, device=device, dtype=torch.long))
    return torch.cat(xx), torch.cat(yy), torch.cat(owners)


def refit(classifier, xx, yy, owners, penalty, max_iter=100):
    """Convex CE + L2 of shared fc; all representations are detached/fixed.

    Each closure is mathematically the mean of local losses/gradients. This
    diagnostic implementation caches owner features in the simulator; it
    does not claim privacy or deployment equivalence to ordinary FedAvg.
    """
    if penalty < 0 or not torch.isfinite(torch.tensor(penalty)):
        raise ValueError('Invalid L2 penalty')
    xx = xx.detach(); yy = yy.detach(); owners = owners.detach()
    head = nn.Linear(xx.shape[1], classifier['fc.weight'].shape[0]).to(xx.device)
    head.load_state_dict({k: v.to(xx.device) for k, v in
                         (('weight', classifier['fc.weight']), ('bias', classifier['fc.bias']))})
    optimizer = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=max_iter,
                                  tolerance_grad=1e-7, tolerance_change=1e-10,
                                  history_size=20, line_search_fn='strong_wolfe')
    history = []
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_ce(head(xx), yy, owners) + .5*penalty*head.weight.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite shared-head objective')
        loss.backward(); history.append(float(loss.detach())); return loss
    optimizer.step(closure)
    state = copy.deepcopy(classifier)
    state['fc.weight'] = head.weight.detach().cpu().clone()
    state['fc.bias'] = head.bias.detach().cpu().clone()
    with torch.no_grad():
        final_ce = float(weighted_ce(head(xx), yy, owners))
    statistics = {'penalty': penalty, 'max_iter': max_iter,
                  'closure_evaluations': len(history), 'objective_history': history,
                  'final_uniform_client_training_ce': final_ce,
                  'head_weight_norm': float(head.weight.detach().norm())}
    assert_finite(state); assert_finite(statistics)
    # Optimizer state is retained separately, with the actual refitted head.
    return state, cpu_copy(optimizer.state_dict()), statistics


def main(config_path, output, physical_device):
    started = time.perf_counter()
    os.environ['CUDA_VISIBLE_DEVICES'] = physical_device.split(':')[-1]
    device = torch.device('cuda:0'); torch.cuda.set_device(device); torch.set_num_threads(2)
    seed_everything(142)
    config = json.loads(config_path.read_text()); p = verify(); source = ROOT/config['checkpoint']
    if file_hash(source) != config['checkpoint_sha256']:
        raise ValueError('Changed source checkpoint')
    state = torch.load(source, map_location='cpu', weights_only=False); assert_finite(state)
    if state['seed'] != 142 or state['round'] != 50 or state['training_indices'] != p['fit']:
        raise ValueError('Head selection must use independent fit-only round50 state')
    if config['gain'] <= 0 or config['mode'] not in ('native', 'train-recalibrated'):
        raise ValueError('Ineligible fusion/inference mode')
    output.mkdir(parents=True, exist_ok=False)
    images, labels = cache(device); settings = state['configuration']['settings']
    decoder = scaled_decoder(state['decoder'], config['gain'])
    validation_ids = [i for cid in range(10) for i in p['validation'][str(cid)]]
    with adapted(settings):
        classifier = state['classifier'] if config['mode'] == 'native' else runner.recalibrate_bn(
            state['classifier'], decoder, state['encoders'], images, p, device)
        baseline = runner.evaluate(classifier, decoder, state['encoders'], images, labels,
                                   validation_ids, settings, device)
        xx, yy, owners = features(classifier, decoder, state['encoders'], images, labels,
                                  p['fit'], settings, device)
        rows = []
        for penalty in config['penalties']:
            before = time.perf_counter()
            fitted, optimizer, statistics = refit(classifier, xx, yy, owners,
                                                  penalty, config['max_iter'])
            metric = runner.evaluate(fitted, decoder, state['encoders'], images, labels,
                                     validation_ids, settings, device)
            name = 'penalty-'+str(penalty)+'.pt'
            atomic_save(output/name, {'classifier': fitted, 'head_optimizer': optimizer,
                                      'statistics': statistics, 'configuration': config,
                                      'source_checkpoint_sha256': file_hash(source)})
            rows.append({'penalty': penalty, 'metrics': metric, 'statistics': statistics,
                         'seconds': time.perf_counter()-before, 'head_checkpoint': name})
            print(json.dumps({'penalty': penalty, 'validation_accuracy_percent':
                               metric['uniform_pipeline_accuracy_percent']}), flush=True)
    result = {'status': 'completed', 'kind': 'extra shared fc calibration; frozen C body/E/D',
              'split': 'training-holdout', 'seed': 142, 'round': 50, 'configuration': config,
              'baseline': baseline, 'rows': rows, 'fit_feature_samples': len(xx),
              'objective': 'mean of local client CE, uniform client weights; L2 on fc weight, not bias',
              'session_seconds': time.perf_counter()-started, 'physical_device': physical_device,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'code': {'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                       'source_sha256': {n: file_hash(ROOT/n) for n in SOURCES},
                       'config_sha256': file_hash(config_path)},
              'head_checkpoints_sha256': {x.name:file_hash(x) for x in output.glob('*.pt')}}
    assert_finite(result); write_json(output/'results.json', result)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--config',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--device',required=True); args = a.parse_args()
    main(args.config,args.output,args.device)
