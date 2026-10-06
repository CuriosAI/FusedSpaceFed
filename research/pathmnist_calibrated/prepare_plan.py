"""Predeclare all candidate settings, budgets and validation-only decisions."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.pathmnist_calibrated.data import PUBLIC, PRIVATE, prepare
from research.pathmnist_pathological.run import file_hash, write_json


def base(c=0.03, ae=0.0003, **extra):
    value = {'classifier_lr': c, 'autoencoder_lr': ae, 'warmup_lr': ae,
             'gradient_clip_norm': 5.0, 'precision': 'bf16', 'schedule': 'constant',
             'schedule_horizon': 100, 'batch_size': 128, 'warmup_epochs': 1,
             'local_epochs': 3, 'aggregation': 'uniform', 'optimizer_reset': False,
             'classifier': 'ResNet20V2', 'autoencoder': 'UNetSmallAE', 'dz': 16,
             'classifier_optimizer': 'SGD(momentum=0,weight_decay=0)',
             'autoencoder_optimizer': 'Adam(betas=(0.9,0.999),eps=1e-8,weight_decay=0)'}
    value.update(extra); return value


def queue(jobs):
    names = ['fusedspacefed_core.py', 'research/pathmnist_pathological/run.py',
             'research/pathmnist_calibrated/data.py', 'research/pathmnist_calibrated/client.py',
             'research/pathmnist_calibrated/runner.py', 'research/pathmnist_calibrated/search_plan.json',
             'research/pathmnist_calibrated/partition.json.gz']
    names += [j['args'][1] for j in jobs]
    return {'gpu_slots': {'cuda:1': 3, 'cuda:0': 3}, 'minimum_free_mib': 8192,
            'frozen_files': {name: file_hash(ROOT/name) for name in names}, 'jobs': jobs}


def job(candidate, seed, stage, settings, partition_sha, rounds):
    config = {'candidate_id': candidate, 'seed': seed, 'stage': stage, 'settings': settings,
              'rounds': rounds, 'partition_sha256': partition_sha}
    config_path = PUBLIC / 'configs' / stage / f'{candidate}-seed-{seed}-r{rounds}.json'
    config_path.parent.mkdir(parents=True, exist_ok=True); write_json(config_path, config)
    output = PRIVATE / stage / f'{candidate}-seed-{seed}'
    return {'name': f'{candidate}-seed-{seed}-r{rounds}', 'script': 'research/pathmnist_calibrated/runner.py',
            'args': ['--config', str(config_path.relative_to(ROOT))], 'output': str(output.relative_to(ROOT))}


def prepare_plan():
    if (PUBLIC/'search_plan.json').exists():
        raise FileExistsError('Search plan already exists')
    p = prepare()
    candidates = {'paper-reference': base(0.01, 0.001, precision='fp16', gradient_clip_norm=None),
                  'fp32-matched-reference': base(0.01, 0.001, precision='fp32', gradient_clip_norm=None)}
    for c in (0.003, 0.01, 0.03, 0.1):
        for ae in (0.0001, 0.0003, 0.001):
            candidates[f'grid-c{c:g}-ae{ae:g}'] = base(c, ae)
    for clip in (1.0, 10.0):
        candidates[f'clip-{clip:g}'] = base(gradient_clip_norm=clip)
    for c in (0.03, 0.1):
        candidates[f'cosine-c{c:g}'] = base(c, schedule='cosine')
        candidates[f'local1-c{c:g}'] = base(c, local_epochs=1)
    candidates['warm-lr-1e-4'] = base(warmup_lr=0.0001)
    candidates['bf16-no-clip'] = base(gradient_clip_norm=None)
    candidates['fp32-clip5'] = base(precision='fp32')
    candidates['local1-c0.3'] = base(0.3, local_epochs=1)
    plan = {'schema': 1, 'candidates': candidates, 'screening_seed': 142, 'screening_rounds': 20,
            'confirmation_seeds': [142,143], 'confirmation_rounds': 50,
            'shortlist_count': 3, 'mandatory_confirmation_reference': 'paper-reference',
            'extension': {'top_distinct_candidates': 2, 'terminal_rounds': 100,
                          'reuse': 'resume each complete 50-round confirmation; unchanged settings, optimizers/RNG and cosine horizon 100'},
            'inference_modes': ['native','train-recalibrated'], 'final_seeds': [41,42,43,44,45],
            'fixed_partition_sha256': p['partition_sha256'],
            'selection_rule': 'highest mean terminal pooled-holdout uniform-pipeline accuracy over the two confirmation seeds; compare 50-round candidates and the top-two extensions at 100; ties prefer fewer rounds, native BN, then candidate ID',
            'screening_rule': 'rank distinct training candidates by their best declared inference mode at round20; confirm top three plus paper reference, fresh starts',
            'method': 'private persistent encoder, shared decoder/classifier, additive fusion, encoder-only MSE warm-up and joint CE only; no architecture changes',
            'variant': 'train-recalibrated BN re-estimates only shared classifier running buffers on 640 fit images/client; no labels, gradients, parameter/optimizer changes or test input',
            'previous_test_exposure': 'Old seed42 paper-settings score39.619777% was already observed before this task; no new test evaluation before the frozen final campaign',
            'budget': '24 x20-round screening; at most4 x2 x50 confirmations; top2 x2 x50 added rounds; five fresh full-training final seeds with one selected50/100-round setting',
            'stopping': 'no tuning after first final test; failures retained, no best-seed or best-checkpoint selection; diagnostics/ablations remain suspended'}
    write_json(PUBLIC/'search_plan.json',plan)
    jobs=[job(k,142,'screening',s,p['partition_sha256'],20) for k,s in candidates.items()]
    write_json(PUBLIC/'screening_queue.json',queue(jobs))
    print(json.dumps({'candidates':len(candidates),'fit':sum(len(x) for x in p['fit'].values()),
                      'holdout':sum(len(x) for x in p['validation'].values()),'partition':p['partition_sha256']},indent=2))


if __name__=='__main__':
    prepare_plan()
