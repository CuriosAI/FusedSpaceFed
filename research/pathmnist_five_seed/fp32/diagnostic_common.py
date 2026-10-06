"""FP32 anchors and paths; reuse the fixed historical training probe and metrics."""
import json
import subprocess
import torch
from research.pathmnist_five_seed.fp32.precision import ROOT, PUBLIC, PRIVATE
from research.pathmnist_five_seed.data import partition
from research.pathmnist_five_seed.diagnostic_common import (
    ANCHORS, STAGES, tensor_metrics, measure, component_states, deltas, stats,
    state_hash, state_distance,
)
from research.pathmnist_five_seed.fp32.runner import SOURCES
from research.pathmnist_pathological.run import file_hash, write_json, assert_finite


def run_path(seed, variant='full'):
    return PRIVATE / variant / f'seed-{seed}'


def prepare_probes():
    original = ROOT / 'research/pathmnist_five_seed/probe.json'
    if file_hash(PUBLIC / 'probe.json') != file_hash(original):
        raise ValueError('The FP32 campaign must reuse the exact fixed training probes')


def freeze_anchors():
    seeds = {}
    for seed in range(42, 47):
        parts, spec = partition(seed)
        anchors = {}
        for name, filename, number in zip(ANCHORS, ('initial.pt', 'final.pt'), (0, 50)):
            path = run_path(seed) / filename
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert_finite(state)
            if state['seed'] != seed or state['round'] != number or state['partitions'] != parts:
                raise ValueError('Wrong FP32 anchor')
            if state['configuration']['settings']['use_amp'] is not False:
                raise ValueError('AMP anchor is forbidden')
            if set(state['clients']) != {str(i) for i in range(10)} or len(state['encoders']) != 10:
                raise ValueError('Incomplete FP32 clients')
            for cid, c in state['clients'].items():
                if c['scaler'] is not None:
                    raise ValueError('Unexpected scaler')
                if state_hash(c['classifier']) != state_hash(state['classifier']) or state_hash(c['decoder']) != state_hash(state['decoder']):
                    raise ValueError('Unsynchronized shared state')
                if state_hash(c['encoder']) != state_hash(state['encoders'][cid]):
                    raise ValueError('Wrong private encoder')
            anchors[name] = {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path),
                             'round': number, 'partition_sha256': spec['sha256'], 'bytes': path.stat().st_size}
        seeds[str(seed)] = anchors
    value = {'precision': 'FP32', 'seeds': seeds}
    path = PUBLIC / 'anchors.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Changed frozen FP32 anchors')
    write_json(path, value)


def load_anchor(seed, anchor):
    spec = json.loads((PUBLIC / 'anchors.json').read_text())['seeds'][str(seed)][anchor]
    path = ROOT / spec['path']
    if file_hash(path) != spec['sha256']:
        raise ValueError('Changed FP32 anchor')
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert_finite(state)
    return state, spec


def probe(seed):
    value = json.loads((PUBLIC / 'probe.json').read_text())['seeds'][str(seed)]
    if value['partition_sha256'] != partition(seed)[1]['sha256']:
        raise ValueError('Wrong paired training probe')
    return value


def source_identity(extra=()):
    names = (*SOURCES, 'research/digits_mechanism_diagnostics/common.py',
             'research/pathmnist_five_seed/diagnostic_common.py',
             'research/pathmnist_five_seed/fp32/diagnostic_common.py',
             'research/pathmnist_five_seed/fp32/anchors.json',
             'research/pathmnist_five_seed/fp32/probe.json', *extra)
    return {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {name: file_hash(ROOT / name) for name in names}}
