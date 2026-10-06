"""Fixed, globally disjoint train-derived holdout; no test access in tuning."""
from pathlib import Path
import gzip
import json
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.pathmnist_pathological.run import file_hash, json_hash, write_json, resident_loader

PUBLIC = ROOT / 'research/pathmnist_calibrated'
PRIVATE = ROOT / '_local/pathmnist_calibrated'
DATA = ROOT / '_local/pathmnist_pathological/prepared'


def prepare():
    destination = PUBLIC / 'partition.json.gz'
    if destination.exists():
        return verify()
    source = json.loads((DATA / 'manifest.json').read_text())
    for name in ('train-images.npy', 'train-labels.npy', 'partitions.json'):
        if file_hash(DATA / name) != source['files_sha256'][name]:
            raise ValueError('Changed original training data')
    parts = json.loads((DATA / 'partitions.json').read_text())['clients']
    labels = np.load(DATA / 'train-labels.npy')
    rng = np.random.default_rng(20261006)
    fit, validation, bn, stats = {}, {}, {}, []
    for key, indices in parts.items():
        fit[key], validation[key] = [], []
        for cls in np.unique(labels[indices]):
            rows = np.array([i for i in indices if labels[i] == cls])
            rng.shuffle(rows)
            count = max(1, len(rows) // 10)
            validation[key] += rows[:count].tolist()
            fit[key] += rows[count:].tolist()
        rng.shuffle(fit[key]); rng.shuffle(validation[key])
        bn[key] = np.random.default_rng(20261006 + int(key)).choice(fit[key], 640, replace=False).tolist()
        stats.append({'client_id': int(key), 'fit': len(fit[key]), 'validation': len(validation[key]),
                      'classes': np.unique(labels[fit[key]]).tolist()})
    value = {'data_seed': 42, 'holdout_seed': 20261006,
             'source_partition_sha256': source['partition_sha256'],
             'source_train_sha256': source['files_sha256']['train-images.npy'],
             'source_labels_sha256': source['files_sha256']['train-labels.npy'],
             'index_space': 'official PathMNIST-64 training array row',
             'fit': fit, 'validation': validation, 'full': parts, 'bn_calibration': bn,
             'statistics': stats,
             'evaluation': 'Every private encoder on pooled holdout; uniform mean of ten pipeline accuracies',
             'bn_note': '640 fixed fit-only images/client; shared classifier buffers re-estimated on shuffled fused mixture; no parameter updates'}
    value['partition_sha256'] = json_hash(value)
    destination.write_bytes(gzip.compress((json.dumps(value, indent=2)+'\n').encode(), mtime=0))
    return verify()


def verify():
    value = json.loads(gzip.decompress((PUBLIC / 'partition.json.gz').read_bytes()))
    if json_hash({k: v for k, v in value.items() if k != 'partition_sha256'}) != value['partition_sha256']:
        raise ValueError('Changed holdout manifest')
    fit = [i for row in value['fit'].values() for i in row]
    val = [i for row in value['validation'].values() for i in row]
    full = [i for row in value['full'].values() for i in row]
    if len(set(fit)) != len(fit) or len(set(val)) != len(val) or set(fit) & set(val) or sorted(fit + val) != list(range(89996)) or sorted(full) != list(range(89996)):
        raise ValueError('Invalid global split')
    labels = np.load(DATA / 'train-labels.npy')
    for key in value['fit']:
        if len(np.unique(labels[value['fit'][key]])) != 2 or len(np.unique(labels[value['validation'][key]])) != 2:
            raise ValueError('Client lost one class')
        if len(value['bn_calibration'][key]) != 640 or not set(value['bn_calibration'][key]) <= set(value['fit'][key]):
            raise ValueError('BN calibration is not fit-only')
    return value


def cache(device):
    return (torch.from_numpy(np.load(DATA / 'train-images.npy')).to(device),
            torch.from_numpy(np.load(DATA / 'train-labels.npy')).to(device))


def loaders(images, labels, parts, settings, seed):
    config = {'batch_size': settings['batch_size'], 'seed': seed}
    return {key: resident_loader(images, labels, rows, config, int(key)) for key, rows in parts.items()}


if __name__ == '__main__':
    p = prepare()
    print(json.dumps({'partition_sha256': p['partition_sha256'], 'statistics': p['statistics']}, indent=2))
