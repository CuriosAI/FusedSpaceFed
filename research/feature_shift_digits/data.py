"""Use author-distributed part0/test files, with author PIL preprocessing.

No new sampling or train/test splitting. Cache only the 743 balanced part0
rows and each full author test. All arrays and source files remain private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DOMAINS = ('MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M')
DIRECTORIES = {'MNIST-M': 'MNIST_M'}
ARCHIVE_SHA256 = '6c006e41ce16404aab520895a5c510166453c8b58ed2e7eb7e23e133e1fa4221'


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def preprocessing(domain: str):
    stages = []
    if domain in ('SVHN', 'USPS', 'SynthDigits'):
        stages.append(transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR))
    if domain in ('MNIST', 'USPS'):
        stages.append(transforms.Grayscale(num_output_channels=3))
    return transforms.Compose(stages)


def preprocess_images(images: np.ndarray, domain: str) -> np.ndarray:
    transform = preprocessing(domain)
    result = np.empty((len(images), 3, 28, 28), dtype=np.uint8)
    for index, source in enumerate(images):
        image = transform(Image.fromarray(source))
        array = np.asarray(image)
        if array.shape != (28, 28, 3):
            raise ValueError(f'Unexpected preprocessed shape {domain}: {array.shape}')
        result[index] = array.transpose(2, 0, 1)
    return result


def prepare(archive: Path, output: Path) -> dict:
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ValueError('Dataset ZIP does not match the pinned author-linked mirror')
    if output.exists():
        return verify(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 1, 'kind': 'fedbn_balanced_digits_part0_v1',
                'archive_sha256': ARCHIVE_SHA256,
                'sampling': 'author train_part0 exactly; no resampling',
                'normalization': 'Float32: ((uint8 / 255) - 0.5) / 0.5',
                'domains': {}, 'files': {}}
    with zipfile.ZipFile(archive) as source:
        for domain in DOMAINS:
            directory = DIRECTORIES.get(domain, domain)
            record = {}
            for split in ('train', 'test'):
                member = f'{directory}/partitions/train_part0.pkl' if split == 'train' else f'{directory}/test.pkl'
                payload = source.read(member)
                # Only parse the checksum-pinned archive identified by the authors' README.
                images, labels = pickle.loads(payload)
                images, labels = np.asarray(images), np.asarray(labels).reshape(-1)
                if images.dtype != np.uint8 or len(images) != len(labels) or not np.isin(labels, np.arange(10)).all():
                    raise ValueError(f'Invalid source data: {member}')
                if split == 'train' and len(images) != 743:
                    raise ValueError(f'Author part0 must already contain 743 rows: {member}, {len(images)}')
                prepared = preprocess_images(images, domain)
                prefix = f'{directory}-{split}'
                np.save(output / f'{prefix}-images.npy', prepared, allow_pickle=False)
                np.save(output / f'{prefix}-labels.npy', labels.astype(np.int64), allow_pickle=False)
                ids = [f'{domain}:{member}:{index}' for index in range(len(images))]
                atomic_json(output / f'{prefix}-ids.json', ids)
                record[split] = {'count': len(labels), 'labels': np.bincount(labels.astype(np.int64), minlength=10).tolist(),
                                 'source_member': member, 'source_member_sha256': hashlib.sha256(payload).hexdigest(),
                                 'source_shape': list(images.shape), 'prepared_shape': list(prepared.shape),
                                 'origin_identifier': 'domain + archive member + row; raw pre-resplit IDs not supplied'}
            manifest['domains'][domain] = record
    for path in sorted(output.iterdir()):
        manifest['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
    manifest['partition_sha256'] = canonical_hash(manifest)
    atomic_json(output / 'manifest.json', manifest)
    return verify(output)


def verify(output: Path) -> dict:
    manifest = json.loads((output / 'manifest.json').read_text())
    expected = manifest.pop('partition_sha256')
    if canonical_hash(manifest) != expected:
        raise ValueError('Prepared data manifest identity differs')
    manifest['partition_sha256'] = expected
    if list(manifest['domains']) != sorted(DOMAINS) or manifest['archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('Wrong domain set or source archive')
    for name, identity in manifest['files'].items():
        path = output / name
        if path.stat().st_size != identity['bytes'] or file_hash(path) != identity['sha256']:
            raise ValueError(f'Prepared data file changed: {name}')
    for domain in DOMAINS:
        directory = DIRECTORIES.get(domain, domain)
        train_ids = json.loads((output / f'{directory}-train-ids.json').read_text())
        test_ids = json.loads((output / f'{directory}-test-ids.json').read_text())
        if len(train_ids) != 743 or len(set(train_ids)) != 743 or len(test_ids) != len(set(test_ids)) or set(train_ids) & set(test_ids):
            raise ValueError('Source row identifiers are duplicated or train/test overlap')
    return manifest


class PreparedDigits(Dataset):
    def __init__(self, root: Path, domain: str, split: str):
        if domain not in DOMAINS or split not in ('train', 'test'):
            raise ValueError('Unknown domain/split')
        directory = DIRECTORIES.get(domain, domain)
        self.images = np.load(root / f'{directory}-{split}-images.npy', mmap_mode='r', allow_pickle=False)
        self.labels = np.load(root / f'{directory}-{split}-labels.npy', mmap_mode='r', allow_pickle=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tensor = torch.from_numpy(self.images[index].copy()).float().div_(255).sub_(0.5).div_(0.5)
        return tensor, int(self.labels[index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.archive, args.output)
    print(json.dumps({'partition_sha256': result['partition_sha256'], 'domains': result['domains']}, indent=2))
