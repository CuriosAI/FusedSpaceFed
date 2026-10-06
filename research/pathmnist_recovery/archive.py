"""Lossless numeric archives for completed PathMNIST recovery steps."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.pathmnist_pathological.run import file_hash, write_json, assert_finite
from research.pathmnist_calibrated.archive import compressed, manifest

PUBLIC = ROOT/'research/pathmnist_recovery/artifacts'
PRIVATE = ROOT/'_local/pathmnist_recovery'


def verify():
    for directory in sorted(PUBLIC.iterdir()):
        if not directory.is_dir(): continue
        info = json.loads((directory/'manifest.json').read_text())
        actual={str(p.relative_to(directory)) for p in directory.rglob('*')
                if p.is_file() and p.name!='manifest.json'}
        if actual!=set(info['files_sha256']):
            raise ValueError('Missing/unlisted file in numeric archive '+str(directory))
        for name, digest in info['files_sha256'].items():
            if file_hash(directory/name) != digest:
                raise ValueError('Changed numeric archive '+str(directory/name))
    print('All recovery archive hashes verified')


def numeric_archive(name, receipt_name, input_root, patterns):
    receipt_path = PRIVATE/receipt_name
    if not receipt_path.exists(): return
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] == 'running': return
    if any(r['exit_code'] != 0 for r in receipt['runs']):
        raise ValueError('This archive requires complete successful worker processes')
    out = PUBLIC/name
    if out.exists(): return
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(receipt_path,out/'campaign.json')
    files = sorted({p for pattern in patterns for p in input_root.glob(pattern) if p.is_file()})
    if not files: raise ValueError('No numeric outputs to archive')
    provenance = {'source_files': {}, 'private_checkpoints': {}}
    for source in files:
        relative = source.relative_to(input_root)
        target = out/(str(relative)+'.gz'); target.parent.mkdir(parents=True, exist_ok=True)
        compressed(source,target)
        provenance['source_files'][str(source.relative_to(ROOT))] = {
            'sha256':file_hash(source),'archived_as':str(target.relative_to(out)),
            'bytes':source.stat().st_size}
        if source.suffix == '.json': assert_finite(json.loads(source.read_text()))
    for source in sorted(input_root.rglob('*.pt')):
        provenance['private_checkpoints'][str(source.relative_to(ROOT))] = {
            'sha256':file_hash(source),'bytes':source.stat().st_size}
    write_json(out/'provenance.json',provenance); manifest(out)


def archive():
    for attempt in range(3,10):
        numeric_archive('attempt-'+f'{attempt:02}', 'attempt-'+f'{attempt:02}'+'-campaign.json',
                        PRIVATE/('attempt-'+f'{attempt:02}'), ['results.json','verification.json'])
    for name, receipt, directory, patterns in (
        ('continuation-validation','continuation_campaign.json','continuation_validation',
         ['*/results*.json','*/timings.jsonl','*/continuation_origin.json']),
        ('groupnorm-validation','groupnorm_campaign.json','groupnorm_validation',
         ['*/results*.json','*/timings.jsonl']),
        ('fusion-validation','fusion_campaign.json','fusion_validation',['*/results.json']),
        ('head-validation','head_campaign.json','head_validation',['*/results.json']),
    ):
        numeric_archive(name,receipt,PRIVATE/directory,patterns)
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('archive','verify'))
    args = parser.parse_args(); {'archive':archive,'verify':verify}[args.mode]()
