"""Original per-seed pathological partitions over the immutable native cache."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from fusedspacefed_core import pathological_partition
from research.pathmnist_pathological.run import file_hash,json_hash,write_json,resident_loader

PUBLIC=ROOT/'research/pathmnist_five_seed'
PRIVATE=ROOT/'_local/pathmnist_five_seed'
DATA=ROOT/'_local/pathmnist_pathological/prepared'


def verify_cache():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        if file_hash(DATA/name)!=digest:raise ValueError('Changed native cache '+name)
    return manifest


def make_partition(labels,seed,clients=10):
    parts=pathological_partition(labels,clients,2,seed)
    value={'seed':seed,'index_space':'official PathMNIST-64 train array row',
           'clients':{str(i):rows for i,rows in enumerate(parts)}}
    return value


def prepare():
    source=verify_cache();labels=np.load(DATA/'train-labels.npy')
    folder=PRIVATE/'partitions';folder.mkdir(parents=True,exist_ok=True)
    specs={}
    for seed in range(42,47):
        value=make_partition(labels,seed);path=folder/f'seed-{seed}.json'
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError('Changed frozen partition')
        else:write_json(path,value)
        if seed==42 and file_hash(path)!=source['partition_sha256']:
            raise ValueError('Seed42 does not reproduce original partition exactly')
        specs[str(seed)]={'path':str(path.relative_to(ROOT)),'sha256':file_hash(path),
                         'clients':[{'client_id':int(cid),'samples':len(ids),'classes':np.unique(labels[ids]).tolist(),
                                     'class_counts':np.bincount(labels[ids],minlength=9).tolist()}
                                    for cid,ids in value['clients'].items()]}
    write_json(PUBLIC/'data_identity.json',{'cache_manifest':source,'partition_seed_rule':'run seed, same original pathological_partition recipe',
                                          'partitions':specs})
    return specs


def partition(seed):
    manifest=json.loads((PUBLIC/'data_identity.json').read_text());spec=manifest['partitions'][str(seed)]
    path=ROOT/spec['path']
    if file_hash(path)!=spec['sha256']:raise ValueError('Changed frozen indices')
    value=json.loads(path.read_text());flat=[i for ids in value['clients'].values() for i in ids]
    if sorted(flat)!=list(range(89996)):raise ValueError('Incomplete/reused examples')
    labels=np.load(DATA/'train-labels.npy')
    if any(len(np.unique(labels[ids]))!=2 for ids in value['clients'].values()):
        raise ValueError('A client does not own exactly two classes')
    return value,spec


def cache(device):
    return torch.from_numpy(np.load(DATA/'train-images.npy')).to(device),torch.from_numpy(np.load(DATA/'train-labels.npy')).to(device)


def loaders(images,labels,parts,settings,seed):
    return {cid:resident_loader(images,labels,ids,{**settings,'seed':seed},int(cid)) for cid,ids in parts['clients'].items()}


if __name__=='__main__':prepare()
