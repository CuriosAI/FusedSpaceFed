"""Frozen stratified 149/743 validation split, derived only from training."""
import hashlib
import json
from pathlib import Path
import numpy as np
from research.feature_shift_digits.data import DOMAINS,DIRECTORIES,canonical_hash,atomic_json


def stratified_indices(labels,ids,domain):
    counts=np.bincount(labels,minlength=10)
    quotas=counts*149//743
    remainders=counts*149%743
    for label in sorted(range(10),key=lambda c:(-int(remainders[c]),c))[:149-int(quotas.sum())]:
        quotas[label]+=1
    validation=[]
    for label in range(10):
        indices=np.flatnonzero(labels==label).tolist()
        indices.sort(key=lambda i:(hashlib.sha256(f'capacity-validation-20261005|{domain}|{ids[i]}'.encode()).hexdigest(),i))
        validation.extend(indices[:int(quotas[label])])
    validation=sorted(validation)
    training=sorted(set(range(len(labels)))-set(validation))
    if len(training)!=594 or len(validation)!=149 or set(training)&set(validation):
        raise ValueError('Wrong validation split')
    return training,validation


def prepare(root,partition_hash,output):
    if output.exists():
        raise FileExistsError('Refusing to replace frozen validation split')
    manifest={'rule':'class-proportional largest remainder 149/743, ID SHA256 order','data_seed':20261005,
              'parent_partition_sha256':partition_hash,'domains':{},'source_split':'training only'}
    for domain in DOMAINS:
        directory=DIRECTORIES.get(domain,domain)
        labels=np.load(root/f'{directory}-train-labels.npy',allow_pickle=False)
        ids=json.loads((root/f'{directory}-train-ids.json').read_text())
        train,val=stratified_indices(labels,ids,domain)
        manifest['domains'][domain]={'fit_indices':train,'validation_indices':val,
                                    'fit_counts':np.bincount(labels[train],minlength=10).tolist(),
                                    'validation_counts':np.bincount(labels[val],minlength=10).tolist(),
                                    'ordered_train_ids_sha256':canonical_hash(ids)}
    manifest['validation_sha256']=canonical_hash(manifest)
    atomic_json(output,manifest)
    return manifest
