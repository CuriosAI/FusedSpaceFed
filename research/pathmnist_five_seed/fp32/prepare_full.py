"""Register exact original initial states and the separate five-run FP32 queue."""
import json
import torch
from research.pathmnist_five_seed.fp32.precision import ROOT, PUBLIC, PRIVATE
from research.pathmnist_five_seed.fp32.runner import SOURCES
from research.pathmnist_five_seed.data import partition, verify_cache
from research.pathmnist_pathological.run import file_hash, write_json, assert_finite


def prepare():
    verify_cache()
    sources={}
    for seed in range(42,47):
        path=ROOT/('_local/pathmnist_pathological/seed-42/initial.pt' if seed==42
                   else f'_local/pathmnist_five_seed/full/seed-{seed}/initial.pt')
        state=torch.load(path,map_location='cpu',weights_only=False);assert_finite(state)
        parts,spec=partition(seed)
        if state['seed']!=seed or state['round']!=0 or state['partitions']!=parts or len(state['clients'])!=10:
            raise ValueError('Original initialization or partition unavailable')
        if any(c['ae_optimizer']['state'] or c['classifier_optimizer']['state'] for c in state['clients'].values()):
            raise ValueError('Source is already trained')
        sources[str(seed)]={'path':str(path.relative_to(ROOT)),'sha256':file_hash(path),'bytes':path.stat().st_size,
                            'partition_sha256':spec['sha256'],'round':0,'original_amp_scaler_discarded':True}
    value={'seeds':sources,'policy':'exact original round-zero model/optimizer/loader states, not trained or refitted states'}
    registry=PUBLIC/'initial_sources.json'
    if registry.exists() and json.loads(registry.read_text())!=value:raise ValueError('Changed original initialization')
    write_json(registry,value)
    plan=json.loads((PUBLIC/'plan.json').read_text());folder=PUBLIC/'configs';folder.mkdir(exist_ok=True);jobs=[]
    for seed in plan['seeds']:
        source=sources[str(seed)]
        config={'seed':seed,'variant':'full','settings':plan['settings'],'partition_sha256':source['partition_sha256'],
                'paired_initial_checkpoint':source['path'],'paired_initial_sha256':source['sha256']}
        path=folder/f'full-seed-{seed}.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Changed full configuration')
        write_json(path,config)
        jobs.append({'name':f'full-seed-{seed}','script':'research/pathmnist_five_seed/fp32/runner.py',
                     'args':['--config',str(path.relative_to(ROOT))],
                     'output':str((PRIVATE/'full'/f'seed-{seed}').relative_to(ROOT))})
    names=(*SOURCES,'research/pathmnist_calibrated/controller.py',*[str(p.relative_to(ROOT)) for p in folder.glob('*.json')])
    write_json(PUBLIC/'full_queue.json',{'gpu_slots':{'cuda:1':3,'cuda:0':2},'minimum_free_mib':8192,
                                       'frozen_files':{name:file_hash(ROOT/name) for name in names},'jobs':jobs})


if __name__=='__main__':prepare()
