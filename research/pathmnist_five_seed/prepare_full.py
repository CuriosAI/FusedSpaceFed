import json
from pathlib import Path
import torch
from research.pathmnist_pathological.run import file_hash, write_json, assert_finite
from research.pathmnist_five_seed.runner import ROOT, PUBLIC, SOURCES
p=json.loads((PUBLIC/'plan.json').read_text());m=json.loads((PUBLIC/'data_identity.json').read_text())
old=ROOT/'_local/pathmnist_pathological/seed-42'
spec={name:file_hash(old/name) for name in ('initial.pt','final.pt','results.json')}
assert spec['initial.pt']=='2e068afbd9caee88b7f41d7032f054f4918b28dea46b73d17f8701faabcc4a40'
assert spec['final.pt']=='fa56da01091debe9031a9acb2e3de0f34ce0f751d2c95d2d0d54c05f15697d1e'
r=json.loads((old/'results.json').read_text());assert r['status']=='completed' and r['seed']==42
assert r['test_evaluations']==1 and r['test_round']==50
assert abs(r['test_metrics']['accuracy']*100-39.61977715877437)<1e-12
for name,number in [('initial.pt',0),('final.pt',50)]:
    s=torch.load(old/name,map_location='cpu',weights_only=False);assert_finite(s)
    assert s['seed']==42 and s['round']==number and len(s['encoders'])==len(s['clients'])==10
    for client in s['clients'].values():
        assert {'classifier_optimizer','ae_optimizer','scaler','loader_generator','classifier','autoencoder'} <= client.keys()
write_json(PUBLIC/'reused_seed42.json',{'path':str(old.relative_to(ROOT)),'files_sha256':spec,'accuracy_percent':39.61977715877437,'round':50,'read_only':True})
folder=PUBLIC/'configs';folder.mkdir(exist_ok=True);jobs=[]
for seed in range(43,47):
    cfg={'seed':seed,'variant':'full','settings':p['settings'],'partition_sha256':m['partitions'][str(seed)]['sha256']}
    cfgpath=folder/f'full-seed-{seed}.json';write_json(cfgpath,cfg)
    jobs.append({'name':f'full-seed-{seed}','script':'research/pathmnist_five_seed/runner.py','args':['--config',str(cfgpath.relative_to(ROOT))],'output':f'_local/pathmnist_five_seed/full/seed-{seed}'})
frozen={name:file_hash(ROOT/name) for name in (*SOURCES,'research/pathmnist_calibrated/controller.py')}
frozen.update({str(c.relative_to(ROOT)):file_hash(c) for c in folder.glob('*.json')})
write_json(PUBLIC/'full_queue.json',{'gpu_slots':{'cuda:1':2,'cuda:0':2},'minimum_free_mib':4096,'frozen_files':frozen,'jobs':jobs})
print(json.dumps({'seed42_integrity':'passed','partitions':{s:v['sha256'] for s,v in m['partitions'].items()}}))
