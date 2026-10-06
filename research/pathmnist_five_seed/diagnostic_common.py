"""Fixed training probes and complete original checkpoint loading."""
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch
from research.pathmnist_five_seed.data import ROOT, PUBLIC, PRIVATE, DATA, partition
from research.pathmnist_pathological.run import file_hash, write_json, assert_finite
from research.digits_mechanism_diagnostics.common import state_hash, state_distance

ANCHORS=('initialization','final-round50')
STAGES=('before','after_warmup','after_classification')


def run_path(seed,variant='full'):
    if seed==42 and variant=='full':return ROOT/'_local/pathmnist_pathological/seed-42'
    return PRIVATE/variant/f'seed-{seed}'


def prepare_probes():
    labels=np.load(DATA/'train-labels.npy');seeds={}
    for seed in range(42,47):
        parts,spec=partition(seed);clients={}
        for cid,ids in parts['clients'].items():
            rng=np.random.default_rng(20261006+seed*10000+int(cid))
            chosen=rng.choice(ids,640,replace=False).tolist()
            classes=np.unique(labels[chosen]).tolist();visual=[]
            for k in classes:visual.extend([i for i in chosen if int(labels[i])==k][:2])
            assert len(classes)==2 and len(visual)==4
            clients[cid]={'indices':chosen,'visual_indices':visual,'visual_labels':labels[visual].tolist()}
        seeds[str(seed)]={'partition_sha256':spec['sha256'],'clients':clients}
    value={'selection':'fixed training-only RNG rule 20261006 + seed*10000 + client_id; 640 without replacement',
           'batch_size':128,'batches_per_client':5,'seeds':seeds}
    path=PUBLIC/'probe.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Changed probe')
    write_json(path,value)


def freeze_anchors():
    seeds={}
    for seed in range(42,47):
        folder=run_path(seed);anchors={}
        for name,filename,number in zip(ANCHORS,('initial.pt','final.pt'),(0,50)):
            path=folder/filename;s=torch.load(path,map_location='cpu',weights_only=False);assert_finite(s)
            parts,spec=partition(seed)
            if s['seed']!=seed or s['round']!=number or s['partitions']!=parts:raise ValueError('Wrong anchor')
            if len(s['clients'])!=10 or len(s['encoders'])!=10:raise ValueError('Incomplete private states')
            for cid,c in s['clients'].items():
                if state_hash(c['classifier'])!=state_hash(s['classifier']) or state_hash(c['decoder'])!=state_hash(s['decoder']):
                    raise ValueError('Initial/final shared client copy not synchronized')
                if state_hash(c['encoder'])!=state_hash(s['encoders'][cid]):raise ValueError('Wrong private encoder')
            anchors[name]={'path':str(path.relative_to(ROOT)),'sha256':file_hash(path),'round':number,
                           'partition_sha256':spec['sha256'],'bytes':path.stat().st_size}
        seeds[str(seed)]=anchors
    path=PUBLIC/'anchors.json';value={'seeds':seeds}
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Changed frozen anchors')
    write_json(path,value)


def load_anchor(seed,anchor):
    spec=json.loads((PUBLIC/'anchors.json').read_text())['seeds'][str(seed)][anchor];path=ROOT/spec['path']
    if file_hash(path)!=spec['sha256']:raise ValueError('Changed original anchor')
    state=torch.load(path,map_location='cpu',weights_only=False);assert_finite(state)
    return state,spec


def probe(seed):
    value=json.loads((PUBLIC/'probe.json').read_text());spec=value['seeds'][str(seed)]
    if spec['partition_sha256']!=partition(seed)[1]['sha256']:raise ValueError('Wrong probe partition')
    return spec


def source_identity(extra=()):
    names=('fusedspacefed_core.py','research/pathmnist_pathological/run.py',
           'research/digits_mechanism_diagnostics/common.py',
           'research/pathmnist_five_seed/data.py','research/pathmnist_five_seed/client.py',
           'research/pathmnist_five_seed/diagnostic_common.py','research/pathmnist_five_seed/plan.json',
           'research/pathmnist_five_seed/anchors.json','research/pathmnist_five_seed/probe.json',*extra)
    return {'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':{name:file_hash(ROOT/name) for name in names}}


def tensor_metrics(inputs,outputs):
    x,d=inputs.double(),outputs.double()
    if x.shape!=d.shape:raise ValueError('Wrong reconstruction shape')
    assert_finite([x,d]);signal=float(x.square().mean());energy=float(d.square().mean());fused=x+d
    outside=lambda v:float(((v<0)|(v>1)).double().mean())
    return {'reconstruction_mse':float((d-x).square().mean()),'zero_reconstruction_mse':signal,
            'relative_reconstruction_mse':float((d-x).square().mean())/signal,
            'input_rms':math.sqrt(signal),'decoder_rms':math.sqrt(energy),
            'decoder_to_input_rms_ratio':math.sqrt(energy/signal),
            'decoder_mean':float(d.mean()),'decoder_std_population':float(d.std(unbiased=False)),
            'decoder_min':float(d.min()),'decoder_max':float(d.max()),'decoder_max_abs':float(d.abs().max()),
            'decoder_fraction_outside_input_range':outside(d),'fused_fraction_outside_input_range':outside(fused),
            'fused_to_input_rms_ratio':math.sqrt(float(fused.square().mean())/signal),
            'input_decoder_cosine':float((x*d).mean())/math.sqrt(signal*energy) if energy else 0.,'finite':True}


@torch.no_grad()
def measure(client,images,labels,indices):
    modes=client.classifier.training,client.autoencoder.training
    client.classifier.eval();client.autoencoder.eval();xs=[];ds=[];orig=fused=0.
    try:
        for start in range(0,len(indices),128):
            ids=torch.tensor(indices[start:start+128],device=images.device);x,y=images[ids],labels[ids]
            d,_=client.autoencoder(x);lo,lf=client.classifier(x),client.classifier(x+d);assert_finite([d,lo,lf])
            orig+=float(torch.nn.functional.cross_entropy(lo,y,reduction='sum'))
            fused+=float(torch.nn.functional.cross_entropy(lf,y,reduction='sum'));xs.append(x.cpu());ds.append(d.cpu())
        return {**tensor_metrics(torch.cat(xs),torch.cat(ds)),'samples':len(indices),
                'original_cross_entropy':orig/len(indices),'fused_cross_entropy':fused/len(indices)}
    finally:client.classifier.train(modes[0]);client.autoencoder.train(modes[1])


def component_states(client):
    return {'encoder':client.encoder_state(),'decoder':client.decoder_state(),'classifier':client.classifier_state()}


def deltas(before,after):
    result={}
    for component in before:
        a,b=before[component],after[component]
        # ResNet BN buffers are identifiable; UNet has no running BN buffers.
        buffers={k for k in a if k.endswith(('running_mean','running_var','num_batches_tracked'))}
        weights=set(a)-buffers
        distance=lambda keys:state_distance({k:a[k] for k in keys},{k:b[k] for k in keys})
        result[component]={'all_state_l2':state_distance(a,b),'parameter_l2':distance(weights),'bn_buffer_l2':distance(buffers)}
    return result


def stats(values):
    import statistics
    if len(values)!=5:raise ValueError('Exactly five paired seeds required')
    return {'values':values,'mean':statistics.mean(values),'sd_sample_ddof1':statistics.stdev(values),'n':5}
