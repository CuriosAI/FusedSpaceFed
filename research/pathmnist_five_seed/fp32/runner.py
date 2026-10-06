"""Native original/paired-ablation PathMNIST, complete per-round checkpoints."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fusedspacefed_core import ResNet20V2,UNetSmallAE,clone_state_dict,seed_everything,weighted_average_states
from research.pathmnist_pathological.run import (file_hash,json_hash,write_json,atomic_save,assert_finite,
    client_snapshot,restore_client,rng_state,restore_rng,resident_loader)
from research.pathmnist_five_seed.data import DATA,partition,cache,loaders,verify_cache
from research.pathmnist_five_seed.fp32.profile import PUBLIC,force_fp32,check_client_precision
from research.pathmnist_five_seed.client import PathClient

SOURCES=('fusedspacefed_core.py','research/pathmnist_pathological/run.py','research/pathmnist_pathological/config.json',
         'research/pathmnist_five_seed/data.py','research/pathmnist_five_seed/data_identity.json',
         'research/pathmnist_five_seed/client.py','research/pathmnist_five_seed/fp32/runner.py',
         'research/pathmnist_five_seed/fp32/profile.py','research/pathmnist_five_seed/fp32/plan.json',
         'research/pathmnist_five_seed/fp32/initial_sources.json')


def identity():
    return {'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':{name:file_hash(ROOT/name) for name in SOURCES}}


def validate(config,parts,spec):
    plan=json.loads((PUBLIC/'plan.json').read_text())
    if config['seed'] not in plan['seeds'] or config['variant'] not in plan['variants'] or config['settings']!=plan['settings']:
        raise ValueError('Unregistered seed, variant or changed original settings')
    if config['partition_sha256']!=spec['sha256']:raise ValueError('Wrong paired partition')
    if not config.get('paired_initial_checkpoint'):
        raise ValueError('All FP32 runs require their frozen original or paired round-zero initialization')
    if config['variant']=='full':
        expected=json.loads((PUBLIC/'initial_sources.json').read_text())['seeds'][str(config['seed'])]
    else:
        expected=json.loads((PUBLIC/'anchors.json').read_text())['seeds'][str(config['seed'])]['initialization']
    if config['paired_initial_checkpoint']!=expected['path'] or config['paired_initial_sha256']!=expected['sha256']:
        raise ValueError('Unregistered round-zero source')
    return plan


def aggregate(clients,variant):
    classifier=weighted_average_states([c.classifier_state() for c in clients],[1]*len(clients))
    decoder=weighted_average_states([c.decoder_state() for c in clients],[1]*len(clients))
    if variant=='shared-encoder':
        encoder=weighted_average_states([c.encoder_state() for c in clients],[1]*len(clients))
        for c in clients:c.set_encoder_state(encoder)
    return classifier,decoder,{str(c.client_id):c.encoder_state() for c in clients}


@torch.no_grad()
def evaluate(classifier,decoder,encoders,images,labels,variant,device):
    model=ResNet20V2(9,3).to(device);model.load_state_dict(classifier);model.eval()
    ae=UNetSmallAE(3,16).to(device);ae.load_decoder_state(decoder);ae.eval()
    loader=resident_loader(images,labels,range(len(labels)),{'batch_size':128,'seed':20261006},100,shuffle=False)
    rows=[]
    for cid in range(10):
        ae.load_encoder_state(encoders[str(cid)]);correct=0
        for x,y in loader:
            d,_=ae(x);logits=model(d if variant=='decoder-only' else x+d)
            if not torch.isfinite(logits).all():raise FloatingPointError('Non-finite test logits')
            correct+=int((logits.argmax(1)==y).sum())
        rows.append({'client_id':cid,'correct':correct,'total':len(labels),'accuracy_percent':100*correct/len(labels)})
    return {'pipeline_metrics':rows,'correct_total':sum(r['correct'] for r in rows),
            'predictions_total':10*len(labels),'uniform_pipeline_accuracy_percent':statistics.mean(r['accuracy_percent'] for r in rows)}


def train(config,output,physical_device,resume=False,*,synthetic=None,stop_after=None):
    started=time.perf_counter();force_fp32()
    if config['settings']['use_amp'] is not False:raise ValueError('AMP forbidden in this campaign')
    if synthetic is None:
        os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1]
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2)
        source=verify_cache();parts,spec=partition(config['seed']);validate(config,parts,spec)
        images,labels=cache(device)
    else:
        device=torch.device('cpu');source={};images,labels,parts=synthetic;spec={'sha256':json_hash(parts)}
    if output.exists() and not resume:raise FileExistsError('Existing attempt; explicit resume required')
    output.mkdir(parents=True,exist_ok=resume);lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=config['settings'];seed_everything(config['seed'])
    classifier=clone_state_dict(ResNet20V2(9,3).state_dict());seed_everything(config['seed']+1000000)
    ae=UNetSmallAE(3,16);initial_ae=clone_state_dict(ae.state_dict());decoder=ae.decoder_state()
    client_loaders=loaders(images,labels,parts,settings,config['seed']);clients=[]
    for cid in range(len(parts['clients'])):
        c=PathClient(cid,client_loaders[str(cid)],settings,device,config['variant'])
        c.set_classifier_state(classifier);c.set_full_autoencoder_state(initial_ae);clients.append(c)
    code=identity();history=[];first=1;prior_seconds=0.;sessions=[]
    def restore(state):
        nonlocal classifier,decoder
        classifier,decoder=state['classifier'],state['decoder']
        for c in clients:
            restore_client(c,state['clients'][str(c.client_id)]);check_client_precision(c)
            c.actual=state.get('optimizer_step_counts',{}).get(str(c.client_id),c.actual).copy()
        if 'rng' in state:restore_rng(state['rng'])
        elif 'coordinator_rng' in state:
            # Original seed42 had two GPU workers. CPU coordinator RNG and the
            # first worker's CUDA RNG define this one-process execution. There
            # is no dropout/augmentation: data order uses the exact per-client
            # loader RNG already restored above, independently of process RNG.
            restored={**state['coordinator_rng']}
            if 'workers' in state:restored['torch_cuda']=state['workers']['0']['rng']['torch_cuda']
            restore_rng(restored)
    if resume:
        old=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if old['config_sha256']!=json_hash(config) or old['code']['source_sha256']!=code['source_sha256']:
            raise ValueError('Changed resume identity')
        if (output/'results.json').exists():raise FileExistsError('Completed attempt')
        restore(old);history=old['history'];first=old['round']+1;prior_seconds=old['elapsed_training_seconds'];sessions=old['sessions'];code=old['code']
    elif config.get('paired_initial_checkpoint'):
        path=ROOT/config['paired_initial_checkpoint']
        if file_hash(path)!=config['paired_initial_sha256']:raise ValueError('Changed paired initial state')
        old=torch.load(path,map_location='cpu',weights_only=False)
        if old['seed']!=config['seed'] or old['round']!=0 or old['partitions']['clients']!=parts['clients']:
            raise ValueError('Wrong paired seed/initialization/partition')
        restore(old)
    encoders={str(c.client_id):c.encoder_state() for c in clients}
    began=time.perf_counter();session={'start_round':first};sessions.append(session)
    def saved(number):
        session.update(end_round=number,wall_seconds=time.perf_counter()-started)
        value={'format':'pathmnist-fp32-five-seed-v1','precision':'FP32; no AMP/scaler/TF32','seed':config['seed'],'round':number,'configuration':config,
               'config_sha256':json_hash(config),'code':code,'data_manifest':source,'partition_sha256':spec['sha256'],
               'partitions':parts,'classifier':classifier,'decoder':decoder,'encoders':encoders,
               'clients':{str(c.client_id):client_snapshot(c) for c in clients},'rng':rng_state(cuda=device.type=='cuda'),
               'optimizer_step_counts':{str(c.client_id):c.actual.copy() for c in clients},'history':history,'sessions':sessions,
               'elapsed_training_seconds':prior_seconds+time.perf_counter()-began,'physical_device':physical_device,
               'shared_state_note':'initial/final client copies synchronized; latest can hold local pre-aggregation C/D'}
        for c in clients:check_client_precision(c)
        assert_finite(value);return value
    if not resume:atomic_save(output/'initial.pt',saved(0))
    for number in range(first,settings['rounds']+1):
        if identity()['source_sha256']!=code['source_sha256']:raise ValueError('Sources changed during training')
        before=time.perf_counter();records=[]
        for c in clients:
            c.set_classifier_state(classifier);c.set_decoder_state(decoder)
            records.append(c.local_round(settings['warmup_epochs'],settings['local_epochs']))
        classifier,decoder,encoders=aggregate(clients,config['variant'])
        if device.type=='cuda':torch.cuda.synchronize(device)
        warm=[r['warmup_reconstruction_loss'] for r in records if r['warmup_reconstruction_loss'] is not None]
        row={'round':number,'seconds':time.perf_counter()-before,'clients':records,
             'warmup_reconstruction_loss':statistics.mean(warm) if warm else None,
             'classification_loss':statistics.mean(r['classification_loss'] for r in records)}
        assert_finite(row);history.append(row);atomic_save(output/'latest.pt',saved(number))
        with (output/'timings.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        estimate=statistics.median(r['seconds'] for r in history[-5:])*(settings['rounds']-number)
        write_json(output/'progress.json',{'round':number,'target':settings['rounds'],'seed':config['seed'],'variant':config['variant'],
                                          'elapsed_seconds':time.perf_counter()-started,'remaining_seconds_estimate':estimate})
        print(json.dumps({'seed':config['seed'],'variant':config['variant'],'round':number,'seconds':row['seconds'],
                          'classification_loss':row['classification_loss'],'remaining_seconds_estimate':estimate}),flush=True)
        if stop_after==number and number<settings['rounds']:return
    training_seconds=prior_seconds+time.perf_counter()-began
    for c in clients:c.set_classifier_state(classifier);c.set_decoder_state(decoder)
    atomic_save(output/'final.pt',saved(settings['rounds']))
    if synthetic is None:
        test_images=torch.from_numpy(np.load(DATA/'test-images.npy')).to(device)
        test_labels=torch.from_numpy(np.load(DATA/'test-labels.npy')).to(device)
        before=time.perf_counter();metric=evaluate(classifier,decoder,encoders,test_images,test_labels,config['variant'],device)
        evaluation_seconds=time.perf_counter()-before
    else:metric=None;evaluation_seconds=0.
    result={'status':'completed','seed':config['seed'],'completed_rounds':settings['rounds'],'configuration':config,'code':code,
            'partition_sha256':spec['sha256'],'metrics':metric,'test_evaluations':1 if synthetic is None else 0,
            'test_round':settings['rounds'] if synthetic is None else None,'history':history,'sessions':sessions,
            'training_seconds':training_seconds,'session_seconds':time.perf_counter()-started,'evaluation_seconds':evaluation_seconds,
            'optimizer_step_counts':{str(c.client_id):c.actual for c in clients},
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'physical_device':physical_device,
            'precision':{'use_amp':False,'scaler':None,'matmul_tf32':False,'cudnn_tf32':False},
            'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                       'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'},
            'checkpoint_sha256':{p.name:file_hash(p) for p in output.glob('*.pt')},'completed_utc':datetime.now(timezone.utc).isoformat()}
    assert_finite(result);write_json(output/'results.json',result)
    print(json.dumps({'status':'completed','seed':config['seed'],'variant':config['variant'],'metrics':metric}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();train(json.loads(a.config.read_text()),a.output,a.device,a.resume)
