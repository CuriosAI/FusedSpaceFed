"""Matched-parameter/compute Digits control. No changes to old experiments."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset,Sampler
from fusedspacefed_core import UNetSmallAE,clone_state_dict,weighted_average_states
from research.feature_shift_digits.data import DOMAINS,PreparedDigits,atomic_json,canonical_hash,file_hash,verify
from research.feature_shift_digits.model import DigitCNN
from research.feature_shift_digits.run_digits import (DigitsClient,aggregate,evaluate,rng_state,restore_rng,runtime,set_seed,synchronize)
from research.capacity_compute_control.model import EnlargedDigitCNN
from research.capacity_compute_control.compute import clip_sgd,fused_round_cost,matched_plan,step_cost,epoch_batches

SOURCES=('fusedspacefed_core.py','research/feature_shift_digits/data.py','research/feature_shift_digits/model.py',
         'research/feature_shift_digits/run_digits.py','research/capacity_compute_control/model.py',
         'research/capacity_compute_control/compute.py','research/capacity_compute_control/validation.py',
         'research/capacity_compute_control/runner.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_hash(value,key):
    content=value.copy();expected=content.pop(key)
    if canonical_hash(content)!=expected:
        raise ValueError(f'Changed {key}')
    return value


class BudgetSampler(Sampler):
    def __init__(self,count,plan,generator):
        self.count=count;self.plan=plan;self.generator=generator
    def __len__(self):
        return len(self.plan)
    def __iter__(self):
        queue=[]
        for size in self.plan:
            while len(queue)<size:
                queue.extend(torch.randperm(self.count,generator=self.generator).tolist())
            yield queue[:size]
            queue=queue[size:]


def train_fedavg(model,loader,settings,device):
    model.train();optimizer=torch.optim.SGD(model.parameters(),lr=settings['classifier_lr'],momentum=0,weight_decay=0)
    losses=[];norms=[];samples=0
    for inputs,labels in loader:
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(inputs);loss=nn.functional.cross_entropy(logits,labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite FedAvg training')
        loss.backward();norms.append(clip_sgd(model,optimizer,settings['gradient_clip_norm']))
        losses.append(float(loss));samples+=len(labels)
    return {'classification_batch_mean_loss':statistics.mean(losses),'classification_steps':len(losses),
            'classification_samples':samples,'gradient_norm_max':max(norms),
            'clipped_steps':sum(norm>settings['gradient_clip_norm'] for norm in norms)}


@torch.no_grad()
def evaluate_fedavg(model,loader,device):
    model.eval();confusion=torch.zeros(10,10,dtype=torch.int64);loss_sum=0.0
    for inputs,labels in loader:
        inputs,labels=inputs.to(device),labels.to(device);logits=model(inputs)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Non-finite FedAvg evaluation')
        predicted=logits.argmax(1)
        confusion+=torch.bincount((labels*10+predicted).cpu(),minlength=100).reshape(10,10)
        loss_sum+=float(nn.functional.cross_entropy(logits,labels,reduction='sum'))
    correct=int(confusion.diag().sum());total=int(confusion.sum())
    if not total:
        raise ValueError('Empty evaluation')
    return {'correct':correct,'total':total,'accuracy_percent':100*correct/total,
            'sample_mean_cross_entropy':loss_sum/total,'confusion_matrix':confusion.tolist()}


def run(config,partition,output,device,*,resume=False,stop_after=None):
    started=time.perf_counter();settings=config['training'];method=config['method'];seed=config['seed']
    if method not in ('FusedSpaceFed','FedAvg') or config['phase'] not in ('validation','final'):
        raise ValueError('Unknown method/phase')
    parent=verify(partition)
    if parent['partition_sha256']!=config['parent_partition_sha256']:
        raise ValueError('Changed parent partition')
    profile=verify_hash(json.loads((REPO/config['profile_file']).read_text()),'profile_sha256')
    validation=verify_hash(json.loads((REPO/config['validation_file']).read_text()),'validation_sha256')
    if profile['profile_sha256']!=config['profile_sha256'] or validation['validation_sha256']!=config['validation_sha256']:
        raise ValueError('Changed profile/validation identity')
    if config['phase']=='final':
        selected=json.loads((REPO/config['selection_file']).read_text())
        if canonical_hash(selected)!=config['selection_sha256'] or any(settings[key]!=value for key,value in selected['selected_training_settings'][method].items()):
            raise ValueError('Final configuration not frozen by validation selection')
        if settings['rounds']!=300 or seed not in (42,43,44):
            raise ValueError('Wrong definitive protocol')
    if output.exists() and not resume:
        raise FileExistsError('Refusing output overwrite')
    if resume and not output.is_dir():
        raise FileNotFoundError('Resume directory missing')
    output.mkdir(parents=True,exist_ok=resume)
    lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'method':method,'seed':seed,'device':str(device),'config_sha256':canonical_hash(config),
              'partition_sha256':parent['partition_sha256'],'validation_sha256':validation['validation_sha256'],
              'profile_sha256':profile['profile_sha256'],
              'code':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      'source_sha256':{name:file_hash(REPO/name) for name in SOURCES}}}
    set_seed(seed,device,settings['torch_threads'])
    datasets={domain:PreparedDigits(partition,domain,'train') for domain in DOMAINS}
    if config['phase']=='validation':
        datasets={domain:Subset(datasets[domain],validation['domains'][domain]['fit_indices']) for domain in DOMAINS}
    count=len(datasets[DOMAINS[0]])
    if any(len(dataset)!=count for dataset in datasets.values()):
        raise ValueError('Unequal training sizes')
    generators={domain:torch.Generator().manual_seed(seed*10000+index) for index,domain in enumerate(DOMAINS)}
    loaders={domain:DataLoader(datasets[domain],batch_size=32,shuffle=True,num_workers=0,generator=generators[domain]) for domain in DOMAINS}
    model=DigitCNN() if method=='FusedSpaceFed' else EnlargedDigitCNN()
    classifier=clone_state_dict(model.state_dict());del model
    decoder={};encoders={}
    if method=='FusedSpaceFed':
        ae=UNetSmallAE(3,settings['dz']);decoder=ae.decoder_state();encoders={domain:ae.encoder_state() for domain in DOMAINS};del ae
    carries={domain:0 for domain in DOMAINS}
    results={'identity':identity,'configuration':config,'runtime':runtime(device),'status':'running','completed_rounds':0,
             'history':[],'evaluations':[],'sessions':[],'parameters':profile['parameters'],
             'matching_metric':profile['metric'],'excluded_flop_operations':profile['excluded'],'result_origin':'ours'}
    first=1
    if resume:
        checkpoint=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
        if checkpoint['results']['identity']!=identity:
            raise ValueError('Resume identity changed')
        results=checkpoint['results']
        if results['status']=='completed':
            return results
        classifier=checkpoint['classifier'];decoder=checkpoint['decoder'];encoders=checkpoint['encoders'];carries=checkpoint['carries']
        restore_rng(checkpoint['rng'],device,loaders);first=results['completed_rounds']+1
        (output/'timings.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in results['history']))
    session={'started_utc':utc(),'start_round':first};results['sessions'].append(session)
    def save():
        session.update(end_round=results['completed_rounds'],wall_seconds=time.perf_counter()-started,ended_utc=utc())
        if results['status']=='completed':
            results['total_session_wall_seconds']=sum(s['wall_seconds'] for s in results['sessions'])
        tmp=output/'checkpoint.pt.tmp'
        torch.save({'results':results,'classifier':classifier,'decoder':decoder,'encoders':encoders,'carries':carries,
                    'rng':rng_state(device,loaders)},tmp);tmp.replace(output/'checkpoint.pt')
        atomic_json(output/'results.json',results)
    save()
    for number in range(first,settings['rounds']+1):
        synchronize(device);round_started=time.perf_counter();states=[];decoders=[];clients={}
        for domain in DOMAINS:
            if method=='FusedSpaceFed':
                client=DigitsClient(domain,loaders[domain],settings,device)
                client.set_classifier_state(classifier);client.set_decoder_state(decoder);client.set_encoder_state(encoders[domain])
                metric=client.train_round(1,1);states.append(client.classifier_state());decoders.append(client.decoder_state())
                encoders[domain]=client.encoder_state();del client
                metric['counted_flops']=fused_round_cost(profile,count)
                metric['dense_flops']=sum(step_cost(profile,phase,batch,dense_only=True) for phase in ('warmup','classification') for batch in epoch_batches(count))
            else:
                plan,carries[domain],spent=matched_plan(profile,count,carries[domain])
                sampler=BudgetSampler(count,plan,generators[domain])
                loader=DataLoader(datasets[domain],batch_sampler=sampler,num_workers=0,generator=generators[domain])
                model=EnlargedDigitCNN().to(device);model.load_state_dict(classifier)
                metric=train_fedavg(model,loader,settings,device);states.append(clone_state_dict(model.state_dict()));del model,loader
                metric.update(counted_flops=spent,batch_sizes=plan,budget_carry=carries[domain],
                              dense_flops=sum(step_cost(profile,'fedavg',batch,dense_only=True) for batch in plan))
            metric['target_counted_flops']=fused_round_cost(profile,count);clients[domain]=metric
        classifier,decoder=aggregate(states,decoders) if method=='FusedSpaceFed' else (weighted_average_states(states,[1]*5),{})
        del states,decoders
        synchronize(device)
        row={'round':number,'clients':clients,'seconds':time.perf_counter()-round_started}
        results['history'].append(row);results['completed_rounds']=number
        with (output/'timings.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        if number<=5 or number%10==0 or number==settings['rounds'] or number==stop_after:
            save()
            print(json.dumps({'method':method,'phase':config['phase'],'seed':seed,'round':number,'seconds':row['seconds'],
                              'remaining_seconds_estimate':statistics.median(r['seconds'] for r in results['history'][-10:])*(settings['rounds']-number)}),flush=True)
        if number==stop_after and number<settings['rounds']:
            return results
    # No test access in calibration; only the held-out training subset is loaded.
    preserved=rng_state(device,loaders);evaluation={'round':settings['rounds'],'split':config['phase'],'domains':{}}
    eval_started=time.perf_counter()
    for domain in DOMAINS:
        dataset=PreparedDigits(partition,domain,'train' if config['phase']=='validation' else 'test')
        if config['phase']=='validation':
            dataset=Subset(dataset,validation['domains'][domain]['validation_indices'])
        loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed*10000+999))
        if method=='FusedSpaceFed':
            client=DigitsClient(domain,loaders[domain],settings,device)
            client.set_classifier_state(classifier);client.set_decoder_state(decoder);client.set_encoder_state(encoders[domain])
            evaluation['domains'][domain]=evaluate(client,loader);del client
        else:
            model=EnlargedDigitCNN().to(device);model.load_state_dict(classifier)
            evaluation['domains'][domain]=evaluate_fedavg(model,loader,device);del model
    restore_rng(preserved,device,loaders)
    evaluation['seconds']=time.perf_counter()-eval_started
    evaluation['uniform_domain_accuracy_percent']=statistics.mean(row['accuracy_percent'] for row in evaluation['domains'].values())
    evaluation['sample_weighted_accuracy_percent']=100*sum(row['correct'] for row in evaluation['domains'].values())/sum(row['total'] for row in evaluation['domains'].values())
    results['evaluations']=[evaluation];results['status']='completed'
    results['peak_cuda_allocated_mib']=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0
    results['peak_cuda_reserved_mib']=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else 0
    results['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    results['total_counted_training_flops']=sum(row['counted_flops'] for record in results['history'] for row in record['clients'].values())
    results['total_dense_training_flops']=sum(row['dense_flops'] for record in results['history'] for row in record['clients'].values())
    save()
    print(json.dumps({'status':'completed','method':method,'seed':seed,'evaluation':evaluation}),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--partition',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.partition,args.output,torch.device(args.device),resume=args.resume)
