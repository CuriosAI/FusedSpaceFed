"""Measured dense forward/backward FLOPs plus declared update/norm costs.

Torch 2.5.1 FlopCounterMode counts convolution/transposed-convolution and
matrix multiplication, including actual backward output masks; FMA=2.
Add semantic scalar costs: SGD+L2 clip=5P, Adam+L2 clip=16P per step.
These are conventional arithmetic counts, not GPU instruction/energy counts.
BN, activations, losses, pooling, copies, finite checks and aggregation are
excluded from the matching metric; report elapsed time and memory separately.
"""
import argparse
import json
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.flop_counter import FlopCounterMode
from research.feature_shift_digits.run_digits import DigitsClient, set_seed
from research.feature_shift_digits.data import atomic_json, canonical_hash, file_hash
from research.capacity_compute_control.model import EnlargedDigitCNN


def clip_sgd(model,optimizer,limit=1.0):
    parameters=[p for p in model.parameters() if p.grad is not None]
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().double()) for p in parameters]))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Non-finite FedAvg gradient')
    factor=(limit/(norm+1e-6)).clamp(max=1.0)
    for parameter in parameters:
        parameter.grad.mul_(factor.to(parameter.grad.dtype))
    optimizer.step()
    return float(norm)


def profile_step(batch,settings,device=torch.device('cpu')):
    set_seed(20261005,device,2)
    inputs=torch.randn(batch,3,28,28); labels=torch.arange(batch)%10
    loader=DataLoader(TensorDataset(inputs,labels),batch_size=batch,shuffle=False)
    client=DigitsClient('synthetic-profile',loader,settings,device)
    phases={}
    for phase in ('warmup','classification'):
        client.phase=phase
        with FlopCounterMode(display=False) as counter:
            if phase=='warmup':
                client._warmup(1)
            else:
                client._joint_train(1)
        phases[phase]={'dense':counter.get_total_flops(),
                       'operators':{str(op):n for op,n in counter.get_flop_counts()['Global'].items()}}
    classifier=EnlargedDigitCNN().to(device)
    optimizer=torch.optim.SGD(classifier.parameters(),lr=settings['classifier_lr'])
    with FlopCounterMode(display=False) as counter:
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(classifier(inputs.to(device)),labels.to(device)).backward()
        clip_sgd(classifier,optimizer,settings['gradient_clip_norm'])
    phases['fedavg']={'dense':counter.get_total_flops(),
                     'operators':{str(op):n for op,n in counter.get_flop_counts()['Global'].items()}}
    parameters={'classifier':sum(p.numel() for p in client.classifier.parameters()),
                'encoder':sum(p.numel() for p in client.autoencoder.encoder_parameters()),
                'decoder':sum(p.numel() for p in client.autoencoder.decoder_parameters()),
                'fedavg':sum(p.numel() for p in classifier.parameters())}
    phases['warmup']['update_and_clip']=16*parameters['encoder']
    phases['classification']['update_and_clip']=16*(parameters['encoder']+parameters['decoder'])+5*parameters['classifier']
    phases['fedavg']['update_and_clip']=5*parameters['fedavg']
    for value in phases.values():
        value['counted_total']=value['dense']+value['update_and_clip']
    return {'batch_size':batch,'phases':phases,'parameters':parameters}


def step_cost(profile,phase,batch,*,dense_only=False):
    if batch<2 or batch>32:
        raise ValueError('Batch outside 2..32 (BN training requires at least two)')
    base=profile['batches']['32']['phases'][phase]
    dense=base['dense']//32*batch
    return dense if dense_only else dense+base['update_and_clip']


def epoch_batches(count):
    if count<2 or count%32==1:
        raise ValueError('Unsupported one-element BN remainder')
    return [32]*(count//32)+([count%32] if count%32 else [])


def fused_round_cost(profile,count):
    return sum(step_cost(profile,phase,batch) for phase in ('warmup','classification') for batch in epoch_batches(count))


def matched_plan(profile,count,carry=0):
    target=fused_round_cost(profile,count)
    plan=epoch_batches(count)
    spent=sum(step_cost(profile,'fedavg',batch) for batch in plan)
    budget=target+carry
    if spent>budget:
        raise ValueError('One FedAvg epoch exceeds the target')
    while spent+step_cost(profile,'fedavg',32)<=budget:
        plan.append(32);spent+=step_cost(profile,'fedavg',32)
    candidates=[batch for batch in range(2,33) if spent+step_cost(profile,'fedavg',batch)<=budget]
    if candidates:
        batch=max(candidates);plan.append(batch);spent+=step_cost(profile,'fedavg',batch)
    return plan,budget-spent,spent


def create_profile(settings,output):
    if output.exists():
        raise FileExistsError('Profile output exists')
    batches={str(batch):profile_step(batch,settings) for batch in (2,7,18,32)}
    for phase in ('warmup','classification','fedavg'):
        for batch,row in batches.items():
            assert row['phases'][phase]['dense']*32==batches['32']['phases'][phase]['dense']*int(batch)
    result={'schema':1,'torch':torch.__version__,'device':'cpu','input':'synthetic only',
            'source':'https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/flop_counter.py',
            'flop_counter_source_sha256':file_hash(Path(torch.__file__).parent/'utils/flop_counter.py'),
            'metric':'measured dense forward/backward FMA=2 + semantic SGD/clip 5P, Adam/clip 16P',
            'excluded':['BN','activation','pooling','loss','finite checks','copies','aggregation','kernel algorithm overhead'],
            'settings':settings,'batches':batches,'parameters':batches['32']['parameters']}
    result['profile_sha256']=canonical_hash(result)
    atomic_json(output,result)
    print(json.dumps({'profile_sha256':result['profile_sha256'],'parameters':result['parameters'],
                      'full_training_round_client_flops':fused_round_cost(result,743),
                      'full_training_fedavg_first_round_plan':matched_plan(result,743)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    create_profile(json.loads(arguments.config.read_text())['training'],arguments.output)
