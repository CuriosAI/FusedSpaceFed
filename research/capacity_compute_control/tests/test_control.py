"""Synthetic CPU checks of capacity, budgets, partition and exact resume."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from research.feature_shift_digits.data import DOMAINS,DIRECTORIES,canonical_hash,atomic_json
from research.feature_shift_digits import run_digits as previous
from research.capacity_compute_control import runner
from research.capacity_compute_control.compute import matched_plan,fused_round_cost,step_cost,profile_step
from research.capacity_compute_control.model import EnlargedDigitCNN
from research.capacity_compute_control.validation import stratified_indices


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2);yield;torch.set_num_threads(old)


def profile():
    return json.loads(Path('research/capacity_compute_control/flop_profile.json').read_text())


def test_capacity_and_logits():
    model=EnlargedDigitCNN();params=profile()['parameters']
    assert sum(p.numel() for p in model.parameters())==14334589
    fused=params['classifier']+params['encoder']+params['decoder']
    assert fused==14336285 and abs(14334589/fused-1)<0.00012
    assert model(torch.randn(2,3,28,28)).shape==(2,10)


@pytest.mark.parametrize('count',[594,743])
def test_cumulative_budget_includes_warmup_and_rounding_carry(count):
    p=profile();carry=spent=0;target=fused_round_cost(p,count)
    assert target>sum(step_cost(p,'classification',batch) for batch in ([32]*(count//32)+[count%32]))
    for number in range(1,301):
        plan,carry,amount=matched_plan(p,count,carry);spent+=amount
        assert sum(plan)>=count and all(2<=size<=32 for size in plan)
        assert 0<=carry<step_cost(p,'fedavg',2)
        assert spent+carry==number*target
    assert abs(spent/(target*300)-1)<0.00001


def test_measured_forward_backward_signature_and_semantic_updates():
    p=profile();observed=profile_step(2,p['settings'])
    assert observed==p['batches']['2']
    assert observed['phases']['warmup']['dense']>0
    assert 'aten.convolution_backward' in observed['phases']['warmup']['operators']
    assert observed['phases']['fedavg']['update_and_clip']==5*14334589


def test_stratified_training_only_split_is_reproducible_and_disjoint():
    labels=np.arange(743)%10;ids=[f'training:{i}' for i in range(743)]
    fit,val=stratified_indices(labels,ids,'synthetic')
    assert (fit,val)==stratified_indices(labels,ids,'synthetic')
    assert len(fit)==594 and len(val)==149
    assert not set(fit)&set(val) and set(fit)|set(val)==set(range(743))
    assert all(np.bincount(labels[fit],minlength=10)>0)
    assert all(np.bincount(labels[val],minlength=10)>0)


def test_budget_sampler_covers_original_epoch_and_rng_reproducible():
    plan=[32]*23+[7,32,17]
    batches=list(runner.BudgetSampler(743,plan,torch.Generator().manual_seed(43)))
    assert batches==list(runner.BudgetSampler(743,plan,torch.Generator().manual_seed(43)))
    assert sorted(sum(batches[:24],[]))==list(range(743))
    assert [len(batch) for batch in batches]==plan


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__();self.fc=nn.Linear(3,10)
    def forward(self,x):
        return self.fc(x.mean((2,3)))


@pytest.mark.parametrize('method',['FusedSpaceFed','FedAvg'])
def test_checkpoint_resume_states_rng_budget_metrics_and_overwrite(tmp_path,monkeypatch,method):
    monkeypatch.setattr(runner,'DigitCNN',TinyCNN);monkeypatch.setattr(previous,'DigitCNN',TinyCNN)
    monkeypatch.setattr(runner,'EnlargedDigitCNN',TinyCNN)
    partition=tmp_path/'data';partition.mkdir()
    for domain in DOMAINS:
        prefix=DIRECTORIES.get(domain,domain)+'-train'
        np.save(partition/(prefix+'-images.npy'),np.random.default_rng(7).integers(0,256,(6,3,28,28),dtype=np.uint8))
        np.save(partition/(prefix+'-labels.npy'),np.arange(6,dtype=np.int64))
    monkeypatch.setattr(runner,'verify',lambda path:{'partition_sha256':'synthetic'})
    split={'domains':{d:{'fit_indices':[0,1,2,3],'validation_indices':[4,5]} for d in DOMAINS}}
    split['validation_sha256']=canonical_hash(split);split_path=tmp_path/'split.json';atomic_json(split_path,split)
    p=profile();profile_path=tmp_path/'profile.json';atomic_json(profile_path,p)
    config={'phase':'validation','method':method,'seed':142,'parent_partition_sha256':'synthetic',
            'profile_file':str(profile_path),'profile_sha256':p['profile_sha256'],
            'validation_file':str(split_path),'validation_sha256':split['validation_sha256'],
            'training':{**p['settings'],'rounds':2,'torch_threads':2}}
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    runner.run(config,partition,full,torch.device('cpu'))
    partial=runner.run(config,partition,resumed,torch.device('cpu'),stop_after=1)
    assert partial['evaluations']==[]
    runner.run(config,partition,resumed,torch.device('cpu'),resume=True)
    a=torch.load(full/'checkpoint.pt',weights_only=False);b=torch.load(resumed/'checkpoint.pt',weights_only=False)
    for component in ('classifier','decoder'):
        assert all(torch.equal(value,b[component][key]) for key,value in a[component].items())
    for domain in a['encoders']:
        assert all(torch.equal(value,b['encoders'][domain][key]) for key,value in a['encoders'][domain].items())
    assert a['carries']==b['carries'] and torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert all(torch.equal(a['rng']['loaders'][domain],b['rng']['loaders'][domain]) for domain in DOMAINS)
    assert a['results']['evaluations'][0]['domains']==b['results']['evaluations'][0]['domains']
    for metric in a['results']['evaluations'][0]['domains'].values():
        assert metric['total']==2 and metric['accuracy_percent']==100*metric['correct']/2
    with pytest.raises(FileExistsError):
        runner.run(config,partition,full,torch.device('cpu'))
