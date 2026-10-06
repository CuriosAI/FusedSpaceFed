import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fusedspacefed_core import FusedSpaceFedClient, clone_state_dict, seed_everything
from research.pathmnist_pathological.run import resident_loader, client_snapshot
from research.pathmnist_five_seed.client import PathClient
from research.pathmnist_five_seed.data import make_partition, partition
from research.pathmnist_five_seed.runner import aggregate, train, validate


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def settings():
    return {'rounds':2,'warmup_epochs':1,'local_epochs':3,'batch_size':4,
            'classifier_lr':.01,'autoencoder_lr':.001,'use_amp':False}


def tensors():
    g=torch.Generator().manual_seed(500)
    return torch.rand(8,3,32,32,generator=g),torch.arange(8)%9


def client(variant='full',cid=0):
    x,y=tensors();cfg={**settings(),'seed':42}
    loader=resident_loader(x,y,range(4),cfg,cid)
    seed_everything(42)
    return PathClient(cid,loader,cfg,torch.device('cpu'),variant)


def assert_tree(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for v,w in zip(a,b):assert_tree(v,w)
    else:assert a==b


def test_full_updates_identical_to_original_and_persistent_optimizers():
    a=client();x,y=tensors();cfg={**settings(),'seed':42}
    b=FusedSpaceFedClient(0,resident_loader(x,y,range(4),cfg,0),9,3,16,'multiclass',torch.device('cpu'),use_amp=False)
    b.set_classifier_state(a.classifier_state());b.set_full_autoencoder_state(a.autoencoder.state_dict())
    for _ in range(2):
        metrics=a.local_round(1,3);original=b.train_round(1,3)
        assert metrics['classification_loss']==original['classification_loss']
        assert metrics['warmup_reconstruction_loss']==original['warmup_reconstruction_loss']
        assert_tree(client_snapshot(a),client_snapshot(b))
    assert a.actual=={'warmup_ae':2,'classification_ae':6,'classification_classifier':6}
    assert max(float(s['step']) for s in a.ae_optimizer.state.values())==8


def test_warmup_only_encoder_and_classification_all_components():
    a=client();before=[a.encoder_state(),a.decoder_state(),a.classifier_state()]
    a._warmup(1);warm=[a.encoder_state(),a.decoder_state(),a.classifier_state()]
    assert any(not torch.equal(before[0][k],warm[0][k]) for k in before[0])
    assert_tree(before[1:],warm[1:]);a.phase='classification';a._joint_train(3)
    after=[a.encoder_state(),a.decoder_state(),a.classifier_state()]
    assert all(any(not torch.equal(s[k],t[k]) for k in s) for s,t in zip(warm,after))


def test_no_warmup_and_decoder_only_input():
    a=client('no-warmup');r=a.local_round(1,3)
    assert r['warmup_steps']==0 and r['warmup_reconstruction_loss'] is None and a.actual['warmup_ae']==0
    b=client('decoder-only');seen=[];produced=[]
    h=b.classifier.register_forward_pre_hook(lambda m,v:seen.append(v[0].detach().clone()))
    j=b.autoencoder.register_forward_hook(lambda m,v,o:produced.append(o[0].detach().clone()))
    b.phase='classification';b._joint_train(1);h.remove();j.remove()
    assert_tree(seen,produced)


def test_shared_encoder_aggregation_keeps_local_optimizer_moments():
    a,b=client(cid=0),client(cid=1)
    a.local_round(1,3);b.local_round(1,3)
    with torch.no_grad():next(b.autoencoder.encoder_parameters()).add_(.1)
    original=[a.encoder_state(),b.encoder_state()];moments=[a.ae_optimizer.state_dict(),b.ae_optimizer.state_dict()]
    aggregate([a,b],'full');assert_tree(original,[a.encoder_state(),b.encoder_state()])
    aggregate([a,b],'shared-encoder');assert_tree(a.encoder_state(),b.encoder_state())
    assert_tree(moments,[a.ae_optimizer.state_dict(),b.ae_optimizer.state_dict()])


def test_partitions_frozen_complete_two_classes_and_reproduce_original():
    labels=np.load('_local/pathmnist_pathological/prepared/train-labels.npy')
    for seed in range(42,47):
        value,spec=partition(seed);assert value==make_partition(labels,seed)
        flat=[i for ids in value['clients'].values() for i in ids]
        assert len(flat)==len(set(flat))==len(labels)
        assert all(len(set(labels[ids]))==2 for ids in value['clients'].values())
    old=json.loads(Path('_local/pathmnist_pathological/prepared/partitions.json').read_text())
    assert partition(42)[0]==old


def test_checkpoint_resume_same_models_optimizers_rng_indices(tmp_path):
    x,y=tensors();parts={'seed':42,'clients':{'0':list(range(4)),'1':list(range(4,8))}}
    config={'seed':42,'variant':'full','settings':settings()};synthetic=(x,y,parts)
    train(config,tmp_path/'all','cpu',synthetic=synthetic)
    train(config,tmp_path/'resumed','cpu',synthetic=synthetic,stop_after=1)
    train(config,tmp_path/'resumed','cpu',resume=True,synthetic=synthetic)
    a=torch.load(tmp_path/'all/final.pt',weights_only=False);b=torch.load(tmp_path/'resumed/final.pt',weights_only=False)
    for k in ('classifier','decoder','encoders','clients','optimizer_step_counts','partitions'):
        assert_tree(a[k],b[k])
    assert torch.equal(a['rng']['torch_cpu'],b['rng']['torch_cpu'])
    assert a['round']==b['round']==2
    with pytest.raises(FileExistsError):train(config,tmp_path/'all','cpu',synthetic=synthetic)


def test_guard_original_settings_and_paired_ablations():
    plan=json.loads(Path('research/pathmnist_five_seed/plan.json').read_text())
    value,spec=partition(42);cfg={'seed':42,'variant':'full','settings':plan['settings'],'partition_sha256':spec['sha256']}
    validate(cfg,value,spec)
    with pytest.raises(ValueError):validate({**cfg,'variant':'no-warmup'},value,spec)
    with pytest.raises(ValueError):validate({**cfg,'settings':{**cfg['settings'],'use_amp':False}},value,spec)
