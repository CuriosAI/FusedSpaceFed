import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from research.pathmnist_five_seed.fp32.precision import PUBLIC, ROOT, force_fp32, check_client_precision
from research.pathmnist_five_seed.fp32.runner import train, validate
from research.pathmnist_five_seed.data import partition
from research.pathmnist_five_seed.fp32.diagnostic_common import probe
from research.pathmnist_five_seed.tests.test_native_runner import (
    client, settings, tensors, assert_tree, threads,
)
from research.pathmnist_pathological.run import file_hash, restore_client, client_snapshot


@pytest.mark.parametrize('entrypoint',['runner.py','phase1/diagnose.py','phase3/probe_gradients.py'])
def test_direct_cli_imports_do_not_shadow_python_standard_library(entrypoint):
    result=subprocess.run([sys.executable,str(PUBLIC/entrypoint),'--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--device' in result.stdout


def test_precision_only_profile_and_same_fixed_training_probes():
    previous=json.loads((PUBLIC.parent/'plan.json').read_text())
    current=json.loads((PUBLIC/'plan.json').read_text())
    assert current['settings']=={**previous['settings'],'use_amp':False}
    assert file_hash(PUBLIC/'probe.json')==file_hash(PUBLIC.parent/'probe.json')
    for seed in range(42,47):
        assert probe(seed)['partition_sha256']==partition(seed)[1]['sha256']


def test_no_amp_scaler_fp32_forward_and_no_tf32():
    force_fp32();c=client();check_client_precision(c)
    x,_=tensors();d,_=c.autoencoder(x);logits=c.classifier(x+d)
    assert d.dtype==logits.dtype==torch.float32
    assert c.scaler is None and not c.use_amp
    c.local_round(1,3);check_client_precision(c)
    assert c.actual=={'warmup_ae':1,'classification_ae':3,'classification_classifier':3}


@pytest.mark.parametrize('seed',range(42,47))
def test_historical_initial_models_optimizers_and_loader_restored_exactly(seed):
    force_fp32();registry=json.loads((PUBLIC/'initial_sources.json').read_text())['seeds'][str(seed)]
    path=ROOT/registry['path'];assert file_hash(path)==registry['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['seed']==seed and state['round']==0 and state['partitions']==partition(seed)[0]
    c=client();restore_client(c,state['clients']['0']);check_client_precision(c)
    actual=client_snapshot(c);old=state['clients']['0']
    for key in actual:
        if key!='scaler':assert_tree(actual[key],old[key])
    assert actual['scaler'] is None and old['scaler'] is not None
    assert not actual['classifier_optimizer']['state'] and not actual['ae_optimizer']['state']


def test_production_guards_reject_amp_unregistered_or_trained_sources():
    parts,spec=partition(42);cfg=json.loads((PUBLIC/'configs/full-seed-42.json').read_text())
    validate(cfg,parts,spec)
    with pytest.raises(ValueError):validate({**cfg,'settings':{**cfg['settings'],'use_amp':True}},parts,spec)
    with pytest.raises(ValueError):validate({**cfg,'paired_initial_checkpoint':'unregistered.pt'},parts,spec)
    with pytest.raises(ValueError):validate({k:v for k,v in cfg.items() if k!='paired_initial_checkpoint'},parts,spec)


def test_fp32_resume_exact_models_optimizers_rng_partition_and_step_counts(tmp_path):
    x,y=tensors();parts={'seed':42,'clients':{'0':list(range(4)),'1':list(range(4,8))}}
    cfg={'seed':42,'variant':'full','settings':settings()};synthetic=x,y,parts
    train(cfg,tmp_path/'continuous','cpu',synthetic=synthetic)
    train(cfg,tmp_path/'resumed','cpu',synthetic=synthetic,stop_after=1)
    train(cfg,tmp_path/'resumed','cpu',resume=True,synthetic=synthetic)
    a=torch.load(tmp_path/'continuous/final.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/final.pt',weights_only=False)
    for key in ('classifier','decoder','encoders','clients','optimizer_step_counts','partitions'):
        assert_tree(a[key],b[key])
    assert torch.equal(a['rng']['torch_cpu'],b['rng']['torch_cpu'])
    assert all(c['scaler'] is None for c in a['clients'].values())
    assert a['precision']==b['precision']=='FP32; no AMP/scaler/TF32'
    assert a['round']==b['round']==2
    with pytest.raises(FileExistsError):train(cfg,tmp_path/'continuous','cpu',synthetic=synthetic)
