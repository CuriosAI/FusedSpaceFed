import copy
import json
import pytest
import torch
from torch import nn
from fusedspacefed_core import ResNet20V2,UNetSmallAE
from research.pathmnist_head_only.run import replace_head_only,tree_equal,verify_counts,validate,ROOT
from research.pathmnist_recovery.head_calibration import features


def test_head_replacement_preserves_original_bn_and_complete_private_training_states():
    classifier={'fc.weight':torch.zeros(2,2),'fc.bias':torch.zeros(2),'body.weight':torch.ones(2,2),
                'bn.running_mean':torch.zeros(2),'bn.running_var':torch.ones(2),'bn.num_batches_tracked':torch.tensor(7)}
    source={'classifier':classifier,'decoder':{'w':torch.ones(2)},'encoders':{'0':{'w':torch.ones(2)}},
            'clients':{'0':{'classifier':copy.deepcopy(classifier),'ae_optimizer':{'state':{'m':torch.ones(2)}},
                              'scaler':{'scale':128.},'loader_generator':torch.tensor([1,2])}},
            'workers':{'0':{'rng':torch.tensor([3,4])}}}
    fitted=copy.deepcopy(classifier);fitted['fc.weight'].add_(1.);fitted['fc.bias'].add_(2.)
    result=replace_head_only(source,fitted)
    for name in ('decoder','encoders','workers'):assert tree_equal(source[name],result[name])
    assert tree_equal(result['clients']['0']['classifier'],fitted)
    for name in ('ae_optimizer','scaler','loader_generator'):
        assert tree_equal(source['clients']['0'][name],result['clients']['0'][name])
    assert torch.equal(source['classifier']['fc.weight'],torch.zeros(2,2))


@pytest.mark.parametrize('key',['body.weight','bn.running_mean','bn.running_var','bn.num_batches_tracked'])
def test_head_only_rejects_classifier_body_and_bn_changes(key):
    original={'fc.weight':torch.zeros(2,2),'fc.bias':torch.zeros(2),'body.weight':torch.ones(2,2),
              'bn.running_mean':torch.zeros(2),'bn.running_var':torch.ones(2),'bn.num_batches_tracked':torch.tensor(7)}
    fitted=copy.deepcopy(original);fitted[key].add_(1)
    with pytest.raises(ValueError,match='frozen classifier weight or BN'):
        replace_head_only({'classifier':original},fitted)


def test_features_use_owner_encoder_with_native_bn_and_no_gradients():
    torch.set_num_threads(1);torch.manual_seed(5)
    model=ResNet20V2(9,3).eval();classifier=copy.deepcopy(model.state_dict())
    ae=UNetSmallAE(3,16).eval();decoder=ae.decoder_state();encoders={}
    for cid in range(10):
        encoder=copy.deepcopy(ae.encoder_state());encoder['enc1.0.bias'].add_(cid*.1);encoders[str(cid)]=encoder
    images=torch.rand(20,3,32,32);labels=torch.arange(20)%9
    split={str(cid):[2*cid,2*cid+1] for cid in range(10)}
    xx,yy,owners=features(classifier,decoder,encoders,images,labels,split,{},torch.device('cpu'))
    assert xx.shape==(20,64) and not xx.requires_grad
    assert torch.equal(yy,labels) and torch.equal(owners,torch.arange(10).repeat_interleave(2))
    assert tree_equal(classifier,model.state_dict())
    model.fc=nn.Identity()
    for cid in (0,9):
        ae.load_encoder_state(encoders[str(cid)])
        x=images[split[str(cid)]]
        with torch.no_grad():d,_=ae(x);expected=model(x+d)
        assert torch.allclose(xx[2*cid:2*cid+2],expected,atol=1e-6)


def test_metric_reconstructed_from_all_pipelines():
    rows=[{'client_id':i,'correct':i*100,'total':7180,'accuracy_percent':100*i*100/7180} for i in range(10)]
    metric={'pipeline_metrics':rows,'correct_total':4500,'predictions_total':71800,
            'uniform_pipeline_accuracy_percent':100*4500/71800}
    verify_counts(metric)
    metric['uniform_pipeline_accuracy_percent']+=1.
    with pytest.raises(AssertionError,match='uniform pipeline'):verify_counts(metric)


def test_source_guard_rejects_incomplete_original_state_and_tuned_settings():
    config=json.loads((ROOT/'research/pathmnist_head_only/config.json').read_text())
    state={'config':json.loads((ROOT/'research/pathmnist_pathological/config.json').read_text()),
           'seed':42,'round':50,'partition_sha256':config['source_partition_sha256'],
           'partitions':{'clients':{'0':[0]}},'clients':{'0':{}},'encoders':{'0':{}}}
    with pytest.raises(ValueError,match='Incomplete'):validate(config,state,{'full':{'0':[0]}})
    state['config']['classifier_lr']=.1
    with pytest.raises(ValueError,match='paper-settings'):validate(config,state,{'full':{'0':[0]}})
