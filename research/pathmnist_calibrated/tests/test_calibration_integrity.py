import copy
import json
import numpy as np
import torch
from fusedspacefed_core import FusedSpaceFedClient, ResNet20V2, UNetSmallAE, clone_state_dict
from research.pathmnist_pathological.run import resident_loader, client_snapshot, restore_client
from research.pathmnist_pathological.test_pathmnist_snapshots import assert_same
from research.pathmnist_calibrated.client import CalibratedClient
from research.pathmnist_calibrated.prepare_plan import base
from research.pathmnist_calibrated.data import verify
from research.pathmnist_calibrated.runner import recalibrate_bn


def setup():
    torch.set_num_threads(1); torch.manual_seed(42)
    x, y = torch.rand(8,3,32,32), torch.tensor([0,1]*4)
    config={'seed':42,'batch_size':4}
    c,a=clone_state_dict(ResNet20V2(9,3).state_dict()),clone_state_dict(UNetSmallAE(3,16).state_dict())
    return x,y,config,c,a


def test_fp32_unclipped_updates_match_original_core_and_optimizer_persistence():
    x,y,cfg,c,a=setup()
    old=FusedSpaceFedClient(0,resident_loader(x,y,range(8),cfg,0),9,3,16,'multiclass',torch.device('cpu'),use_amp=False)
    new=CalibratedClient(0,resident_loader(x,y,range(8),cfg,0),base(.01,.001,precision='fp32',gradient_clip_norm=None,batch_size=4),torch.device('cpu'))
    for client in (old,new):
        client.set_classifier_state(c);client.set_full_autoencoder_state(a)
    for round_index in (1,2):
        old.train_round(1,3);new.train_round_at(round_index)
        assert_same(client_snapshot(old),client_snapshot(new))


def test_clipping_keeps_two_phase_gradient_flow_and_complete_resume():
    x,y,cfg,c,a=setup()
    settings=base(gradient_clip_norm=.001,precision='fp32',batch_size=4)
    client=CalibratedClient(0,resident_loader(x,y,range(8),cfg,0),settings,torch.device('cpu'))
    client.set_classifier_state(c);client.set_full_autoencoder_state(a)
    before=client_snapshot(client)
    client._warmup(1)
    warm=client_snapshot(client)
    assert_same(before['classifier'],warm['classifier']);assert_same(before['decoder'],warm['decoder'])
    client._joint_train(1)
    after=client_snapshot(client)
    assert any(not torch.equal(after['decoder'][k],warm['decoder'][k]) for k in warm['decoder'])
    restored=CalibratedClient(0,resident_loader(x,y,range(8),cfg,0),settings,torch.device('cpu'))
    restore_client(restored,after)
    client.train_round_at(2);restored.train_round_at(2)
    assert_same(client_snapshot(client),client_snapshot(restored))


def test_holdout_globally_disjoint_and_bn_uses_fit_only():
    p=verify()
    fit={i for row in p['fit'].values() for i in row}
    val={i for row in p['validation'].values() for i in row}
    assert not fit&val and len(fit|val)==89996
    assert all(set(row)<=fit for row in p['bn_calibration'].values())


def test_train_only_bn_recalibration_preserves_weights_rng_and_inputs():
    x,y,cfg,c,a=setup()
    decoder={k:v for k,v in a.items() if k.startswith(UNetSmallAE.DECODER_PREFIXES)}
    encoder={k:v for k,v in a.items() if k.startswith(UNetSmallAE.ENCODER_PREFIXES)}
    p={'bn_calibration':{str(i):list(range(8)) for i in range(10)}}
    before=torch.get_rng_state().clone(); inputs=x.clone()
    result=recalibrate_bn(c,decoder,{str(i):encoder for i in range(10)},x,p,torch.device('cpu'))
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(x,inputs)
    for k in c:
        if not k.endswith(('running_mean','running_var','num_batches_tracked')):
            assert torch.equal(c[k],result[k])
    assert any(not torch.equal(c[k],result[k]) for k in c if k.endswith('running_mean'))


def test_cosine_schedule_and_warm_lr_do_not_reset_adam_states():
    x,y,cfg,c,a=setup()
    settings=base(precision='fp32',batch_size=4,schedule='cosine',warmup_lr=.0001)
    client=CalibratedClient(0,resident_loader(x,y,range(8),cfg,0),settings,torch.device('cpu'))
    client.set_classifier_state(c);client.set_full_autoencoder_state(a)
    first=client.train_round_at(1)
    steps=[float(v['step']) for v in client.ae_optimizer.state.values()]
    second=client.train_round_at(2)
    assert first['learning_rate_factor']==1 and second['learning_rate_factor']<1
    assert all(float(v['step'])>s for v,s in zip(client.ae_optimizer.state.values(),steps))
    assert client.ae_optimizer.param_groups[0]['lr']==settings['autoencoder_lr']*second['learning_rate_factor']
