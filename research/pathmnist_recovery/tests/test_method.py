import copy
import torch
from torch.utils.data import DataLoader,TensorDataset
from research.pathmnist_calibrated.prepare_plan import base
from research.pathmnist_recovery.method import build_classifier,augment_images,RecoveryClient
from research.pathmnist_recovery.training import adapted
from research.pathmnist_calibrated import runner
from research.pathmnist_pathological.run import client_snapshot,restore_client,rng_state,restore_rng


def test_gn_keeps_parameter_count_logits_and_sample_independent_normalization():
    torch.set_num_threads(1);torch.manual_seed(5)
    original=build_classifier({});model=build_classifier({'classifier_normalization':'groupnorm8'})
    assert sum(p.numel() for p in original.parameters())==sum(p.numel() for p in model.parameters())
    assert sum(isinstance(m,torch.nn.GroupNorm) for m in model.modules())==19
    assert not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules())
    x=torch.randn(2,3,32,32)
    with torch.no_grad():
        model.train();a=model(x);model.eval();b=model(x)
        changed=x.clone();changed[1].mul_(7);c=model(changed)
    assert a.shape==(2,9) and torch.equal(a,b)
    assert torch.allclose(b[0],c[0],atol=1e-6,rtol=1e-6)


def test_augmentation_preserves_pixels_input_and_rng_replay():
    torch.manual_seed(8);x=torch.arange(12*3*4*4).reshape(12,3,4,4).float();original=x.clone()
    state=torch.get_rng_state();a=augment_images(x,'flip-rot90')
    torch.set_rng_state(state);b=augment_images(x,'flip-rot90')
    assert torch.equal(a,b) and torch.equal(x,original)
    assert torch.equal(a.flatten(1).sort().values,x.flatten(1).sort().values)


def test_gn_augmented_client_preserves_two_phase_component_updates():
    torch.set_num_threads(1);torch.manual_seed(9)
    settings=base();settings.update(classifier_normalization='groupnorm8',augmentation='flip-rot90',local_epochs=1)
    data=TensorDataset(torch.rand(4,3,32,32),torch.tensor([0,1,0,1]))
    loader=DataLoader(data,batch_size=2,generator=torch.Generator().manual_seed(42),shuffle=True)
    client=RecoveryClient(0,loader,settings,torch.device('cpu'))
    c=client.classifier_state();d=client.decoder_state();e=client.encoder_state()
    client._warmup(1)
    assert all(torch.equal(v,client.classifier_state()[k]) for k,v in c.items())
    assert all(torch.equal(v,client.decoder_state()[k]) for k,v in d.items())
    assert any(not torch.equal(v,client.encoder_state()[k]) for k,v in e.items())
    client._joint_train(1)
    assert any(not torch.equal(v,client.classifier_state()[k]) for k,v in c.items())
    assert any(not torch.equal(v,client.decoder_state()[k]) for k,v in d.items())


def test_runner_adapter_restores_original_globals_after_error():
    before={k:getattr(runner,k) for k in ('SOURCES','CalibratedClient','ResNet20V2','validate_configuration','recalibrate_bn')}
    try:
        with adapted({'classifier_normalization':'groupnorm8'}):
            assert runner.CalibratedClient is RecoveryClient
            raise RuntimeError('test')
    except RuntimeError:pass
    assert all(getattr(runner,k) is v for k,v in before.items())


def test_augmented_gn_resume_restores_optimizer_loader_and_global_rng():
    torch.set_num_threads(1);torch.manual_seed(17)
    settings=base(local_epochs=1,classifier_normalization='groupnorm8',augmentation='flip-rot90')
    data=TensorDataset(torch.rand(4,3,32,32),torch.tensor([0,1,0,1]))
    def make():
        loader=DataLoader(data,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(19))
        return RecoveryClient(0,loader,settings,torch.device('cpu'))
    a=make();a.train_round_at(1);checkpoint=client_snapshot(a);random_state=rng_state(False)
    a.train_round_at(2);expected=client_snapshot(a)
    b=make();restore_client(b,checkpoint);restore_rng(random_state);b.train_round_at(2)
    actual=client_snapshot(b)
    for component in ('classifier','autoencoder'):
        assert all(torch.equal(v,actual[component][k]) for k,v in expected[component].items())
    assert torch.equal(expected['loader_generator'],actual['loader_generator'])
    for param_id,state in expected['ae_optimizer']['state'].items():
        assert all(torch.equal(value,actual['ae_optimizer']['state'][param_id][key])
                   for key,value in state.items())
