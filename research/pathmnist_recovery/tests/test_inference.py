import copy
import pytest
import torch
from research.pathmnist_recovery.inference import substitute_shared_bn, inference_decoder


def test_inference_substitution_preserves_weights_encoders_and_optimizers():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
    classifier=copy.deepcopy(model.state_dict())
    source={'classifier':copy.deepcopy(classifier),'encoders':{'0':{'w':torch.ones(2)}},
            'decoder':{'w':torch.ones(3)},'clients':{'0':{'classifier':copy.deepcopy(classifier),
            'ae_optimizer':{'state':{'first_moment':torch.ones(2)}},'scaler':{'scale':128.}}}}
    classifier['1.running_mean'].add_(2.)
    out=substitute_shared_bn(source,classifier)
    assert torch.equal(source['classifier']['1.running_mean'],torch.zeros(2))
    assert torch.equal(out['clients']['0']['classifier']['1.running_mean'],torch.full((2,),2.))
    assert torch.equal(out['encoders']['0']['w'],source['encoders']['0']['w'])
    assert torch.equal(out['decoder']['w'],source['decoder']['w'])
    assert torch.equal(out['clients']['0']['ae_optimizer']['state']['first_moment'],torch.ones(2))
    assert out['clients']['0']['scaler']==source['clients']['0']['scaler']


def test_inference_substitution_rejects_parameter_changes():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
    original=model.state_dict();changed=copy.deepcopy(original);changed['0.weight'].add_(1.)
    with pytest.raises(ValueError,match='changed a weight'):
        substitute_shared_bn({'classifier':original},changed)


def test_gain_checkpoint_keeps_original_training_state_and_reconstructs_inference():
    source={'decoder':{'final.weight':torch.ones(2,2),'final.bias':torch.full((2,),2.),
                       'body.weight':torch.full((2,),3.)},
            'recovery':{'config':{'fusion_gain':.25}}}
    before=copy.deepcopy(source)
    decoder=inference_decoder(source)
    assert torch.equal(decoder['final.weight'],torch.full((2,2),.25))
    assert torch.equal(decoder['final.bias'],torch.full((2,),.5))
    assert torch.equal(decoder['body.weight'],before['decoder']['body.weight'])
    assert all(torch.equal(source['decoder'][k],v) for k,v in before['decoder'].items())


@pytest.mark.parametrize('gain',[0.,-1.,float('nan'),float('inf')])
def test_inference_requires_positive_finite_decoder_path(gain):
    with pytest.raises(ValueError,match='finite and positive'):
        inference_decoder({'decoder':{},'recovery':{'config':{'fusion_gain':gain}}})
