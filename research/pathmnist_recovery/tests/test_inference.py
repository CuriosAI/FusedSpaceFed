import copy
import pytest
import torch
from research.pathmnist_recovery.inference import substitute_shared_bn


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
