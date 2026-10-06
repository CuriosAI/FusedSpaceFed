import copy
import pytest
import torch
from torch.nn import functional as F
from research.pathmnist_recovery.head_calibration import weighted_ce, refit
from research.pathmnist_recovery.head_final import substitute_head


def test_uniform_client_objective_uses_local_means_not_sample_counts():
    logits=torch.tensor([[3.,0.],[0.,3.],[0.,3.],[0.,3.]])
    labels=torch.tensor([0,0,1,0]);owners=torch.tensor([0,1,1,1])
    expected=(F.cross_entropy(logits[:1],labels[:1])+F.cross_entropy(logits[1:],labels[1:]))/2
    assert torch.allclose(weighted_ce(logits,labels,owners),expected)
    assert not torch.allclose(expected,F.cross_entropy(logits,labels))


def test_refit_updates_only_fc_and_detaches_features_with_optimizer_saved():
    torch.manual_seed(3);torch.set_num_threads(1)
    x=torch.tensor([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]],requires_grad=True)
    labels=torch.tensor([0,0,1,1]);owners=torch.tensor([0,0,1,1])
    source={'fc.weight':torch.zeros(2,2),'fc.bias':torch.zeros(2),
            'body.weight':torch.randn(2,2),'bn.running_mean':torch.randn(2)}
    before=copy.deepcopy(source)
    fitted,optimizer,statistics=refit(source,x,labels,owners,.01,40)
    assert statistics['final_uniform_client_training_ce'] < .2
    assert x.grad is None
    assert torch.equal(fitted['body.weight'],before['body.weight'])
    assert torch.equal(fitted['bn.running_mean'],before['bn.running_mean'])
    assert all(torch.equal(source[k],v) for k,v in before.items())
    assert optimizer['state'] and statistics['closure_evaluations'] > 0


def test_complete_head_checkpoint_syncs_only_shared_head_and_bn():
    original={'fc.weight':torch.zeros(2,2),'fc.bias':torch.zeros(2),
              'body.weight':torch.ones(2,2),'bn.running_mean':torch.zeros(2)}
    source={'classifier':original,'decoder':{'w':torch.ones(3)},'encoders':{'0':{'w':torch.ones(1)}},
            'clients':{'0':{'classifier':copy.deepcopy(original),'ae_optimizer':{'state':{'w':torch.ones(3)}},
                             'loader_generator':torch.tensor([3,4])}},'rng':{'torch':torch.tensor([7,8])}}
    modified=copy.deepcopy(original);modified['fc.weight'].add_(2.);modified['bn.running_mean'].add_(1.)
    result=substitute_head(source,modified)
    assert torch.equal(result['clients']['0']['classifier']['fc.weight'],modified['fc.weight'])
    assert torch.equal(result['decoder']['w'],source['decoder']['w'])
    assert torch.equal(result['encoders']['0']['w'],source['encoders']['0']['w'])
    assert torch.equal(result['clients']['0']['ae_optimizer']['state']['w'],torch.ones(3))
    assert torch.equal(result['rng']['torch'],source['rng']['torch'])
    modified['body.weight'].add_(1.)
    with pytest.raises(ValueError,match='body weights'):substitute_head(source,modified)
