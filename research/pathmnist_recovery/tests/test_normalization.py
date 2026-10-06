import copy
import torch
from research.pathmnist_recovery.normalization import assignments,calibrate_layerwise


def test_cross_assignment_uses_same_unique_fit_pool_with_equal_encoder_counts():
    p={'fit':{str(i):list(range(i*10,(i+1)*10)) for i in range(10)},
       'bn_calibration':{str(i):list(range(i*10,i*10+4)) for i in range(10)}}
    first=assignments(p,True);second=assignments(p,True)
    assert first==second and all(len(x)==4 for x in first.values())
    assert sorted(i for x in first.values() for i in x)==sorted(i for x in p['bn_calibration'].values() for i in x)


def test_layerwise_statistics_match_eval_inputs_and_preserve_parameters():
    torch.manual_seed(7)
    model=torch.nn.Sequential(torch.nn.BatchNorm1d(3),torch.nn.ReLU(),torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2))
    x=torch.randn(100,3)*torch.tensor([.2,2.,7.])+torch.tensor([3.,-2.,10.])
    parameters=copy.deepcopy(dict(model.named_parameters()))
    calibrate_layerwise(model,x,batch_size=13)
    assert torch.allclose(model[0].running_mean,x.mean(0),atol=1e-6)
    assert torch.allclose(model[0].running_var,x.var(0,unbiased=True),atol=1e-5)
    with torch.no_grad():hidden=model[:3](x)
    assert torch.allclose(model[3].running_mean,hidden.mean(0),atol=1e-6)
    assert torch.allclose(model[3].running_var,hidden.var(0,unbiased=True),atol=1e-6)
    assert all(torch.equal(v,dict(model.named_parameters())[k]) for k,v in parameters.items())
    assert not model.training and all(not m.training for m in model.modules())
