import json
from pathlib import Path
import pytest
import torch
from torch import nn
from fusedspacefed_core import ResNet20V2,UNetSmallAE
from research.pathmnist_recovery.head_calibration import weighted_ce,refit
from research.pathmnist_federated_head.client import LocalObjective,extract_features,logits_metrics,merge_metrics
from research.pathmnist_federated_head.protocol import (PARAMETERS,request,reply,read_reply,read_vector,
    WireLedger,mean_responses,flatten,optimize,central_gradient_anchor)
from research.digits_mechanism_diagnostics.common import state_hash


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def example():
    g=torch.Generator().manual_seed(4);head=nn.Linear(64,9)
    xs=[torch.randn(n,64,generator=g) for n in (7,13,21)]
    ys=[torch.arange(len(x))%9 for x in xs]
    return head,xs,ys


def test_fixed_width_messages_never_accept_feature_or_label_arrays():
    v=torch.arange(PARAMETERS,dtype=torch.float32);message=request(v)
    assert len(message)==2341 and torch.equal(read_vector(message[1:]),v)
    loss,g=read_reply(reply(1.5,v));assert loss==1.5 and torch.equal(g,v)
    assert len(reply(loss,g))==2345
    with pytest.raises(ValueError):request(torch.ones(100,64))
    with pytest.raises(ValueError):read_reply(b'L'+v.numpy().tobytes())
    with pytest.raises(ValueError):request(torch.ones(PARAMETERS,dtype=torch.int64))
    with pytest.raises(FloatingPointError):reply(float('nan'),v)


def test_uniform_local_losses_and_gradients_equal_central_objective_unequal_counts():
    head,xs,ys=example();theta=flatten(head)
    local=[LocalObjective(x,y,theta) for x,y in zip(xs,ys)]
    loss,g=mean_responses([c.value_gradient(theta) for c in local],torch.device('cpu'))
    owners=torch.cat([torch.full((len(x),),cid) for cid,x in enumerate(xs)])
    central=weighted_ce(head(torch.cat(xs)),torch.cat(ys),owners);central.backward()
    reference=torch.cat([p.grad.reshape(-1) for p in head.parameters()])
    assert torch.allclose(loss,central,atol=1e-6) and torch.allclose(g,reference,atol=1e-6)
    sample_weighted=nn.functional.cross_entropy(head(torch.cat(xs)),torch.cat(ys))
    assert not torch.isclose(loss,sample_weighted,atol=1e-5)
    assert all(not c.features.requires_grad and c.features.grad is None for c in local)


def test_server_lbfgs_matches_central_procedure_on_synthetic_fixed_features():
    head,xs,ys=example();theta=flatten(head);local=[LocalObjective(x,y,theta) for x,y in zip(xs,ys)]
    class Oracle:
        def evaluate(self,v):return mean_responses([c.value_gradient(v) for c in local],torch.device('cpu'))
    cfg=json.loads(Path('research/pathmnist_federated_head/config.json').read_text())
    fed,opt,trace=optimize(theta,Oracle(),cfg['lbfgs'],max_iter=6)
    owners=torch.cat([torch.full((len(x),),cid) for cid,x in enumerate(xs)])
    state={'fc.weight':head.weight.detach().clone(),'fc.bias':head.bias.detach().clone()}
    fitted,co,stats=refit(state,torch.cat(xs),torch.cat(ys),owners,0.,6)
    central=torch.cat([fitted['fc.weight'].reshape(-1),fitted['fc.bias']])
    assert torch.allclose(fed,central,atol=2e-5,rtol=2e-5)
    assert len(trace)==next(iter(opt['state'].values()))['func_evals']
    assert opt['param_groups'][0]['max_iter']==6 and opt['param_groups'][0]['line_search_fn']=='strong_wolfe'


def test_stored_gradient_anchor_identifies_pre_last_step_not_final():
    x=nn.Parameter(torch.tensor([3.,-2.,4.,1.]))
    optimizer=torch.optim.LBFGS([x],lr=1,max_iter=3,line_search_fn='strong_wolfe')
    diagonal=torch.tensor([1.,2.,3.,4.])
    def closure():
        optimizer.zero_grad();loss=(diagonal*x.square()).sum();loss.backward();return loss
    optimizer.step(closure)
    anchor,gradient,loss=central_gradient_anchor(x.detach().clone(),optimizer.state_dict())
    assert torch.allclose(2*diagonal*anchor,gradient,atol=3e-6)
    assert torch.allclose((diagonal*anchor.square()).sum(),torch.tensor(loss),atol=3e-6)
    assert not torch.allclose(gradient,2*diagonal*x.detach())


def test_local_features_frozen_bn_encoder_decoder_and_classifier_body():
    body=ResNet20V2(9,3);body.fc=nn.Identity();body.eval().requires_grad_(False)
    ae=UNetSmallAE(3,16).eval().requires_grad_(False)
    before=state_hash(body.state_dict()),state_hash(ae.state_dict())
    x=torch.rand(5,3,32,32);features=extract_features(body,ae,x,batch_size=3)
    with torch.no_grad():d,_=ae(x);expected=body(x+d)
    assert features.shape==(5,64) and not features.requires_grad
    assert torch.allclose(features,expected,atol=2e-6)
    assert before==(state_hash(body.state_dict()),state_hash(ae.state_dict()))
    assert all(p.grad is None for p in [*body.parameters(),*ae.parameters()])


def test_bytes_count_every_client_including_line_search_and_verification():
    ledger=WireLedger();v=torch.zeros(PARAMETERS)
    for _ in range(3):
        for _ in range(10):ledger.record('optimization','down',request(v));ledger.record('optimization','up',reply(1.,v))
    for _ in range(10):ledger.record('control','up',b'R')
    s=ledger.summary();a=s['categories']['optimization']
    assert a['payload_total']==3*10*(2341+2345) and a['pipe_framing_bytes']==3*10*2*4
    assert s['payload_total_bytes']==3*46860+10 and s['head_bytes']==2340 and s['gradient_and_loss_bytes']==2344


def test_logits_and_accuracy_metadata_have_no_logits_or_labels():
    a=torch.tensor([[1.,0.,0.],[0.,2.,0.]])
    b=torch.tensor([[0.,1.,0.],[0.,2.,0.]])
    v=merge_metrics([logits_metrics(a,b,{'atol':1e-6,'rtol':1e-6})])
    assert v['prediction_disagreements']==1 and v['max_abs']==1 and v['entries_outside_tolerance']==2
    assert v['samples']==2 and v['entries']==6
