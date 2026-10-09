import json
from pathlib import Path
import pytest
import torch
from fusedspacefed_core import ResNet20V2,UNetSmallAE
from research.pathmnist_five_seed.diagnostic_common import tensor_metrics,state_hash,measure,probe,deltas
from research.pathmnist_five_seed.diagnostic_common import stats
from research.pathmnist_five_seed.phase3.probe_gradients import measure_state
from research.pathmnist_five_seed.tests.test_native_runner import client


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(old)


def test_path_image_range_and_reconstruction_metrics():
    x=torch.ones(2,3,4,4)*.5
    assert tensor_metrics(x,x)['reconstruction_mse']==0
    v=tensor_metrics(x,-x);assert v['decoder_fraction_outside_input_range']==1
    assert v['relative_reconstruction_mse']==4 and v['fused_to_input_rms_ratio']==0
    with pytest.raises(FloatingPointError):tensor_metrics(x,x*float('nan'))


def test_training_probe_immutable_bn_rng_and_loader_generator():
    a=client();x=torch.rand(8,3,32,32);y=torch.arange(8)%9
    before=(state_hash(a.classifier.state_dict()),state_hash(a.autoencoder.state_dict()),torch.get_rng_state().clone(),a.loader.generator.get_state().clone())
    modes=a.classifier.training,a.autoencoder.training
    value=measure(a,x,y,list(range(8)))
    assert value['samples']==8 and value['finite']
    assert before[:2]==(state_hash(a.classifier.state_dict()),state_hash(a.autoencoder.state_dict()))
    assert torch.equal(before[2],torch.get_rng_state()) and torch.equal(before[3],a.loader.generator.get_state())
    assert modes==(a.classifier.training,a.autoencoder.training)


@pytest.mark.local_artifacts
def test_frozen_probes_match_seed_specific_train_indices():
    value=json.loads(Path('research/pathmnist_five_seed/probe.json').read_text())
    assert value['batches_per_client']==5 and value['batch_size']==128
    for seed in range(42,47):
        rows=probe(seed)['clients'];parts=json.loads(Path(f'_local/pathmnist_five_seed/partitions/seed-{seed}.json').read_text())['clients']
        assert len(rows)==10
        for cid,row in rows.items():
            assert len(row['indices'])==len(set(row['indices']))==640
            assert set(row['indices'])<=set(parts[cid]) and set(row['visual_indices'])<=set(row['indices'])
            assert len(set(row['visual_labels']))==2


@pytest.mark.parametrize('mode',['eval','batch-stateless'])
def test_resnet_paired_gradients_bn_and_exact_decomposition(mode):
    model=ResNet20V2(9,3);ae=UNetSmallAE(3,16)
    state={'classifier':model.state_dict(),'decoder':ae.decoder_state(),'encoders':{'0':ae.encoder_state(),'1':ae.encoder_state()}}
    batches={str(cid):[(torch.rand(4,3,32,32),torch.arange(4)+cid),(torch.rand(4,3,32,32),torch.arange(4)+cid)] for cid in range(2)}
    value=measure_state(state,batches,mode,torch.device('cpu'))
    assert value['state_before']==value['state_after'] and value['rng_unchanged']
    assert value['classifier']['client_count']==2 and len(value['per_batch'])==2
    assert abs(value['classifier']['relative_identity_residual'])<1e-10
    assert value['classifier']['parameter_count']==sum(p.numel() for p in model.parameters())
    assert value['decoder']['parameter_count']==sum(p.numel() for p in ae.decoder_parameters())


def test_state_changes_separate_bn_buffers_from_weights():
    a={'classifier':{'weight':torch.ones(2),'bn.running_mean':torch.zeros(2),'bn.num_batches_tracked':torch.tensor(0)}}
    b={'classifier':{'weight':torch.ones(2),'bn.running_mean':torch.ones(2),'bn.num_batches_tracked':torch.tensor(5)}}
    value=deltas(a,b)['classifier'];assert value['parameter_l2']==0 and value['bn_buffer_l2']>0


def test_five_seed_summary_rejects_missing_seed_and_uses_sample_sd():
    with pytest.raises(ValueError):stats([1.,2.,3.,4.])
    result=stats([1.,2.,3.,4.,5.])
    assert result['mean']==3 and result['sd_sample_ddof1']==pytest.approx(2.5**.5)
