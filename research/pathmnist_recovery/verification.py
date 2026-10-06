"""Independent count/state audit; never evaluates images or launches training."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fusedspacefed_core import UNetSmallAE
from research.pathmnist_pathological.run import file_hash,write_json,assert_finite
from research.pathmnist_calibrated.data import verify,DATA
from research.pathmnist_recovery.method import build_classifier
from research.pathmnist_recovery.inference import inference_decoder


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def audit():
    directory=ROOT/'_local/pathmnist_recovery/attempt-05'
    result=json.loads((directory/'results.json').read_text());assert_finite(result)
    receipt=json.loads((ROOT/'_local/pathmnist_recovery/attempt-05-campaign.json').read_text())
    if receipt['status']!='completed' or len(receipt['runs'])!=1 or receipt['runs'][0]['exit_code']!=0:
        raise ValueError('Incomplete final process')
    if result['seed']!=42 or result['round']!=50 or result['test_evaluations']!=1:
        raise ValueError('Wrong final identity')
    metric=result['metrics'];rows=metric['pipeline_metrics']
    assert {r['client_id'] for r in rows}==set(range(10)) and len(rows)==10
    assert all(r['total']==7180 for r in rows)
    for r in rows:assert abs(r['accuracy_percent']-100*r['correct']/r['total'])<1e-10
    assert metric['correct_total']==sum(r['correct'] for r in rows)
    assert metric['predictions_total']==sum(r['total'] for r in rows)==71800
    reconstructed=100*metric['correct_total']/metric['predictions_total']
    assert abs(reconstructed-metric['uniform_pipeline_accuracy_percent'])<1e-10
    assert result['above_target']==(reconstructed>50.94)
    for name,digest in result['checkpoint_sha256'].items():assert file_hash(directory/name)==digest
    for name,digest in receipt['frozen_files'].items():assert file_hash(ROOT/name)==digest
    assert file_hash(ROOT/'research/pathmnist_recovery/attempt-05.json')==result['code']['config_sha256']
    assert result['config']==json.loads((ROOT/'research/pathmnist_recovery/attempt-05.json').read_text())
    partition=verify()
    data_manifest=json.loads((DATA/'manifest.json').read_text())
    for name,digest in data_manifest['files_sha256'].items():
        assert file_hash(DATA/name)==digest,('data hash',name)
    full=[i for ids in partition['full'].values() for i in ids]
    fit=[i for ids in partition['fit'].values() for i in ids]
    validation=[i for ids in partition['validation'].values() for i in ids]
    assert len(full)==len(set(full))==89996 and len(fit)==len(set(fit))==81002
    assert len(validation)==len(set(validation))==8994
    assert not set(fit)&set(validation) and set(fit)|set(validation)==set(full)
    assert all(set(partition['bn_calibration'][str(i)])<=set(partition['fit'][str(i)]) and
               len(partition['bn_calibration'][str(i)])==640 for i in range(10))
    loaded={name:torch.load(directory/name,map_location='cpu',weights_only=False)
            for name in ('initial.pt','precalibration.pt','final.pt')}
    before=loaded['precalibration.pt'];final=loaded['final.pt']
    checks={}
    for name,state in loaded.items():
        assert_finite(state);assert state['seed']==42
        assert state['round']==(0 if name=='initial.pt' else 50)
        assert state['training_indices']==partition['full']
        assert set(state['clients'])==set(state['encoders'])=={str(i) for i in range(10)}
        model=build_classifier(result['config']['training_settings']);model.load_state_dict(state['classifier'])
        bn_buffers=sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in state['classifier'])
        assert bn_buffers==57
        for cid,client in state['clients'].items():
            model.load_state_dict(client['classifier'])
            ae=UNetSmallAE(3,16);ae.load_state_dict(client['autoencoder'])
            assert equal(ae.encoder_state(),state['encoders'][cid])
            assert equal(ae.decoder_state(),state['decoder'])
            assert equal(client['classifier'],state['classifier'])
            assert 'state' in client['ae_optimizer'] and client['ae_optimizer']['param_groups']
            assert 'state' in client['classifier_optimizer'] and client['classifier_optimizer']['param_groups']
            assert client['classifier_optimizer']['state']=={}  # SGD has no momentum.
            assert isinstance(client['loader_generator'],torch.Tensor)
        assert 'rng' in state
        checks[name]={'sha256':file_hash(directory/name),'bytes':(directory/name).stat().st_size,
                      'seed':state['seed'],'round':state['round'],'clients':10,'bn_buffers':bn_buffers,
                      'optimizer_dictionaries':20,'classifier_parameters':sum(p.numel() for p in model.parameters())}
    for name in ('encoders','decoder','rng','training_indices','history','configuration','code'):
        assert equal(final[name],before[name]),name
    for cid in before['clients']:
        for name,value in before['clients'][cid].items():
            if name!='classifier':assert equal(final['clients'][cid][name],value),(cid,name)
    for name,value in before['classifier'].items():
        if name not in ('fc.weight','fc.bias') and not name.endswith(('running_mean','running_var','num_batches_tracked')):
            assert equal(final['classifier'][name],value),name
    assert not equal(final['classifier']['fc.weight'],before['classifier']['fc.weight'])
    effective=inference_decoder(final);gain=result['config']['fusion_gain']
    assert torch.equal(effective['final.weight'],gain*before['decoder']['final.weight'])
    assert torch.equal(effective['final.bias'],gain*before['decoder']['final.bias'])
    assert final['recovery']['head_optimizer']['state'] and final['recovery']['head_rng']
    # Reports/summaries cannot conceal changes to earlier experiments.
    changed=subprocess.check_output(['git','diff','--name-only','4bbc24ba12a98c498197bfe3d3187027f49395a3'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith('research/pathmnist_recovery/') for p in changed)
    baseline=json.loads((ROOT/'_local/pathmnist_calibrated/source-v1/manifest.json').read_text())
    for name,digest in baseline['source_sha256'].items():assert file_hash(ROOT/name)==digest
    report={'status':'passed','test_reconstructed_accuracy_percent':reconstructed,
            'test_correct':metric['correct_total'],'test_predictions':71800,'unique_test_images':7180,
            'test_evaluations_in_final_attempt':1,'partition_sha256':partition['partition_sha256'],
            'full_fit_validation_counts':[len(full),len(fit),len(validation)],'checkpoints':checks,
            'encoder_decoder_body_original_optimizers_rng_preserved':True,
            'head_optimizer_and_gain_loadable':True,'source_and_frozen_configuration_hashes_match':True,
            'prepared_training_test_and_original_partition_hashes_match':True,
            'old_scientific_sources_and_previous_experiments_unchanged':True,
            'no_training_or_image_evaluation_in_this_audit':True}
    write_json(directory/'verification.json',report);print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':audit()
