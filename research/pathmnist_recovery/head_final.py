"""One frozen training-only shared-head refit, followed by one terminal test."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fusedspacefed_core import seed_everything
from research.pathmnist_pathological.run import file_hash,write_json,atomic_save,assert_finite,rng_state
from research.pathmnist_calibrated.data import verify,cache,DATA
from research.pathmnist_calibrated import runner
from research.pathmnist_recovery.training import adapted
from research.pathmnist_recovery.fusion_probe import scaled_decoder
from research.pathmnist_recovery.head_calibration import features,refit
from research.pathmnist_recovery.inference import inference_decoder


def substitute_head(checkpoint,classifier):
    """Change only shared fc weights and BN buffers; retain complete states."""
    original=checkpoint['classifier']
    if set(original)!=set(classifier):raise ValueError('Changed classifier architecture')
    for name,value in original.items():
        if name not in ('fc.weight','fc.bias') and not name.endswith(
                ('running_mean','running_var','num_batches_tracked')):
            if not torch.equal(value,classifier[name]):
                raise ValueError('Head phase changed classifier body weights')
    result=copy.deepcopy(checkpoint);result['classifier']=copy.deepcopy(classifier)
    for client in result['clients'].values():client['classifier']=copy.deepcopy(classifier)
    assert_finite(result);return result


def main(config_path,output,physical_device):
    started=time.perf_counter();os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1]
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2);seed_everything(42)
    config=json.loads(config_path.read_text());partition=verify()
    selection_file=ROOT/config['head_selection_file']
    if file_hash(selection_file)!=config['head_selection_sha256']:
        raise ValueError('Changed frozen validation selection')
    selection=json.loads(selection_file.read_text());selected=selection['selected']
    if selection['settings']!=config['training_settings'] or any(
            selected[a]!=config[b] for a,b in (('gain','fusion_gain'),('penalty','head_penalty'),
                                               ('max_iter','head_max_iter'),('mode','normalization'))):
        raise ValueError('Final head phase differs from validation selection')
    if config['test_access']!='one frozen candidate; known benchmark; exploratory threshold stop':
        raise ValueError('Undeclared exploratory test access')
    if partition['partition_sha256']!=config['partition_sha256']:raise ValueError('Changed partition')
    source=ROOT/config['checkpoint'];initial=ROOT/config['initial_checkpoint']
    if file_hash(source)!=config['checkpoint_sha256'] or file_hash(initial)!=config['initial_checkpoint_sha256']:
        raise ValueError('Changed complete checkpoint')
    state=torch.load(source,map_location='cpu',weights_only=False);assert_finite(state)
    if state['seed']!=42 or state['round']!=50 or state['training_indices']!=partition['full']:
        raise ValueError('Wrong final seed/horizon/training split')
    if state['configuration']['settings']!=config['training_settings']:
        raise ValueError('Incompatible training settings')
    if state['configuration']['candidate_id']!=selected['candidate_id']:
        raise ValueError('Wrong selected training profile')
    if config['fusion_gain']<=0 or config['normalization']!='train-recalibrated':
        raise ValueError('Unregistered final inference')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(initial,output/'initial.pt');shutil.copyfile(source,output/'precalibration.pt')
    images,labels=cache(device);settings=config['training_settings']
    decoder=scaled_decoder(state['decoder'],config['fusion_gain'])
    code={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_sha256':{p:file_hash(ROOT/p) for p in (
              'research/pathmnist_recovery/head_final.py','research/pathmnist_recovery/head_calibration.py',
              'research/pathmnist_recovery/inference.py','research/pathmnist_recovery/fusion_probe.py',
              'research/pathmnist_recovery/training.py','research/pathmnist_recovery/method.py',
              'research/pathmnist_calibrated/runner.py','research/pathmnist_calibrated/data.py','fusedspacefed_core.py')},
          'config_sha256':file_hash(config_path)}
    with adapted(settings):
        phase_started=time.perf_counter()
        classifier=runner.recalibrate_bn(state['classifier'],decoder,state['encoders'],images,partition,device)
        bn_seconds=time.perf_counter()-phase_started;phase_started=time.perf_counter()
        xx,yy,owners=features(classifier,decoder,state['encoders'],images,labels,partition['full'],settings,device)
        feature_seconds=time.perf_counter()-phase_started;phase_started=time.perf_counter()
        fitted,optimizer,statistics=refit(classifier,xx,yy,owners,config['head_penalty'],config['head_max_iter'])
        refit_seconds=time.perf_counter()-phase_started
        final=substitute_head(state,fitted)
        final['recovery']={'config':config,'code':code,'head_optimizer':optimizer,'head_statistics':statistics,
                           'head_rng':rng_state(cuda=True),'original_checkpoint_sha256':file_hash(source),
                           'encoder_decoder_body_weights_unchanged':True,
                           'decoder_representation':'original trained decoder; apply stored fusion_gain via inference_decoder()',
                           'original_training_resume_source':str(output/'precalibration.pt')}
        atomic_save(output/'final.pt',final)
        # Test is loaded only after the final inference state has been saved.
        test_images=torch.from_numpy(np.load(DATA/'test-images.npy')).to(device)
        test_labels=torch.from_numpy(np.load(DATA/'test-labels.npy')).to(device)
        phase_started=time.perf_counter()
        metric=runner.evaluate(final['classifier'],inference_decoder(final),final['encoders'],
                               test_images,test_labels,list(range(7180)),settings,device)
        evaluation_seconds=time.perf_counter()-phase_started
    result={'status':'completed','kind':'extra training-only shared-head phase; positive fusion/BN variant',
            'seed':42,'round':50,'test_evaluations':1,'config':config,'code':code,'metrics':metric,
            'above_target':metric['uniform_pipeline_accuracy_percent']>50.94,
            'head_statistics':statistics,'head_training_samples':len(xx),'bn_training_samples':6400,
            'bn_seconds':bn_seconds,'feature_extraction_seconds':feature_seconds,'head_refit_seconds':refit_seconds,
            'evaluation_seconds':evaluation_seconds,'session_seconds':time.perf_counter()-started,
            'reused_training_seconds':state['training_seconds'],
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'physical_device':physical_device,
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint_sha256':{p.name:file_hash(p) for p in output.glob('*.pt')}}
    assert_finite(result);write_json(output/'results.json',result)
    print(json.dumps({'status':'completed','accuracy_percent':metric['uniform_pipeline_accuracy_percent'],
                       'above_target':result['above_target'],'session_seconds':result['session_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True);a=p.parse_args();main(a.config,a.output,a.device)
