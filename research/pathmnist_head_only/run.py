"""One head-only refit on the original PathMNIST paper-settings checkpoint."""
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fusedspacefed_core import ResNet20V2, UNetSmallAE, seed_everything
from research.pathmnist_pathological.run import file_hash, write_json, atomic_save, assert_finite, rng_state
from research.pathmnist_calibrated.data import verify, cache, DATA
from research.pathmnist_calibrated.runner import evaluate
from research.pathmnist_recovery.head_calibration import features, refit

ORIGINAL_CHECKPOINT_SHA256='fa56da01091debe9031a9acb2e3de0f34ce0f751d2c95d2d0d54c05f15697d1e'
SOURCES=('research/pathmnist_head_only/run.py','research/pathmnist_head_only/config.json',
         'research/pathmnist_recovery/head_calibration.py','research/pathmnist_recovery/method.py',
         'research/pathmnist_recovery/training.py','research/pathmnist_recovery/fusion_probe.py',
         'research/pathmnist_calibrated/data.py','research/pathmnist_calibrated/runner.py',
         'research/pathmnist_calibrated/client.py',
         'research/pathmnist_pathological/run.py','research/pathmnist_pathological/config.json',
         'fusedspacefed_core.py')


def replace_head_only(source,classifier):
    """Copy all original training states; change exactly shared fc weights/bias."""
    original=source['classifier']
    if set(original)!=set(classifier):raise ValueError('Changed classifier architecture')
    for name,value in original.items():
        if name not in ('fc.weight','fc.bias') and not torch.equal(value,classifier[name]):
            raise ValueError('Refit changed a frozen classifier weight or BN buffer: '+name)
    result=copy.deepcopy(source);result['classifier']=copy.deepcopy(classifier)
    for client in result['clients'].values():client['classifier']=copy.deepcopy(classifier)
    assert_finite(result);return result


def tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def validate(config,state,partition):
    expected=json.loads((ROOT/'research/pathmnist_pathological/config.json').read_text())
    if state['config']!=expected or state['seed']!=42 or state['round']!=50:
        raise ValueError('Requires original paper-settings seed42 round50 state')
    if config['checkpoint_sha256']!=ORIGINAL_CHECKPOINT_SHA256:
        raise ValueError('Requires immutable original checkpoint, not tuned weights')
    if config['fusion_gain']!=1 or config['bn_mode']!='original-frozen' or config['head_max_iter']!=100 or config['head_penalty']!=0:
        raise ValueError('Head-only protocol differs from user-fixed settings')
    if config['seed']!=42 or config['source_round']!=50 or config['lbfgs']!={
            'lr':1.,'history_size':20,'line_search_fn':'strong_wolfe',
            'tolerance_grad':1e-7,'tolerance_change':1e-10}:
        raise ValueError('Changed original seed or fixed L-BFGS procedure')
    if config['source_partition_sha256']!=state['partition_sha256'] or state['partitions']['clients']!=partition['full']:
        raise ValueError('Changed original training partition')
    expected_clients={str(i) for i in range(10)}
    if set(state['clients'])!=expected_clients or set(state['encoders'])!=expected_clients:
        raise ValueError('Incomplete clients/encoders')


def check_complete(source,final):
    for name,value in source.items():
        if name not in ('classifier','clients') and not tree_equal(final[name],value):
            raise AssertionError('Modified original state '+name)
    for cid,client in source['clients'].items():
        for name,value in client.items():
            if name!='classifier' and not tree_equal(final['clients'][cid][name],value):
                raise AssertionError('Modified client state '+str((cid,name)))
        for name,value in client['classifier'].items():
            if name not in ('fc.weight','fc.bias') and not torch.equal(value,final['clients'][cid]['classifier'][name]):
                raise AssertionError('Modified original client classifier body/BN '+str((cid,name)))
        if not tree_equal(final['clients'][cid]['classifier'],final['classifier']):
            raise AssertionError('Private classifier copy not synchronized')
        ae=UNetSmallAE(3,16);ae.load_state_dict(final['clients'][cid]['autoencoder'])
        if not tree_equal(ae.encoder_state(),final['encoders'][cid]) or not tree_equal(ae.decoder_state(),final['decoder']):
            raise AssertionError('Incomplete private AE state')
        model=ResNet20V2(9,3);model.load_state_dict(final['clients'][cid]['classifier'])
    for name,value in source['classifier'].items():
        if name not in ('fc.weight','fc.bias') and not torch.equal(value,final['classifier'][name]):
            raise AssertionError('Modified original classifier body/BN '+name)
    if torch.equal(source['classifier']['fc.weight'],final['classifier']['fc.weight']):
        raise AssertionError('Shared head did not change')
    assert_finite(final)


def verify_counts(metric):
    rows=metric['pipeline_metrics']
    if len(rows)!=10 or {r['client_id'] for r in rows}!=set(range(10)) or any(r['total']!=7180 for r in rows):
        raise AssertionError('Wrong test population')
    for row in rows:
        if abs(row['accuracy_percent']-100*row['correct']/row['total'])>1e-10:
            raise AssertionError('Wrong pipeline accuracy')
    if metric['correct_total']!=sum(r['correct'] for r in rows) or metric['predictions_total']!=71800:
        raise AssertionError('Wrong aggregate counts')
    if abs(metric['uniform_pipeline_accuracy_percent']-100*metric['correct_total']/71800)>1e-10:
        raise AssertionError('Wrong uniform pipeline accuracy')


def main(config_path,output,physical_device):
    started=time.perf_counter()
    os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1]
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2);seed_everything(42)
    config=json.loads(config_path.read_text());source=ROOT/config['checkpoint']
    if file_hash(source)!=ORIGINAL_CHECKPOINT_SHA256:raise ValueError('Original checkpoint changed')
    state=torch.load(source,map_location='cpu',weights_only=False);assert_finite(state)
    partition=verify();validate(config,state,partition)
    for name,digest in state['data_manifest']['files_sha256'].items():
        if file_hash(DATA/name)!=digest:raise ValueError('Changed source data '+name)
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,output/'before.pt')
    shutil.copyfile(ROOT/'_local/pathmnist_pathological/seed-42/initial.pt',output/'training-initial.pt')
    code={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_sha256':{name:file_hash(ROOT/name) for name in SOURCES},'config_sha256':file_hash(config_path)}
    images,labels=cache(device);settings={'classifier_normalization':'batchnorm'}
    before=time.perf_counter()
    # eval/no_grad feature extraction leaves all original parameters/BN frozen.
    xx,yy,owners=features(state['classifier'],state['decoder'],state['encoders'],images,labels,
                          state['partitions']['clients'],settings,device)
    feature_seconds=time.perf_counter()-before;before=time.perf_counter()
    fitted,optimizer,statistics=refit(state['classifier'],xx,yy,owners,0.,100)
    refit_seconds=time.perf_counter()-before
    head_state=next(iter(optimizer['state'].values()))
    statistics.update(actual_iterations=head_state['n_iter'],function_evaluations=head_state['func_evals'])
    final=replace_head_only(state,fitted)
    final['head_only_refit']={'configuration':config,'code':code,'head_optimizer':optimizer,
                              'statistics':statistics,'phase_rng':rng_state(cuda=True),
                              'training_samples':len(xx),'frozen':'private E, shared D, classifier body and all BN buffers',
                              'formula':'x+D(E_i(x))','original_checkpoint_sha256':file_hash(source)}
    atomic_save(output/'final.pt',final)
    # Save the final head first; neither test score can affect training/settings.
    loaded=torch.load(output/'final.pt',map_location='cpu',weights_only=False)
    check_complete(state,loaded)
    test_images=torch.from_numpy(np.load(DATA/'test-images.npy')).to(device)
    test_labels=torch.from_numpy(np.load(DATA/'test-labels.npy')).to(device)
    before=time.perf_counter()
    metrics_before=evaluate(state['classifier'],state['decoder'],state['encoders'],test_images,test_labels,
                            list(range(7180)),settings,device)
    metrics_after=evaluate(loaded['classifier'],loaded['decoder'],loaded['encoders'],test_images,test_labels,
                           list(range(7180)),settings,device)
    test_seconds=time.perf_counter()-before
    verify_counts(metrics_before);verify_counts(metrics_after)
    historical=json.loads((ROOT/'research/pathmnist_pathological/artifacts/results.json').read_text())
    historical_counts={r['client_id']:r['correct'] for r in historical['client_test_metrics']}
    if historical_counts!={r['client_id']:r['correct'] for r in metrics_before['pipeline_metrics']}:
        raise AssertionError('Native before accuracy does not reproduce original pipeline counts')
    result={'status':'completed','seed':42,'source_round':50,'configuration':config,'code':code,
            'test_evaluations':2,'metrics_before':metrics_before,'metrics_after':metrics_after,
            'delta_pp':metrics_after['uniform_pipeline_accuracy_percent']-metrics_before['uniform_pipeline_accuracy_percent'],
            'training_samples':len(xx),'head_statistics':statistics,
            'feature_extraction_seconds':feature_seconds,'head_refit_seconds':refit_seconds,'test_seconds':test_seconds,
            'session_seconds':time.perf_counter()-started,'original_training_seconds':state['elapsed_training_seconds'],
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'physical_device':physical_device,
            'runtime':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                       'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(device),
                       'cudnn_deterministic':torch.backends.cudnn.deterministic,
                       'cudnn_benchmark':torch.backends.cudnn.benchmark},
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'verification':{'complete_checkpoint_loaded':True,'only_fc_changed':True,'all_BN_buffers_bit_identical':True,
                            'private_encoder_shared_decoder_optimizer_scaler_rng_and_partition_preserved':True,
                            'original_before_test_counts_reproduced':True,'no_test_or_validation_used_for_refit':True},
            'checkpoint_sha256':{p.name:file_hash(p) for p in output.glob('*.pt')}}
    for name,digest in code['source_sha256'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Sources changed during refit')
    assert_finite(result);write_json(output/'results.json',result)
    print(json.dumps({'status':'completed','before_percent':metrics_before['uniform_pipeline_accuracy_percent'],
                       'after_percent':metrics_after['uniform_pipeline_accuracy_percent'],'delta_pp':result['delta_pp'],
                       'session_seconds':result['session_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True);a=p.parse_args();main(a.config,a.output,a.device)
