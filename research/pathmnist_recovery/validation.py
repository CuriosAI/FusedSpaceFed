"""Inference candidates evaluated only on the globally disjoint fit holdout."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from fusedspacefed_core import seed_everything
from research.pathmnist_pathological.run import file_hash,write_json,atomic_save,assert_finite
from research.pathmnist_calibrated.data import verify,cache
from research.pathmnist_calibrated.runner import evaluate
from research.pathmnist_recovery.normalization import calibrate


def main(config_path,output,physical_device):
    os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1]
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2)
    config=json.loads(config_path.read_text());p=verify()
    source=ROOT/config['checkpoint']
    if file_hash(source)!=config['checkpoint_sha256']:raise ValueError('Changed source checkpoint')
    state=torch.load(source,map_location='cpu',weights_only=False);assert_finite(state)
    if state['training_indices']!=p['fit']:raise ValueError('Validation was used in training')
    if state['seed']!=142:raise ValueError('Unexpected selection seed')
    if p['partition_sha256']!=config['partition_sha256']:raise ValueError('Changed partition')
    output.mkdir(parents=True,exist_ok=False);seed_everything(142)
    started=time.perf_counter();images,labels=cache(device)
    validation=[i for cid in range(10) for i in p['validation'][str(cid)]]
    modes={}
    for mode in config['modes']:
        begin=time.perf_counter()
        classifier=calibrate(state['classifier'],state['decoder'],state['encoders'],images,p,device,mode)
        metric=evaluate(classifier,state['decoder'],state['encoders'],images,labels,validation,{},device)
        modes[mode]={'metrics':metric,'seconds':time.perf_counter()-begin}
        atomic_save(output/(mode+'.pt'),{'classifier':classifier,'mode':mode,'source_sha256':file_hash(source)})
        print(json.dumps({'mode':mode,'validation_accuracy_percent':metric['uniform_pipeline_accuracy_percent']}),flush=True)
    result={'status':'completed','split':'training-holdout','seed':142,'round':state['round'],
            'config':config,'modes':modes,'session_seconds':time.perf_counter()-started,
            'physical_device':physical_device,'validation_samples':len(validation),
            'code':{'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'source_sha256':{f:file_hash(ROOT/f) for f in ('research/pathmnist_recovery/validation.py',
                        'research/pathmnist_recovery/normalization.py','research/pathmnist_calibrated/runner.py')}},
            'completed_utc':datetime.now(timezone.utc).isoformat()}
    assert_finite(result);write_json(output/'results.json',result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--device',required=True)
    v=a.parse_args();main(v.config,v.output,v.device)
