"""Fit-only validation of positive additive fusion gains; alpha0 is diagnostic."""
import argparse
import copy
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
from research.pathmnist_pathological.run import file_hash,write_json,assert_finite
from research.pathmnist_calibrated.data import verify,cache
from research.pathmnist_calibrated import runner
from research.pathmnist_recovery.training import adapted


def scaled_decoder(decoder,gain):
    if gain<0:raise ValueError('Negative fusion gain')
    result=copy.deepcopy(decoder)
    # UNetSmallAE has a linear final Conv2d, so this exactly gives gain*D(E(x)).
    result['final.weight'].mul_(gain);result['final.bias'].mul_(gain)
    return result


def main(config_path,output,physical_device):
    os.environ['CUDA_VISIBLE_DEVICES']=physical_device.split(':')[-1]
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2);seed_everything(142)
    config=json.loads(config_path.read_text());p=verify();source=ROOT/config['checkpoint']
    if file_hash(source)!=config['checkpoint_sha256']:raise ValueError('Changed checkpoint')
    state=torch.load(source,map_location='cpu',weights_only=False);assert_finite(state)
    if state['training_indices']!=p['fit'] or state['seed']!=142:raise ValueError('Validation is not independent')
    output.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    images,labels=cache(device);ids=[i for cid in range(10) for i in p['validation'][str(cid)]]
    rows=[];settings=state['configuration']['settings']
    with adapted(settings):
        for gain in config['gains']:
            decoder=scaled_decoder(state['decoder'],gain)
            for mode in config['modes']:
                classifier=state['classifier'] if mode=='native' else runner.recalibrate_bn(state['classifier'],decoder,state['encoders'],images,p,device)
                result=runner.evaluate(classifier,decoder,state['encoders'],images,labels,ids,settings,device)
                row={'gain':gain,'mode':mode,'metrics':result,'eligible_as_fused':gain>0}
                rows.append(row);print(json.dumps({'gain':gain,'mode':mode,'validation_accuracy':result['uniform_pipeline_accuracy_percent']}),flush=True)
    result={'status':'completed','split':'training-holdout','seed':142,'round':state['round'],
            'config':config,'rows':rows,'session_seconds':time.perf_counter()-began,'physical_device':physical_device,
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'code':{'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'source_sha256':{name:file_hash(ROOT/name) for name in ('research/pathmnist_recovery/fusion_probe.py','research/pathmnist_recovery/training.py','research/pathmnist_recovery/method.py','research/pathmnist_calibrated/runner.py')}}}
    assert_finite(result);write_json(output/'results.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True)
    a=p.parse_args();main(a.config,a.output,a.device)
