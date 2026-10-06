"""Isolated adapter to the unchanged complete-checkpoint PathMNIST runner."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from fusedspacefed_core import clone_state_dict
from research.pathmnist_pathological.run import file_hash
from research.pathmnist_calibrated import runner as original
from research.pathmnist_recovery.method import build_classifier,RecoveryClient

SOURCES=('fusedspacefed_core.py','research/pathmnist_pathological/run.py',
         'research/pathmnist_calibrated/data.py','research/pathmnist_calibrated/client.py',
         'research/pathmnist_calibrated/runner.py','research/pathmnist_calibrated/partition.json.gz',
         'research/pathmnist_calibrated/search_plan.json',
         'research/pathmnist_recovery/method.py','research/pathmnist_recovery/training.py',
         'research/pathmnist_recovery/training_plan.json')


def validate(config,partition):
    plan=json.loads((ROOT/'research/pathmnist_recovery/training_plan.json').read_text())
    original_plan=json.loads((ROOT/'research/pathmnist_calibrated/search_plan.json').read_text())
    candidates={**original_plan['candidates'],**plan['candidates']}
    if config['candidate_id'] not in candidates or config['settings']!=candidates[config['candidate_id']]:
        raise ValueError('Unregistered/changed recovery profile')
    if config['partition_sha256']!=partition['partition_sha256'] or config['rounds']!=50:
        raise ValueError('Changed partition or final horizon')
    if config['stage']=='validation':
        if config['seed']!=142:raise ValueError('Unexpected validation seed')
    elif config['stage']=='final':
        selected_path=ROOT/config['selection_file']
        selected=json.loads(selected_path.read_text())
        if file_hash(selected_path)!=config['selection_sha256']:raise ValueError('Changed frozen selection')
        if config['candidate_id']!=selected['candidate_id'] or config['settings']!=selected['settings'] or config['bn_mode']!=selected['bn_mode'] or config['seed']!=42:
            raise ValueError('Final configuration differs from validation selection')
        if config['settings'].get('classifier_normalization')=='groupnorm8' and config['bn_mode']!='native':
            raise ValueError('GN has no running BN calibration')
    else:raise ValueError('Unregistered recovery stage')
    return plan


@contextmanager
def adapted(settings):
    names=('SOURCES','CalibratedClient','ResNet20V2','validate_configuration','recalibrate_bn')
    saved={name:getattr(original,name) for name in names}
    try:
        original.SOURCES=SOURCES
        original.CalibratedClient=RecoveryClient
        original.ResNet20V2=lambda num_classes,in_channels:build_classifier(settings)
        original.validate_configuration=validate
        if settings.get('classifier_normalization')=='groupnorm8':
            original.recalibrate_bn=lambda classifier,*args,**kwargs:clone_state_dict(classifier)
        yield
    finally:
        for name,value in saved.items():setattr(original,name,value)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();config=json.loads(a.config.read_text())
    with adapted(config['settings']):original.run(config,a.output,a.device,a.resume)
