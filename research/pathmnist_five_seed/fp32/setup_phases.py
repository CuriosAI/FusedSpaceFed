"""Freeze the probe/anchors and seed-paired queues after the original campaign."""
import argparse
import json
from research.pathmnist_five_seed.data import partition
from research.pathmnist_five_seed.fp32.profile import ROOT,PUBLIC,PRIVATE
from research.pathmnist_five_seed.fp32.diagnostic_common import prepare_probes,freeze_anchors,load_anchor
from research.pathmnist_five_seed.fp32.runner import SOURCES
from research.pathmnist_pathological.run import file_hash,write_json


def prepare(phase):
    prepare_probes();freeze_anchors();folder=PUBLIC/f'phase{phase}';folder.mkdir(exist_ok=True)
    plan=json.loads((PUBLIC/'plan.json').read_text());jobs=[];extra=[]
    if phase==2:
        cfgfolder=folder/'configs';cfgfolder.mkdir(exist_ok=True)
        # Interleave seed-paired variants; no candidate search or early stopping.
        for seed in plan['seeds']:
            _,anchor=load_anchor(seed,'initialization');_,ps=partition(seed)
            for variant in plan['variants'][1:]:
                path=cfgfolder/f'{variant}-seed-{seed}.json'
                cfg={'seed':seed,'variant':variant,'settings':plan['settings'],'partition_sha256':ps['sha256'],
                     'paired_initial_checkpoint':anchor['path'],'paired_initial_sha256':anchor['sha256']}
                if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Changed paired config')
                write_json(path,cfg);extra.append(str(path.relative_to(ROOT)))
                jobs.append({'name':f'{variant}-seed-{seed}','script':'research/pathmnist_five_seed/fp32/runner.py',
                             'args':['--config',str(path.relative_to(ROOT))],'output':str((PRIVATE/variant/f'seed-{seed}').relative_to(ROOT))})
    else:
        script=f'research/pathmnist_five_seed/fp32/phase{phase}/'+('diagnose.py' if phase==1 else 'probe_gradients.py')
        extra.append(script)
        if phase==3:extra+=['research/digits_mechanism_diagnostics/phase3/gradient_stats.py',
                           'research/digits_mechanism_diagnostics/phase3/probe_gradients.py']
        for seed in plan['seeds']:
            jobs.append({'name':f'phase{phase}-seed-{seed}','script':script,'args':['--seed',str(seed)],
                         'output':str((PRIVATE/f'phase{phase}'/f'seed-{seed}').relative_to(ROOT))})
    names=(*SOURCES,'research/pathmnist_five_seed/diagnostic_common.py',
           'research/pathmnist_five_seed/fp32/diagnostic_common.py','research/pathmnist_five_seed/fp32/probe.json',
           'research/pathmnist_five_seed/fp32/anchors.json','research/pathmnist_calibrated/controller.py',
           'research/digits_mechanism_diagnostics/common.py',*extra)
    write_json(folder/'queue.json',{'gpu_slots':{'cuda:1':3,'cuda:0':3},'minimum_free_mib':8192,
                                  'frozen_files':{name:file_hash(ROOT/name) for name in names},'jobs':jobs})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',type=int,choices=(1,2,3))
    prepare(p.parse_args().phase)
