"""Prepare exactly three matched final pairs after the one immutable selection."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from research.feature_shift_digits.data import atomic_json,canonical_hash,file_hash

public=REPO/'research/capacity_compute_control'
if (public/'final_queue.json').exists():
    raise FileExistsError('Final registry already exists')
selection=json.loads((public/'selection.json').read_text())
plan=json.loads((public/'search_plan_v2.json').read_text())
profile=json.loads((public/'flop_profile.json').read_text());split=json.loads((public/'validation_split.json').read_text())
original=json.loads((REPO/'research/feature_shift_digits/config_five.json').read_text())
settings={method:{**plan['training'],**chosen} for method,chosen in selection['selected_training_settings'].items()}
compatible=settings['FusedSpaceFed']==original['training']
proof={'decision_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':canonical_hash(selection),
       'reused_fused':compatible,'all_training_fields_identical':compatible,
       'differences':{key:{'selected':value,'previous':original['training'].get(key)} for key,value in settings['FusedSpaceFed'].items() if value!=original['training'].get(key)},
       'rule':'reuse only exact full training settings + partition + source bytes; otherwise all three Fused fresh',
       'architecture_unchanged':True,'partition_sha256':plan['parent_partition_sha256']}
jobs=[];registry=[]
for seed in (42,43,44):
    for method in ('FusedSpaceFed','FedAvg'):
        name=f'{method}-seed-{seed}';config={'phase':'final','method':method,'seed':seed,
            'parent_partition_sha256':plan['parent_partition_sha256'],
            'profile_file':'research/capacity_compute_control/flop_profile.json','profile_sha256':profile['profile_sha256'],
            'validation_file':'research/capacity_compute_control/validation_split.json','validation_sha256':split['validation_sha256'],
            'selection_file':'research/capacity_compute_control/selection.json','selection_sha256':canonical_hash(selection),
            'training':settings[method],'evaluation':'fixed round300 final, no adaptation'}
        path=public/'configs/final'/f'{name}.json';path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,config)
        relative=str(path.relative_to(REPO))
        if method=='FusedSpaceFed' and compatible:
            directory=Path('_local/feature_shift_digits/runs')/f'seed-{seed}'
            result=json.loads((REPO/directory/'results.json').read_text())
            if result['status']!='completed' or result['completed_rounds']!=300 or result['identity']['seed']!=seed or result['identity']['partition_sha256']!=plan['parent_partition_sha256'] or result['configuration']['training']!=settings[method]:
                raise ValueError('Reused run is not compatible/complete')
            for source,digest in result['identity']['code']['source_sha256'].items():
                if file_hash(REPO/source)!=digest:
                    raise ValueError('Reused source changed')
            registry.append({'name':name,'method':method,'seed':seed,'config':relative,'origin':'reused_previous',
                             'source_directory':str(directory),'reused_result_sha256':file_hash(REPO/directory/'results.json')})
        else:
            directory=f'_local/capacity_compute_control/final/{name}'
            if (REPO/directory).exists():
                raise FileExistsError('New run output already exists')
            jobs.append({'name':name,'config':relative,'output':directory})
            registry.append({'name':name,'method':method,'seed':seed,'config':relative,'origin':'new','source_directory':directory})
atomic_json(public/'reuse_proof.json',proof)
atomic_json(public/'final_queue.json',{'jobs':jobs,'runs_registry':registry,'selection_sha256':canonical_hash(selection),
                                     'freeze_utc':datetime.now(timezone.utc).isoformat()})
print(json.dumps({'selected':selection['selected_training_settings'],'reuse_fused':compatible,'new_jobs':len(jobs),'proof':proof},indent=2))
