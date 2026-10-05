"""Queue three additional seeds after the original pair exits successfully."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from research.feature_shift_digits.data import atomic_json,canonical_hash,file_hash,verify
from research.feature_shift_digits.run_registered_seed import validate_extension,scientific_configuration

PYTHON='/home/schroeder/miniconda3/envs/general_ml/bin/python'
PRIVATE=REPO/'_local/feature_shift_digits'
PUBLIC=REPO/'research/feature_shift_digits'


def main():
    original=json.loads((PUBLIC/'config.json').read_text())
    config=json.loads((PUBLIC/'config_five.json').read_text())
    validate_extension(original,config)
    proof=json.loads((PUBLIC/'five_run_authorization.json').read_text())
    if proof['extended_config_sha256']!=canonical_hash(config) or proof['scientific_config_sha256']!=canonical_hash(scientific_configuration(config)):
        raise ValueError('Frozen registry/scientific configuration differs')
    manifest=verify(PRIVATE/'prepared')
    if manifest['partition_sha256']!=config['partition_sha256']:
        raise ValueError('Frozen data differ')
    if any((PRIVATE/'runs'/f'seed-{seed}').exists() for seed in (44,45,46)):
        raise FileExistsError('Additional output exists; explicit individual resume required')
    destination=PRIVATE/'campaign_extension.json'
    if destination.exists():
        raise FileExistsError('Do not overwrite extension campaign')
    record={'status':'waiting_original_pair','queued_seeds':[44,45,46],'config_sha256':canonical_hash(config),
            'scientific_config_sha256':canonical_hash(scientific_configuration(config)),
            'partition_sha256':manifest['partition_sha256'],'scheduler_sha256':file_hash(Path(__file__)),
            'authorization_sha256':file_hash(PUBLIC/'five_run_authorization.json'),
            'maximum_training_processes':2,'maximum_per_gpu':1,'external_actions':'none','runs':[]}
    atomic_json(destination,record)
    while True:
        pair=json.loads((PRIVATE/'campaign.json').read_text())
        if pair['status']=='failed':
            raise RuntimeError('Original pair failed; do not start further seeds')
        if pair['status']=='completed' and all(row['exit_code']==0 for row in pair['runs']):
            break
        time.sleep(2)
    record['original_pair_sha256']=file_hash(PRIVATE/'campaign.json')
    record['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    record['started_utc']=datetime.now(timezone.utc).isoformat()
    record['status']='running'
    started=time.perf_counter()
    atomic_json(destination,record)
    workers=[]
    pending=[44,45,46]
    logs=PRIVATE/'logs';logs.mkdir(parents=True,exist_ok=True)
    while pending or any(item['exit_code'] is None for _,_,item,_ in workers):
        for process,log,item,worker_started in workers:
            if item['exit_code'] is None and process.poll() is not None:
                item['exit_code']=process.returncode
                item['ended_utc']=datetime.now(timezone.utc).isoformat()
                item['process_wall_seconds']=time.perf_counter()-worker_started
                log.close()
                atomic_json(destination,record)
                print(json.dumps({'finished_seed':item['seed'],'exit_code':process.returncode}),flush=True)
        if any(item['exit_code'] not in (None,0) for _,_,item,_ in workers):
            record['not_started_due_to_error']=pending.copy();pending.clear()
        busy={item['device'] for _,_,item,_ in workers if item['exit_code'] is None}
        for seed in pending.copy():
            device=config['per_seed_device'][str(seed)]
            if device in busy:
                continue
            snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
            free={int(fields[0]):float(fields[1])-float(fields[2]) for row in snapshot.strip().splitlines() if (fields:=row.split(','))}
            if free[int(device.split(':')[1])]<2048:
                continue
            command=[PYTHON,'research/feature_shift_digits/run_registered_seed.py','--config','research/feature_shift_digits/config_five.json',
                     '--partition','_local/feature_shift_digits/prepared','--output',f'_local/feature_shift_digits/runs/seed-{seed}',
                     '--seed',str(seed),'--device',device]
            log=(logs/f'seed-{seed}.log').open('x')
            worker_started=time.perf_counter()
            process=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            item={'seed':seed,'device':device,'command':command,'pid':process.pid,'exit_code':None,
                  'started_utc':datetime.now(timezone.utc).isoformat()}
            record['runs'].append(item);workers.append((process,log,item,worker_started))
            pending.remove(seed);busy.add(device)
            record['queued_seeds']=pending.copy()
            atomic_json(destination,record)
            print(json.dumps({'started_seed':seed,'device':device}),flush=True)
        time.sleep(1)
    record['elapsed_wall_seconds']=time.perf_counter()-started
    record['process_wall_seconds_sum']=sum(item['process_wall_seconds'] for item in record['runs'])
    record['ended_utc']=datetime.now(timezone.utc).isoformat()
    record['status']='completed' if len(record['runs'])==3 and all(item['exit_code']==0 for item in record['runs']) else 'failed'
    atomic_json(destination,record)
    print(json.dumps({'extension_status':record['status']}),flush=True)
    return 0 if record['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
