"""Two GPU queue, one worker/GPU, no external process actions/overwrites."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from research.feature_shift_digits.data import atomic_json,file_hash

PYTHON='/home/schroeder/miniconda3/envs/general_ml/bin/python'


def utc():
    return datetime.now(timezone.utc).isoformat()


def launch(queue_file,receipt,logs):
    queue=json.loads(queue_file.read_text());pending=queue['jobs'].copy()
    if receipt.exists():
        raise FileExistsError('Controller receipt already exists')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit the implementation/configs before launching')
    for job in pending:
        if (REPO/job['output']).exists():
            raise FileExistsError('Output exists; no implicit restart/resume')
    logs.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    state={'status':'running','queue_sha256':file_hash(queue_file),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'started_utc':utc(),'runs':[],'max_processes':2,'max_per_gpu':1,'external_actions':'none'}
    atomic_json(receipt,state);workers=[]
    while pending or any(row['exit_code'] is None for _,_,row,_ in workers):
        for process,log,row,begin in workers:
            if row['exit_code'] is None and process.poll() is not None:
                row.update(exit_code=process.returncode,ended_utc=utc(),process_wall_seconds=time.perf_counter()-begin)
                log.close();atomic_json(receipt,state);print(json.dumps({'finished':row['name'],'exit_code':process.returncode}),flush=True)
                if process.returncode:
                    state['status']='failed';state['pending_not_started']=[j['name'] for j in pending];pending=[]
        busy={row['device'] for _,_,row,_ in workers if row['exit_code'] is None}
        for device in ('cuda:1','cuda:0'):
            if device in busy or not pending:
                continue
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
            available={int(cols[0]):float(cols[1])-float(cols[2]) for line in text.strip().splitlines() if (cols:=line.split(','))}
            if available[int(device[-1])]<2048:
                continue
            job=pending.pop(0)
            command=[PYTHON,'research/capacity_compute_control/runner.py','--config',job['config'],'--partition','_local/feature_shift_digits/prepared',
                     '--output',job['output'],'--device',device]
            log=(logs/f"{job['name']}.log").open('x');begin=time.perf_counter()
            process=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            row={**job,'device':device,'command':command,'pid':process.pid,'started_utc':utc(),'exit_code':None}
            state['runs'].append(row);workers.append((process,log,row,begin));atomic_json(receipt,state)
            print(json.dumps({'started':job['name'],'device':device,'pid':process.pid}),flush=True)
        time.sleep(1)
    state.update(ended_utc=utc(),wall_seconds=time.perf_counter()-started,
                 process_wall_seconds_sum=sum(row['process_wall_seconds'] for _,_,row,_ in workers))
    if state['status']!='failed':
        state['status']='completed'
    atomic_json(receipt,state)
    return 0 if state['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--logs',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(launch(args.queue,args.receipt,args.logs))
