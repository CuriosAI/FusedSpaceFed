"""GPU job receipts; retain infeasible numerical candidates and continue search."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import os

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.pathmnist_pathological.run import file_hash,write_json
PYTHON='/home/schroeder/miniconda3/envs/general_ml/bin/python'


def utc():
    return datetime.now(timezone.utc).isoformat()


def launch(queue_path, receipt_path, logs, allow_existing=False):
    queue=json.loads(queue_path.read_text())
    if receipt_path.exists():
        raise FileExistsError('Campaign receipt already exists')
    for name,digest in queue['frozen_files'].items():
        if file_hash(ROOT/name)!=digest:
            raise ValueError('Changed frozen file '+name)
    for job in queue['jobs']:
        if (ROOT/job['output']).exists() and '--resume' not in job['args']:
            raise FileExistsError('Output exists '+job['output'])
    logs.mkdir(parents=True,exist_ok=True);receipt_path.parent.mkdir(parents=True,exist_ok=True)
    jobs=queue['jobs'].copy();workers=[];began=time.perf_counter()
    state={'status':'running','queue_sha256':file_hash(queue_path),'frozen_files':queue['frozen_files'],
           'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'gpu_slots':queue['gpu_slots'],'started_utc':utc(),'runs':[],
           'external_actions':'none','policy':'keep failures, continue other declared candidates; no process signals'}
    write_json(receipt_path,state)
    while jobs or any(row['exit_code'] is None for _,_,row,_ in workers):
        for process,log,row,start in workers:
            if row['exit_code'] is None and process.poll() is not None:
                row.update(exit_code=process.returncode,ended_utc=utc(),process_wall_seconds=time.perf_counter()-start)
                log.close();write_json(receipt_path,state)
                print(json.dumps({'finished':row['name'],'exit_code':row['exit_code']}),flush=True)
        for device in ('cuda:1','cuda:0'):
            busy=sum(row['device']==device and row['exit_code'] is None for _,_,row,_ in workers)
            if not jobs or busy>=queue['gpu_slots'][device]:continue
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
            free={int(cols[0]):float(cols[1])-float(cols[2]) for line in output.splitlines() if (cols:=line.split(','))}
            if free[int(device[-1])]<queue.get('minimum_free_mib',8192):continue
            job=jobs.pop(0);command=[PYTHON,'-u',job['script'],*job['args'],'--output',job['output'],'--device',device]
            log=(logs/(job['name']+'.log')).open('x');start=time.perf_counter()
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            row={**job,'command':command,'device':device,'pid':process.pid,'started_utc':utc(),'exit_code':None}
            state['runs'].append(row);workers.append((process,log,row,start));write_json(receipt_path,state)
            print(json.dumps({'started':row['name'],'pid':row['pid'],'device':device}),flush=True)
        time.sleep(1)
    state.update(status='completed' if all(row['exit_code']==0 for _,_,row,_ in workers) else 'completed_with_failed_candidates',
                 ended_utc=utc(),wall_seconds=time.perf_counter()-began,
                 process_wall_seconds_sum=sum(row['process_wall_seconds'] for _,_,row,_ in workers))
    write_json(receipt_path,state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--logs',type=Path,required=True)
    args=p.parse_args();launch(args.queue,args.receipt,args.logs)
