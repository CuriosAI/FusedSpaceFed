"""Launch only the two frozen definitive runs, one process per assigned GPU."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.feature_shift_digits.data import atomic_json, canonical_hash, file_hash, verify

PYTHON = '/home/schroeder/miniconda3/envs/general_ml/bin/python'
CONFIG = REPO / 'research/feature_shift_digits/config.json'
PRIVATE = REPO / '_local/feature_shift_digits'


def main():
    config = json.loads(CONFIG.read_text())
    if config['run_seeds'] != [42,43] or config['per_seed_device'] != {'42':'cuda:1','43':'cuda:0'}:
        raise ValueError('Campaign must contain the two fixed seeds on separate GPUs')
    manifest = verify(PRIVATE / 'prepared')
    if config['partition_sha256'] != manifest['partition_sha256']:
        raise ValueError('Prepared data/config mismatch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise RuntimeError('Commit the implementation before the definitive campaign')
    query = ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits']
    resource_snapshot = subprocess.check_output(query, text=True)
    free = {int(fields[0]): float(fields[2])-float(fields[3])
            for row in resource_snapshot.strip().splitlines() if (fields := row.split(','))}
    if any(free[index] < 2048 for index in (0,1)):
        raise RuntimeError('Insufficient GPU headroom; no external process will be modified')
    logs = PRIVATE / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    campaign_path = PRIVATE / 'campaign.json'
    if campaign_path.exists() or any((PRIVATE / 'runs' / f'seed-{seed}').exists() for seed in (42,43)):
        raise FileExistsError('Campaign/output already exists; do not overwrite or silently relaunch')
    campaign = {'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'config_sha256':canonical_hash(config), 'config_file_sha256':file_hash(CONFIG),
                'partition_sha256':manifest['partition_sha256'], 'gpu_preflight_csv':resource_snapshot,
                'maximum_training_processes':2, 'maximum_per_gpu':1, 'external_actions':'none', 'runs':[]}
    campaign['scheduler_sha256'] = file_hash(Path(__file__))
    started = time.perf_counter()
    workers = []
    for seed in (42,43):
        device = config['per_seed_device'][str(seed)]
        command = [PYTHON, 'research/feature_shift_digits/run_digits.py', '--config',
                   'research/feature_shift_digits/config.json', '--partition', '_local/feature_shift_digits/prepared',
                   '--output', f'_local/feature_shift_digits/runs/seed-{seed}', '--seed', str(seed), '--device', device]
        log = (logs/f'seed-{seed}.log').open('x')
        worker_started = time.perf_counter()
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        record = {'seed':seed,'device':device,'command':command,'pid':process.pid,
                  'started_utc':datetime.now(timezone.utc).isoformat(), 'exit_code':None}
        campaign['runs'].append(record)
        workers.append((process, log, record, worker_started))
        atomic_json(campaign_path,campaign)
        print(json.dumps({'started_seed':seed,'device':device,'pid':process.pid}),flush=True)
    while any(record['exit_code'] is None for _,_,record,_ in workers):
        for process,log,record,worker_started in workers:
            if record['exit_code'] is None and process.poll() is not None:
                record['exit_code'] = process.returncode
                record['ended_utc'] = datetime.now(timezone.utc).isoformat()
                record['process_wall_seconds'] = time.perf_counter()-worker_started
                log.close()
                atomic_json(campaign_path,campaign)
                print(json.dumps({'finished_seed':record['seed'],'exit_code':process.returncode}),flush=True)
        time.sleep(1)
    campaign['elapsed_wall_seconds'] = time.perf_counter()-started
    campaign['process_wall_seconds_sum'] = sum(record['process_wall_seconds'] for record in campaign['runs'])
    campaign['ended_utc'] = datetime.now(timezone.utc).isoformat()
    campaign['status'] = 'completed' if all(record['exit_code']==0 for record in campaign['runs']) else 'failed'
    atomic_json(campaign_path,campaign)
    print(json.dumps({'status':campaign['status'],'elapsed_wall_seconds':campaign['elapsed_wall_seconds']}),flush=True)
    return 0 if campaign['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
