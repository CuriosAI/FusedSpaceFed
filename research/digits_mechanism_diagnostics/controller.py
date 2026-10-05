"""Independent GPU workers with receipts, bounded slots, and no process signals."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.feature_shift_digits.data import atomic_json, file_hash

PYTHON = '/home/schroeder/miniconda3/envs/general_ml/bin/python'


def utc():
    return datetime.now(timezone.utc).isoformat()


def launch(queue_path, receipt_path, logs):
    queue = json.loads(queue_path.read_text())
    if receipt_path.exists():
        raise FileExistsError('Receipt exists; no implicit restart or overwrite')
    jobs = queue['jobs'].copy()
    for job in jobs:
        if (REPO / job['output']).exists():
            raise FileExistsError('Registered output already exists')
    for name, digest in queue['frozen_files'].items():
        if file_hash(REPO / name) != digest:
            raise ValueError('Frozen campaign file changed: ' + name)
    logs.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    state = {'status': 'running', 'queue_sha256': file_hash(queue_path), 'frozen_files': queue['frozen_files'],
             'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
             'started_utc': utc(), 'gpu_slots': queue['gpu_slots'], 'external_actions': 'none', 'runs': [],
             'source_note': 'New phase sources identified by SHA256, committed with completed per-phase results'}
    atomic_json(receipt_path, state)
    workers = []
    while jobs or any(r['exit_code'] is None for _, _, r, _ in workers):
        for process, log, row, started in workers:
            if row['exit_code'] is None and process.poll() is not None:
                row.update(exit_code=process.returncode, ended_utc=utc(), process_wall_seconds=time.perf_counter() - started)
                log.close(); atomic_json(receipt_path, state)
                print(json.dumps({'finished': row['name'], 'exit_code': process.returncode}), flush=True)
                if process.returncode:
                    state['status'] = 'failed'; state['pending_not_started'] = [j['name'] for j in jobs]; jobs = []
        for device in ('cuda:1', 'cuda:0'):
            busy = sum(r['device'] == device and r['exit_code'] is None for _, _, r, _ in workers)
            if not jobs or busy >= queue['gpu_slots'][device]:
                continue
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
            free = {int(cols[0]): float(cols[1]) - float(cols[2]) for line in usage.splitlines() if (cols := line.split(','))}
            if free[int(device[-1])] < queue.get('minimum_free_mib', 2048):
                continue
            job = jobs.pop(0)
            command = [PYTHON, job['script'], *job['args'], '--output', job['output'], '--device', device]
            log = (logs / (job['name'] + '.log')).open('x')
            started = time.perf_counter()
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            row = {**job, 'command': command, 'device': device, 'pid': process.pid, 'started_utc': utc(), 'exit_code': None}
            state['runs'].append(row); workers.append((process, log, row, started)); atomic_json(receipt_path, state)
            print(json.dumps({'started': job['name'], 'device': device, 'pid': process.pid}), flush=True)
        time.sleep(1)
    state.update(ended_utc=utc(), wall_seconds=time.perf_counter() - began,
                 process_wall_seconds_sum=sum(r['process_wall_seconds'] for _, _, r, _ in workers))
    if state['status'] != 'failed':
        state['status'] = 'completed'
    atomic_json(receipt_path, state)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.queue, args.receipt, args.logs))
