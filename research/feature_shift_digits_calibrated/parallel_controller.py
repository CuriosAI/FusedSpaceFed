"""User-authorized GPU slot scheduler; adopt live workers without restarting them."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.feature_shift_digits.data import atomic_json, file_hash
from research.feature_shift_digits_calibrated.controller import PYTHON


def utc():
    return datetime.now(timezone.utc).isoformat()


def proc_stat(pid):
    # Linux fields 3, 4, 22, 52: state, PPID, start ticks, encoded wait status.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'state': fields[0], 'ppid': int(fields[1]), 'start_ticks': int(fields[19]), 'wait_status': int(fields[49])}


def adopted_exit(pid, parent, start_ticks):
    value = proc_stat(pid)
    if value['ppid'] != parent or value['start_ticks'] != start_ticks:
        raise RuntimeError('Adopted process identity changed')
    return os.waitstatus_to_exitcode(value['wait_status']) if value['state'] == 'Z' else None


def next_device(runs, slots):
    counts = {device: sum(row['device'] == device and row['exit_code'] is None for row in runs) for device in slots}
    eligible = [device for device in ('cuda:1', 'cuda:0') if counts[device] < slots[device]]
    return min(eligible, key=lambda device: counts[device]) if eligible else None


def launch(queue_file, receipt, logs, slots, scheduler_pid=None):
    queue = json.loads(queue_file.read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise RuntimeError('Commit implementation/configs before launch')
    if scheduler_pid is None:
        if receipt.exists():
            raise FileExistsError('Receipt exists; use explicit adoption of the stopped scheduler')
        state = {'status': 'running', 'queue_sha256': file_hash(queue_file), 'code_commit': commit,
                 'started_utc': utc(), 'runs': [], 'external_actions': 'none'}
    else:
        if proc_stat(scheduler_pid)['state'] != 'T':
            raise RuntimeError('Only adopt an explicitly stopped coordinator')
        command = Path(f'/proc/{scheduler_pid}/cmdline').read_bytes().split(b'\0')
        if b'research/feature_shift_digits_calibrated/controller.py' not in command or str(queue_file).encode() not in command:
            raise RuntimeError('Wrong scheduler; refuse process signals')
        raw = receipt.read_bytes()
        state = json.loads(raw)
        if state['status'] != 'running' or state['queue_sha256'] != file_hash(queue_file):
            raise RuntimeError('Wrong campaign for adoption')
        backup = receipt.with_name(receipt.stem + '_initial_scheduler.json')
        with backup.open('xb') as handle:
            handle.write(raw)
        for row in state['runs']:
            row['code_commit'] = state['code_commit']
            if row['exit_code'] is None:
                value = proc_stat(row['pid'])
                if value['ppid'] != scheduler_pid:
                    raise RuntimeError('Worker does not belong to the stopped coordinator')
                row['adopted_start_ticks'] = value['start_ticks']
                row['duration_source'] = 'UTC interval across coordinator handoff'
        state['handoff'] = {'scheduler_pid': scheduler_pid, 'adopted_utc': utc(), 'snapshot_file': str(backup),
                            'snapshot_sha256': file_hash(backup), 'training_signals': 'none',
                            'retirement': 'SIGTERM + SIGCONT to this stopped coordinator only, after adopted workers exit'}
        state['initial_scheduler_commit'] = state['code_commit']
        state['code_commit'] = commit
    registered = {row['name'] for row in state['runs']}
    pending = [job for job in queue['jobs'] if job['name'] not in registered]
    for job in pending:
        if (REPO / job['output']).exists():
            raise FileExistsError('Pending output already exists')
    state.update(max_processes=sum(slots.values()), max_per_gpu=max(slots.values()), gpu_slots=slots, controller_started_utc=utc())
    logs.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, state)
    workers = {}
    retired = scheduler_pid is None
    while pending or any(row['exit_code'] is None for row in state['runs']):
        for row in state['runs']:
            if row['exit_code'] is not None:
                continue
            if row['name'] in workers:
                process, log, began = workers[row['name']]
                code = process.poll()
            else:
                code = adopted_exit(row['pid'], scheduler_pid, row['adopted_start_ticks'])
            if code is None:
                continue
            ended = utc()
            if row['name'] in workers:
                seconds = time.perf_counter() - began
                log.close()
            else:
                seconds = (datetime.fromisoformat(ended) - datetime.fromisoformat(row['started_utc'])).total_seconds()
            row.update(exit_code=code, ended_utc=ended, process_wall_seconds=seconds)
            atomic_json(receipt, state)
            print(json.dumps({'finished': row['name'], 'exit_code': code}), flush=True)
            if code:
                state['status'] = 'failed'
                state['pending_not_started'] = [job['name'] for job in pending]
                pending = []
        if not retired and all(row['exit_code'] is not None for row in state['runs'] if 'adopted_start_ticks' in row):
            # All training children of the old coordinator have exited; new children
            # use separate sessions. Never signal a training process/group.
            if proc_stat(scheduler_pid)['state'] != 'T':
                raise RuntimeError('Old coordinator unexpectedly resumed')
            os.kill(scheduler_pid, signal.SIGTERM)
            os.kill(scheduler_pid, signal.SIGCONT)
            retired = True
            state['handoff']['retired_utc'] = utc()
            atomic_json(receipt, state)
        while pending and (device := next_device(state['runs'], slots)) is not None:
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
            free = {int(cols[0]): float(cols[1]) - float(cols[2]) for line in usage.strip().splitlines() if (cols := line.split(','))}
            if free[int(device[-1])] < 2048:
                break
            job = pending.pop(0)
            command = [PYTHON, 'research/feature_shift_digits_calibrated/runner.py', '--config', job['config'],
                       '--partition', '_local/feature_shift_digits/prepared', '--output', job['output'], '--device', device]
            log = (logs / (job['name'] + '.log')).open('x')
            began = time.perf_counter()
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row = {**job, 'command': command, 'pid': process.pid, 'device': device, 'started_utc': utc(), 'exit_code': None,
                   'code_commit': commit, 'duration_source': 'monotonic process wall clock'}
            state['runs'].append(row)
            workers[row['name']] = process, log, began
            atomic_json(receipt, state)
            print(json.dumps({'started': row['name'], 'pid': process.pid, 'device': device}), flush=True)
        time.sleep(1)
    state.update(ended_utc=utc(), wall_seconds=(datetime.now(timezone.utc) - datetime.fromisoformat(state['started_utc'])).total_seconds(),
                 process_wall_seconds_sum=sum(row['process_wall_seconds'] for row in state['runs']))
    if state['status'] != 'failed':
        state['status'] = 'completed'
    atomic_json(receipt, state)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--gpu1-slots', type=int, required=True)
    parser.add_argument('--gpu0-slots', type=int, required=True)
    parser.add_argument('--adopt-stopped-scheduler', type=int)
    args = parser.parse_args()
    if not 1 <= args.gpu1_slots <= 3 or not 1 <= args.gpu0_slots <= 3:
        parser.error('Supported slot range1..3/GPU')
    raise SystemExit(launch(args.queue, args.receipt, args.logs,
                           {'cuda:1': args.gpu1_slots, 'cuda:0': args.gpu0_slots}, args.adopt_stopped_scheduler))
