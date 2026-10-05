"""GPU slot limits and exact external-worker exit attribution, without GPUs."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from research.feature_shift_digits_calibrated import parallel_controller as controller


@pytest.mark.parametrize('slots,initial,expected', [
    ({'cuda:1': 3, 'cuda:0': 3}, ['cuda:1', 'cuda:0'], ['cuda:1', 'cuda:0', 'cuda:1', 'cuda:0']),
    ({'cuda:1': 3, 'cuda:0': 2}, [], ['cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1'])])
def test_exact_slot_distribution_and_adoption(slots, initial, expected):
    rows = [{'device': device, 'exit_code': None} for device in initial]
    for wanted in expected:
        device = controller.next_device(rows, slots)
        assert device == wanted
        rows.append({'device': device, 'exit_code': None})
    assert controller.next_device(rows, slots) is None
    rows[0]['exit_code'] = 0
    assert controller.next_device(rows, slots) == rows[0]['device']


def test_linux_exit_status_and_pid_reuse_protection(monkeypatch):
    value = {'state': 'R', 'ppid': 123, 'start_ticks': 456, 'wait_status': 0}
    monkeypatch.setattr(controller, 'proc_stat', lambda pid: value)
    assert controller.adopted_exit(789, 123, 456) is None
    value.update(state='Z', wait_status=7 << 8)
    assert controller.adopted_exit(789, 123, 456) == 7
    value['wait_status'] = 0
    assert controller.adopted_exit(789, 123, 456) == 0
    with pytest.raises(RuntimeError, match='identity'):
        controller.adopted_exit(789, 123, 457)
    with pytest.raises(RuntimeError, match='identity'):
        controller.adopted_exit(789, 124, 456)


def test_proc_stat_parses_documented_linux_fields(monkeypatch):
    fields = ['0'] * 50
    fields[0] = 'Z'
    fields[1] = '123'
    fields[19] = '456'
    fields[49] = str(7 << 8)
    monkeypatch.setattr(Path, 'read_text', lambda path: '789 (python worker) ' + ' '.join(fields))
    assert controller.proc_stat(789) == {'state': 'Z', 'ppid': 123, 'start_ticks': 456, 'wait_status': 7 << 8}


def test_direct_cli_imports():
    result = subprocess.run([sys.executable, '-B', 'research/feature_shift_digits_calibrated/parallel_controller.py', '--help'],
                            capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, timeout=30)
    assert result.returncode == 0, result.stderr
