"""Numerical archive checks independent of a fresh benchmark evaluation."""
import copy
import json
import math
from pathlib import Path

import pytest

from research.feature_shift_digits.data import canonical_hash, file_hash
from research.feature_shift_digits_calibrated.archive import stats, validate_receipt, write_new
from research.feature_shift_digits_calibrated.audit import check_result, rivals
from research.feature_shift_digits_calibrated.tests.test_campaign import synthetic_final


def test_sample_standard_deviation_and_all_seeds_are_preserved():
    result = stats([1., 2., 3., 4., 5.])
    assert result['values'] == [1., 2., 3., 4., 5.]
    assert result['mean'] == 3.
    assert result['sample_standard_deviation_ddof1'] == math.sqrt(2.5)
    assert stats([3.])['sample_standard_deviation_ddof1'] is None


def test_published_domain_mapping_keeps_methods_and_uncertainties_distinct():
    rows = rivals()
    assert len(rows) == 15
    means = {(row['method'], row['domain']): float(row['mean_accuracy_percent']) for row in rows}
    assert means['FedBN', 'SVHN'] == 76.93
    assert means['FedAvg', 'MNIST-M'] == 82.44
    assert means['FedProx', 'SynthDigits'] == 86.60
    assert all(row['result_origin'] == 'reported' and row['trials'] == '5' for row in rows)


def test_source_mismatch_confusion_corruption_and_duplicate_rounds_rejected():
    config, result = synthetic_final()
    result['identity']['code'] = {'source_sha256': {'model': 'correct'}}
    check_result(result, config, expected_source={'model': 'correct'})
    with pytest.raises(ValueError, match='source'):
        check_result(result, config, expected_source={'model': 'changed'})
    broken = copy.deepcopy(result)
    broken['history'][1]['round'] = 1
    with pytest.raises(ValueError, match='sequence'):
        check_result(broken, config)
    broken = copy.deepcopy(result)
    broken['evaluations'][0]['domains']['MNIST']['confusion_matrix'][0][0] -= 1
    with pytest.raises(ValueError, match='confusion'):
        check_result(broken, config)


def test_receipt_requires_success_registered_outputs_and_fresh_commands(tmp_path):
    queue = {'jobs': [{'name': 'a', 'config': 'config.json', 'output': 'output'}]}
    path = tmp_path / 'queue.json'
    path.write_text(json.dumps(queue))
    receipt = {'status': 'completed', 'queue_sha256': file_hash(path),
               'runs': [{'name': 'a', 'config': 'config.json', 'output': 'output', 'exit_code': 0, 'command': ['python', 'runner']}]}
    validate_receipt(queue, receipt, path)
    broken = copy.deepcopy(receipt)
    broken['runs'][0]['exit_code'] = 1
    with pytest.raises(ValueError, match='successfully'):
        validate_receipt(queue, broken, path)
    broken = copy.deepcopy(receipt)
    broken['runs'][0]['command'].append('--resume')
    with pytest.raises(ValueError, match='fresh'):
        validate_receipt(queue, broken, path)


def test_archival_writes_refuse_overwrites(tmp_path):
    path = tmp_path / 'nested' / 'result.gz'
    write_new(path, b'unchanged')
    with pytest.raises(FileExistsError):
        write_new(path, b'new')
    assert path.read_bytes() == b'unchanged'


def test_cli_archive_help():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-B', 'research/feature_shift_digits_calibrated/archive.py', '--help'],
                            capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, timeout=30)
    assert result.returncode == 0, result.stderr
