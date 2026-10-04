"""Audit saved counts only; no dataset or training involved."""
import copy
import json
from pathlib import Path

import pytest

from calibrate_femnist_reconstructed import PRIMARY, SECONDARY, verify_definitive_result
from femnist_reconstructed_data import canonical_hash
from train_femnist_reconstructed import accuracy_metrics


@pytest.fixture
def complete_record():
    config = json.loads(Path('configs/femnist_reconstructed.json').read_text())
    totals = {f'f_{i:05d}': 18 if i < 53 else 17 for i in range(150)}
    counts = {name: {'correct': total // 2, 'total': total, 'participations': 20}
              for name, total in totals.items()}
    metric = {'split': 'test', **accuracy_metrics(counts)}
    result = {'identity': {'seed': 41, 'mode': 'definitive', 'config_sha256': canonical_hash(config),
                          'partition_sha256': config['partition_sha256'], 'code_sha256': {}},
              'status': 'completed', 'completed_round': 200,
              'participations': {name: 20 for name in counts},
              'history': [{'round': r, 'evaluation': copy.deepcopy(metric) if r >= 191 else None}
                          for r in range(1, 201)],
              'summary': {name: metric[name] for name in (PRIMARY, SECONDARY)}}
    return result, config, totals


def test_exact_window_all_clients_and_fixed_means(complete_record):
    result, config, totals = complete_record
    assert verify_definitive_result(result, config, 41, {}, totals) == pytest.approx(result['summary'])


@pytest.mark.parametrize('value', [True, 8.0, 8.5])
def test_saved_counts_must_be_integers(complete_record, value):
    result, config, totals = complete_record
    result['history'][190]['evaluation']['clients']['f_00000']['correct'] = value
    with pytest.raises(ValueError, match='integers'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_global_total_alone_cannot_hide_redistributed_client_tests(complete_record):
    result, config, totals = complete_record
    counts = result['history'][190]['evaluation']['clients']
    counts['f_00000']['total'] += 1
    counts['f_00001']['total'] -= 1
    assert sum(c['total'] for c in counts.values()) == 2603
    with pytest.raises(ValueError, match='Invalid definitive counts'):
        verify_definitive_result(result, config, 41, {}, totals)


@pytest.mark.parametrize('where', ['metric', 'summary'])
def test_nonfinite_saved_values_fail(complete_record, where):
    result, config, totals = complete_record
    target = result['summary'] if where == 'summary' else result['history'][190]['evaluation']
    target[PRIMARY] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_test_evaluation_outside_agreed_window_fails(complete_record):
    result, config, totals = complete_record
    result['history'][189]['evaluation'] = copy.deepcopy(result['history'][190]['evaluation'])
    with pytest.raises(ValueError, match='restricted'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_changed_summary_is_not_a_best_round_or_seed(complete_record):
    result, config, totals = complete_record
    result['summary'][PRIMARY] += 1
    with pytest.raises(ValueError, match='fixed ten-round'):
        verify_definitive_result(result, config, 41, {}, totals)
