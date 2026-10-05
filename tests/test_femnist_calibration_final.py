"""Audit saved counts only; no dataset or training involved."""
import copy
import json
import random
from pathlib import Path

import pytest

from calibrate_femnist_reconstructed import PRIMARY, SECONDARY, verify_definitive_result
from femnist_reconstructed_data import canonical_hash
from train_femnist_reconstructed import accuracy_metrics


@pytest.fixture
def complete_record():
    config = json.loads(Path('configs/femnist_reconstructed.json').read_text())
    totals = {f'f_{i:05d}': 18 if i < 53 else 17 for i in range(150)}
    clients = sorted(totals)
    selection = random.Random(41)
    participations = {name: 0 for name in clients}
    history = []
    for round_number in range(1, 201):
        active = selection.sample(clients, 15)
        for name in active:
            participations[name] += 1
        metric = None
        if round_number >= 191:
            counts = {name: {'correct': total // 2 + round_number % 3 - 1,
                             'total': total, 'participations': participations[name]}
                      for name, total in totals.items()}
            metric = {'split': 'test', **accuracy_metrics(counts)}
        history.append({'round': round_number, 'active_clients': active,
                        'clients': [{'client': name, 'participations': participations[name]}
                                    for name in active],
                        'evaluation': metric})
    result = {'identity': {'seed': 41, 'mode': 'definitive', 'config_sha256': canonical_hash(config),
                          'partition_sha256': config['partition_sha256'], 'code_sha256': {}},
              'status': 'completed', 'completed_round': 200,
              'participations': participations,
              'history': history,
              'summary': {name: sum(row['evaluation'][name] for row in history[-10:]) / 10
                          for name in (PRIMARY, SECONDARY)}}
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


def test_evaluation_participations_are_cumulative_at_each_round(complete_record):
    result, config, totals = complete_record
    assert len(set(result['participations'].values())) > 1
    for row in result['history'][-10:]:
        assert sum(c['participations'] for c in row['evaluation']['clients'].values()) == 15 * row['round']
    first = result['history'][190]['evaluation']['clients']
    assert any(c['participations'] != result['participations'][name] for name, c in first.items())
    assert verify_definitive_result(result, config, 41, {}, totals) == pytest.approx(result['summary'])


@pytest.mark.parametrize('value', [True, 8.0, 8.5])
@pytest.mark.parametrize('field', ['total', 'participations'])
def test_saved_non_correct_counts_must_be_integers(complete_record, value, field):
    result, config, totals = complete_record
    result['history'][190]['evaluation']['clients']['f_00000'][field] = value
    with pytest.raises(ValueError, match='integers'):
        verify_definitive_result(result, config, 41, {}, totals)


@pytest.mark.parametrize('field', ['correct', 'total'])
def test_top_level_counts_must_be_integers(complete_record, field):
    result, config, totals = complete_record
    metric = result['history'][190]['evaluation']
    metric[field] = float(metric[field])
    with pytest.raises(ValueError, match='integers'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_final_snapshot_cannot_replace_earlier_participations(complete_record):
    result, config, totals = complete_record
    counts = result['history'][190]['evaluation']['clients']
    for name, count in counts.items():
        count['participations'] = result['participations'][name]
    with pytest.raises(ValueError, match='cumulative participation'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_evaluation_occurs_after_current_round_updates(complete_record):
    result, config, totals = complete_record
    row = result['history'][190]
    for name in row['active_clients']:
        row['evaluation']['clients'][name]['participations'] -= 1
    with pytest.raises(ValueError, match='cumulative participation'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_redistributed_snapshot_participations_fail_even_with_same_total(complete_record):
    result, config, totals = complete_record
    counts = result['history'][193]['evaluation']['clients']
    counts['f_00000']['participations'] += 1
    counts['f_00001']['participations'] -= 1
    assert sum(c['participations'] for c in counts.values()) == 194 * 15
    with pytest.raises(ValueError, match='cumulative participation'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_final_vector_must_match_history_even_with_3000_total(complete_record):
    result, config, totals = complete_record
    result['participations']['f_00000'] += 1
    result['participations']['f_00001'] -= 1
    assert sum(result['participations'].values()) == 3000
    with pytest.raises(ValueError, match='200-round active-client history'):
        verify_definitive_result(result, config, 41, {}, totals)


@pytest.mark.parametrize('size', [0, 14, 16])
def test_every_round_requires_exactly_fifteen_active_clients(complete_record, size):
    result, config, totals = complete_record
    result['history'][0]['active_clients'] = sorted(totals)[:size]
    with pytest.raises(ValueError, match='15 distinct'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_duplicate_active_client_is_rejected(complete_record):
    result, config, totals = complete_record
    active = result['history'][0]['active_clients']
    active[-1] = active[0]
    assert len(active) == 15
    with pytest.raises(ValueError, match='15 distinct'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_unknown_active_client_is_rejected(complete_record):
    result, config, totals = complete_record
    result['history'][0]['active_clients'][0] = 'unknown_client'
    with pytest.raises(ValueError, match='unknown active'):
        verify_definitive_result(result, config, 41, {}, totals)


@pytest.mark.parametrize('value', [None, {}, 'invalid', [True] * 15, [[]] * 15])
def test_missing_or_invalid_active_client_history_is_rejected(complete_record, value):
    result, config, totals = complete_record
    if value is None:
        result['history'][0].pop('active_clients')
    else:
        result['history'][0]['active_clients'] = value
    with pytest.raises(ValueError, match='15 distinct'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_different_early_selection_is_detected_from_cumulative_counts(complete_record):
    result, config, totals = complete_record
    active = result['history'][0]['active_clients']
    replacement = next(name for name in sorted(totals) if name not in active)
    active[0] = replacement
    with pytest.raises(ValueError, match='cumulative participation'):
        verify_definitive_result(result, config, 41, {}, totals)


@pytest.mark.parametrize('value', [True, 20.0, -1])
def test_invalid_final_participation_value_is_rejected(complete_record, value):
    result, config, totals = complete_record
    result['participations']['f_00000'] = value
    with pytest.raises(ValueError, match='Invalid definitive participation'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_missing_final_client_is_rejected(complete_record):
    result, config, totals = complete_record
    result['participations'].pop('f_00000')
    with pytest.raises(ValueError, match='Invalid definitive participation'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_missing_evaluation_client_is_rejected(complete_record):
    result, config, totals = complete_record
    result['history'][190]['evaluation']['clients'].pop('f_00000')
    with pytest.raises(ValueError, match='all 150 clients'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_validation_split_cannot_enter_final_evaluation(complete_record):
    result, config, totals = complete_record
    result['history'][190]['evaluation']['split'] = 'validation'
    with pytest.raises(ValueError, match='different evaluation split'):
        verify_definitive_result(result, config, 41, {}, totals)


def test_last_fixed_evaluation_cannot_be_missing(complete_record):
    result, config, totals = complete_record
    result['history'][-1]['evaluation'] = None
    with pytest.raises(ValueError, match='restricted'):
        verify_definitive_result(result, config, 41, {}, totals)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'out_of_order', 'bool_round'])
def test_exact_two_hundred_round_history_required(complete_record, change):
    result, config, totals = complete_record
    history = result['history']
    if change == 'missing':
        history.pop(0)
    elif change == 'duplicate':
        history[0]['round'] = 2
    elif change == 'out_of_order':
        history[0], history[1] = history[1], history[0]
    else:
        history[0]['round'] = True
    with pytest.raises(ValueError, match='round history'):
        verify_definitive_result(result, config, 41, {}, totals)


@pytest.mark.parametrize('change', ['missing_client', 'bool_total', 'wrong_global_total'])
def test_frozen_test_metadata_requires_original_integer_totals(complete_record, change):
    result, config, totals = complete_record
    if change == 'missing_client':
        totals.pop('f_00000')
    elif change == 'bool_total':
        totals['f_00000'] = True
    else:
        totals['f_00000'] += 1
    with pytest.raises(ValueError, match='original definitive client totals'):
        verify_definitive_result(result, config, 41, {}, totals)
