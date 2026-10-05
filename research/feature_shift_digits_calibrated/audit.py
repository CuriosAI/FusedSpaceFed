"""Reconstruct saved accuracies and costs without evaluating a model/dataset."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.feature_shift_digits.data import DOMAINS, canonical_hash, file_hash

TEST_COUNTS = {'MNIST': 14000, 'SVHN': 19858, 'USPS': 1860, 'SynthDigits': 97791, 'MNIST-M': 14000}


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite saved value')
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def check_result(result, config, *, expected_source=None):
    finite_tree(result)
    if result['configuration'] != config or result['identity']['config_sha256'] != canonical_hash(config):
        raise ValueError('Configuration identity differs')
    identity = result['identity']
    for key in ('seed', 'phase', 'method'):
        if identity[key] != config[key]:
            raise ValueError('Run identity differs')
    for key in ('validation_sha256', 'plan_sha256', 'profile_sha256'):
        if identity[key] != config[key]:
            raise ValueError('Frozen hash differs')
    if identity['partition_sha256'] != config['parent_partition_sha256']:
        raise ValueError('Partition identity differs')
    if expected_source is not None and identity['code']['source_sha256'] != expected_source:
        raise ValueError('Scientific source bytes differ')
    rounds = config['training']['rounds']
    if result['status'] != 'completed' or result['completed_rounds'] != rounds or len(result['history']) != rounds:
        raise ValueError('Training incomplete')
    if [row['round'] for row in result['history']] != list(range(1, rounds + 1)):
        raise ValueError('Round sequence differs')
    final = config['phase'] == 'final'
    count = 743 if final else 594
    steps = math.ceil(count / 32)
    for row in result['history']:
        if set(row['clients']) != set(DOMAINS):
            raise ValueError('Missing training domain')
        for metric in row['clients'].values():
            if any(metric[key] != expected for key, expected in (
                ('warmup_steps', steps), ('classification_steps', steps), ('encoder_steps', 2 * steps),
                ('decoder_steps', steps), ('classifier_steps', steps), ('warmup_samples', count), ('classification_samples', count))):
                raise ValueError('Training budget differs')
    if len(result['evaluations']) != 1:
        raise ValueError('Unexpected evaluation/checkpoint selection')
    evaluation = result['evaluations'][0]
    if evaluation['round'] != rounds or evaluation['split'] != ('test' if final else 'validation') or set(evaluation['domains']) != set(DOMAINS):
        raise ValueError('Wrong evaluation split/round/domain')
    accuracies = []
    for domain, row in evaluation['domains'].items():
        expected = TEST_COUNTS[domain] if final else 149
        matrix = row['confusion_matrix']
        if len(matrix) != 10 or any(len(line) != 10 for line in matrix) or any(type(n) is not int or n < 0 for line in matrix for n in line):
            raise ValueError('Invalid confusion matrix')
        if row['total'] != expected or sum(map(sum, matrix)) != expected or sum(matrix[i][i] for i in range(10)) != row['correct']:
            raise ValueError('Counts differ from confusion matrix')
        accuracy = 100 * row['correct'] / expected
        if not math.isclose(accuracy, row['accuracy_percent'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('Accuracy differs from counts')
        accuracies.append(accuracy)
    uniform = statistics.mean(accuracies)
    weighted = 100 * sum(row['correct'] for row in evaluation['domains'].values()) / sum(row['total'] for row in evaluation['domains'].values())
    if not math.isclose(uniform, evaluation['uniform_domain_accuracy_percent'], abs_tol=1e-10) or not math.isclose(weighted, evaluation['sample_weighted_accuracy_percent'], abs_tol=1e-10):
        raise ValueError('Aggregate accuracy differs from counts')
    for cost in ('counted', 'dense'):
        total = sum(metric[cost + '_training_flops'] for row in result['history'] for metric in row['clients'].values())
        if total != result['total_' + cost + '_training_flops']:
            raise ValueError('FLOPs differ from per-round costs')
    wall = sum(session['wall_seconds'] for session in result['sessions'])
    if not math.isclose(wall, result['total_session_wall_seconds'], abs_tol=1e-10):
        raise ValueError('Whole-run time differs from session sum')
    return evaluation


def rivals():
    with (REPO / 'research/feature_shift_digits/fedbn_table11.csv').open() as handle:
        return list(csv.DictReader(handle))


def load_result(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = load_result(args.run / 'results.json')
    value = check_result(result, result['configuration'])
    print(json.dumps({'seed': result['identity']['seed'], 'uniform_percent': value['uniform_domain_accuracy_percent'],
                      'per_domain': {domain: row['accuracy_percent'] for domain, row in value['domains'].items()},
                      'process_session_seconds': result['total_session_wall_seconds'],
                      'peak_cuda_allocated_mib': result['peak_cuda_allocated_mib']}, indent=2))
