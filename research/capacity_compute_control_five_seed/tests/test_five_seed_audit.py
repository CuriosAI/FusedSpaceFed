"""The five paired seeds, sample SD and domain results must all survive aggregation."""
import statistics

import pytest

from research.capacity_compute_control_five_seed.audit import DOMAINS, METHODS, SEEDS, summarize


def records():
    return {method: {seed: {'evaluations': [{
        'uniform_domain_accuracy_percent': seed + method_index,
        'sample_weighted_accuracy_percent': seed * 2 + method_index,
        'domains': {domain: {'accuracy_percent': seed + domain_index + method_index}
                    for domain_index, domain in enumerate(DOMAINS)},
    }]} for seed in SEEDS} for method_index, method in enumerate(METHODS)}


def test_all_five_values_and_sample_sd_not_population_sd():
    summary = summarize(records(), {})
    assert summary['seeds'] == [42, 43, 44, 45, 46]
    metric = summary['methods']['FedAvg']['uniform_domain_accuracy_percent']
    assert metric['values'] == list(SEEDS)
    assert metric['mean'] == 44
    assert metric['sample_sd_ddof1'] == statistics.stdev(SEEDS)
    assert metric['sample_sd_ddof1'] != statistics.pstdev(SEEDS)
    assert summary['methods']['FedAvg']['domains']['SVHN']['values'] == [43, 44, 45, 46, 47]
    assert summary['methods']['FedAvg']['sample_weighted_accuracy_percent']['values'] == [84, 86, 88, 90, 92]


def test_paired_differences_preserve_seeds_and_fail_on_missing_seed():
    data = records()
    summary = summarize(data, {})
    difference = summary['paired_fused_minus_fedavg']['uniform_domain_accuracy_percent']
    assert difference == {'values': [1] * 5, 'mean': 1, 'sample_sd_ddof1': 0}
    assert summary['paired_fused_minus_fedavg']['domains']['USPS']['values'] == [1] * 5
    del data['FedAvg'][46]
    with pytest.raises(KeyError):
        summarize(data, {})
