"""Five-seed paired statistics and honest compute reduction of warm-up ablation."""
import json
from pathlib import Path
import statistics

from research.digits_mechanism_diagnostics.phase2.archive import METHODS, SEEDS, DOMAINS, summarize
from research.capacity_compute_control.audit_results import cost, epoch_batches


def test_summary_includes_all_seed_values_and_paired_sample_sd():
    records = {}
    for number, method in enumerate(METHODS):
        records[method] = {}
        for seed in SEEDS:
            records[method][seed] = {
                'identity': {'seed': seed}, 'evaluations': [{
                    'uniform_domain_accuracy_percent': seed - number,
                    'sample_weighted_accuracy_percent': seed * 2 - number,
                    'domains': {d: {'accuracy_percent': seed + index - number} for index, d in enumerate(DOMAINS)},
                }], 'total_counted_training_flops': 100, 'total_dense_training_flops': 90,
                'total_session_wall_seconds': 1, 'peak_cuda_allocated_mib': 1,
                'peak_cuda_reserved_mib': 1, 'peak_rss_mib': 1,
            }
    summary = summarize(records)
    for number, method in enumerate(METHODS):
        values = summary['methods'][method]['uniform_domain_accuracy_percent']
        assert values['values'] == [seed - number for seed in SEEDS]
        assert values['sample_sd_ddof1'] == statistics.stdev(SEEDS)
    assert summary['paired_full_minus_variant_pp']['decoder-only']['uniform_domain_accuracy_percent']['values'] == [3] * 5
    assert summary['methods']['shared-encoder']['domains']['SVHN']['values'] == [41, 42, 43, 44, 45]


def test_no_warmup_cost_counts_only_classification_not_zero_fake_phase():
    profile = json.loads(Path('research/capacity_compute_control/flop_profile.json').read_text())
    warm = sum(cost(profile, 'warmup', n) for n in epoch_batches(743))
    classification = sum(cost(profile, 'classification', n) for n in epoch_batches(743))
    assert classification * 1500 == 449968148424000
    assert (warm + classification) * 1500 == 551898005448000
    assert 0.18 < warm / (warm + classification) < 0.19
