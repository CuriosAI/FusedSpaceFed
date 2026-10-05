"""Advance a pre-registered validation search and freeze before final tests."""
import argparse
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.feature_shift_digits.data import atomic_json, canonical_hash, file_hash
from research.feature_shift_digits_calibrated.audit import check_result
from research.feature_shift_digits_calibrated.prepare_plan import PUBLIC, PRIVATE, write_queue
from research.feature_shift_digits_calibrated.runner import SOURCES, validate_config, read_hashed


def tie_key(candidate):
    settings = candidate['training']
    return settings['classifier_lr'], settings['gradient_clip_norm'], settings['autoencoder_lr'], candidate['id']


def rank_candidates(candidates, scores):
    return sorted(candidates, key=lambda row: (-statistics.mean(scores[row['id']]), *tie_key(row)))


def collect(phase, plan):
    queue = json.loads((PUBLIC / (phase + '_queue.json')).read_text())
    receipt_path = PRIVATE / (phase + '_campaign.json')
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'completed' or len(receipt['runs']) != len(queue['jobs']) or any(row['exit_code'] != 0 for row in receipt['runs']):
        raise ValueError('All calibration workers must have exited successfully')
    if receipt['queue_sha256'] != file_hash(PUBLIC / (phase + '_queue.json')):
        raise ValueError('Controller queue identity differs')
    if {row['name'] for row in receipt['runs']} != {row['name'] for row in queue['jobs']}:
        raise ValueError('Controller jobs differ from registered queue')
    rows, grouped = [], {}
    seen = set()
    for job in queue['jobs']:
        path = Path(job['output']) / 'results.json'
        result = json.loads(path.read_text())
        config = json.loads(Path(job['config']).read_text())
        if config['phase'] != phase or config['seed'] not in plan[phase]['seeds'] or config['training']['rounds'] != plan[phase]['rounds']:
            raise ValueError('Wrong calibration phase/seed/horizon')
        validate_config(config)
        key = (config['candidate_id'], config['seed'])
        if key in seen:
            raise ValueError('Duplicate calibration seed')
        seen.add(key)
        evaluation = check_result(result, config, expected_source={name: file_hash(REPO / name) for name in SOURCES})
        score = evaluation['uniform_domain_accuracy_percent']
        grouped.setdefault(config['candidate_id'], []).append(score)
        rows.append({'candidate_id': config['candidate_id'], 'seed': config['seed'], 'uniform_validation_accuracy_percent': score,
                     'result_sha256': file_hash(path), 'configuration_sha256': canonical_hash(config), 'directory': job['output']})
    if any(len(scores) != len(plan[phase]['seeds']) for scores in grouped.values()):
        raise ValueError('Unpaired validation seeds')
    return grouped, rows, file_hash(receipt_path)


def advance(stage):
    filename = 'shortlist.json' if stage == 'confirmation' else 'selection.json'
    destination = PUBLIC / filename
    if destination.exists():
        raise FileExistsError('Already frozen; never reopen after test')
    plan = read_hashed(PUBLIC / 'search_plan.json', 'plan_sha256')
    grouped, rows, receipt_hash = collect('screening' if stage == 'confirmation' else 'confirmation', plan)
    if stage == 'confirmation':
        if set(grouped) != {row['id'] for row in plan['candidates']}:
            raise ValueError('Screening candidate set differs')
        ranked = rank_candidates(plan['candidates'], grouped)
        ids = sorted({row['id'] for row in ranked[:2]} | set(plan['mandatory_confirmation_references']))
        value = {'candidate_ids': ids, 'scores': rows, 'plan_sha256': plan['plan_sha256'],
                 'rule': plan['confirmation']['selection'], 'screening_campaign_sha256': receipt_hash,
                 'no_test_scores_used': True}
        value['shortlist_sha256'] = canonical_hash(value)
        attachment = {'shortlist_file': str(destination), 'shortlist_sha256': value['shortlist_sha256']}
    else:
        shortlist = json.loads((PUBLIC / 'shortlist.json').read_text())
        if set(grouped) != set(shortlist['candidate_ids']):
            raise ValueError('Confirmation candidate set differs')
        eligible = [row for row in plan['candidates'] if row['id'] in grouped]
        winner = rank_candidates(eligible, grouped)[0]
        value = {'selected_candidate_id': winner['id'], 'selected_training_settings': winner['training'],
                 'uniform_validation_accuracy_percent': statistics.mean(grouped[winner['id']]),
                 'scores': rows, 'plan_sha256': plan['plan_sha256'], 'shortlist_sha256': shortlist['shortlist_sha256'],
                 'rule': plan['confirmation']['selection'], 'confirmation_campaign_sha256': receipt_hash,
                 'no_test_scores_used': True, 'final_seeds': plan['final']['seeds'], 'initialization': 'fresh full-training runs'}
        value['selection_sha256'] = canonical_hash(value)
        ids = [winner['id']]
        attachment = {'selection_file': str(destination), 'selection_sha256': value['selection_sha256']}
    atomic_json(destination, value)
    write_queue(plan, stage, [row for row in plan['candidates'] if row['id'] in ids], attachment)
    print(json.dumps(value, indent=2))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('confirmation', 'final'), required=True)
    advance(parser.parse_args().stage)
