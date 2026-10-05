"""Register a bounded validation-only search before any new training/test."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.feature_shift_digits.data import atomic_json, canonical_hash

PUBLIC = Path('research/feature_shift_digits_calibrated')
PRIVATE = Path('_local/feature_shift_digits_calibrated')


def configuration(plan, candidate, phase, seed):
    return {'method': 'FusedSpaceFed', 'candidate_id': candidate['id'], 'phase': phase, 'seed': seed,
            'parent_partition_sha256': plan['parent_partition_sha256'],
            'plan_file': str(PUBLIC / 'search_plan.json'), 'plan_sha256': plan['plan_sha256'],
            'validation_file': plan['validation_file'], 'validation_sha256': plan['validation_sha256'],
            'profile_file': plan['profile_file'], 'profile_sha256': plan['profile_sha256'],
            'training': {**plan['fixed_training'], **candidate['training'], 'rounds': plan[phase]['rounds']}}


def write_queue(plan, phase, candidates, attachment=None):
    directory = PUBLIC / 'configs' / phase
    directory.mkdir(parents=True, exist_ok=True)
    queue = {'phase': phase, 'plan_sha256': plan['plan_sha256'], 'jobs': []}
    for candidate in candidates:
        for seed in plan[phase]['seeds']:
            name = f"{candidate['id']}-seed-{seed}"
            config_path = directory / (name + '.json')
            config = configuration(plan, candidate, phase, seed)
            if attachment:
                config.update(attachment)
            if config_path.exists():
                raise FileExistsError('Configuration already registered')
            atomic_json(config_path, config)
            queue['jobs'].append({'name': name, 'config': str(config_path), 'output': str(PRIVATE / phase / name)})
    output = PUBLIC / (phase + '_queue.json')
    if output.exists():
        raise FileExistsError('Queue already registered')
    atomic_json(output, queue)
    return queue


def prepare():
    if (PUBLIC / 'search_plan.json').exists():
        raise FileExistsError('Search plan already exists')
    original = json.loads(Path('research/feature_shift_digits/config.json').read_text())
    validation = json.loads(Path('research/capacity_compute_control/validation_split.json').read_text())
    profile = json.loads(Path('research/capacity_compute_control/flop_profile.json').read_text())
    candidates = [{'id': 'pilot-reference', 'training': {'classifier_lr': .01, 'autoencoder_lr': .0003, 'gradient_clip_norm': 1.}}]
    for a, lr in enumerate((.02, .05, .1)):
        for b, ae in enumerate((.0001, .0003, .001)):
            clip = (2., 5., 10.)[(a + b + 2) % 3]
            name = 'previous-selected-reference' if (lr, ae, clip) == (.02, .0003, 2.) else f'expanded-{a}-{b}'
            candidates.append({'id': name, 'training': {'classifier_lr': lr, 'autoencoder_lr': ae, 'gradient_clip_norm': clip}})
    plan = {'schema': 1, 'method': 'FusedSpaceFed', 'created_for': 'new five-seed Digits calibration request',
            'parent_partition_sha256': original['partition_sha256'],
            'validation_file': 'research/capacity_compute_control/validation_split.json',
            'validation_sha256': validation['validation_sha256'], 'validation_origin': 'training only, frozen before this campaign',
            'profile_file': 'research/capacity_compute_control/flop_profile.json', 'profile_sha256': profile['profile_sha256'],
            'full_samples_per_client': 743, 'fit_samples_per_client': 594, 'validation_samples_per_client': 149,
            'fixed_training': original['training'], 'candidates': candidates,
            'screening': {'rounds': 120, 'seeds': [142], 'selection': 'top two by final uniform-domain validation accuracy'},
            'confirmation': {'rounds': 300, 'seeds': [142, 143],
                             'selection': 'top two screening candidates plus both references (deduplicated), fresh starts; greatest mean of two final validation accuracies'},
            'tie_rule': 'lower classifier LR, then clip, then AE LR, then candidate ID',
            'mandatory_confirmation_references': ['pilot-reference', 'previous-selected-reference'],
            'final': {'rounds': 300, 'seeds': [42, 43, 44, 45, 46], 'initialization': 'fresh, full 743/domain',
                      'evaluation': 'one test at round300, no adaptation, no checkpoint/seed selection'},
            'budget': {'screening_runs': 10, 'maximum_confirmation_runs': 8, 'final_runs': 5,
                       'maximum_training_rounds': 5100, 'gpu_policy': 'one own worker per GPU, cuda0 shared by user authorization'},
            'rationale': 'Previous training-validation search at60 rounds favored upper classifier LR/clip boundaries; expand these and AE LR, confirm at the final300 horizon.',
            'limitations': ['Previous pilot and capacity-control tests have already been reported; this is a retrospective follow-up.',
                           'No previous or new test score enters candidate ranking, confirmation, tie rules, or freezing.',
                           'Screening uses an L9 design, not a full factorial; interactions may be missed.',
                           'Validation re-use and a small745-example validation pool can overfit; no claims of globally optimal hyperparameters.',
                           'All data, architecture, warmup, local epochs, optimizer types/resets, seeds42-46 and final evaluation are fixed.']}
    plan['plan_sha256'] = canonical_hash(plan)
    atomic_json(PUBLIC / 'search_plan.json', plan)
    write_queue(plan, 'screening', candidates)
    print(json.dumps({'plan_sha256': plan['plan_sha256'], 'screening_runs': len(candidates)}, indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
