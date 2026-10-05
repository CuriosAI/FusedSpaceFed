"""Synthetic provenance checks; never execute Git, models or dataset loaders."""
import ast
import copy
import hashlib
import json

import pytest

import calibrate_femnist_reconstructed as calibration
from femnist_reconstructed_data import canonical_hash, file_hash


CODE_FILES = (
    'fusedspacefed_core.py', 'femnist_reconstructed_data.py', 'train_femnist_reconstructed.py',
    'femnist_calibration_data.py', 'calibrate_femnist_reconstructed.py',
)
MODULE = 'calibrate_femnist_reconstructed.py'
FROZEN_SOURCE = b'''"""Frozen synthetic module."""
SCIENCE_SETTING = 1

def validate_selected_config(config):
    return 0

def verify_definitive_result(result, config, seed, code_hashes, test_counts):
    return 0

def training_update():
    return SCIENCE_SETTING
'''


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def _protected_digest(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {'validate_selected_config', 'verify_definitive_result'})]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


@pytest.fixture
def transition_context(tmp_path, monkeypatch):
    config = json.loads(calibration.REFERENCE.read_text())
    root = tmp_path / 'repo'
    artifacts = root / 'artifacts' / 'femnist_calibration'
    artifacts.mkdir(parents=True)
    reference = root / 'reference.json'
    selected_path = artifacts / 'selected_config.json'
    receipt_path = artifacts / 'selection.json'
    campaign_path = artifacts / 'completed_campaign.json'
    transition_path = artifacts / 'verification_transition.json'
    selected = copy.deepcopy(config)
    selected['training']['classifier_lr'] = 0.005
    _write(reference, config)
    _write(selected_path, selected)
    current_source = FROZEN_SOURCE.replace(b'return 0', b'return 1')
    (root / MODULE).write_bytes(current_source)
    from_code = {name: hashlib.sha256(name.encode()).hexdigest() for name in CODE_FILES}
    from_code[MODULE] = hashlib.sha256(FROZEN_SOURCE).hexdigest()
    to_code = copy.deepcopy(from_code)
    to_code[MODULE] = hashlib.sha256(current_source).hexdigest()
    frozen_git = {'commit': '1' * 40, 'working_tree_clean': True, 'code_sha256': from_code}
    receipt = {'status': 'frozen', 'selection_data': 'training-derived validation only',
               'selected_config_sha256': canonical_hash(selected), 'partition_sha256': config['partition_sha256'],
               'plan_sha256': '2' * 64, 'view_sha256': '3' * 64, 'calibration_code': frozen_git}
    _write(receipt_path, receipt)
    campaign = {'status': 'completed', 'selection_sha256': file_hash(receipt_path),
                'plan_sha256': receipt['plan_sha256'], 'view_sha256': receipt['view_sha256'],
                'git': copy.deepcopy(frozen_git)}
    _write(campaign_path, campaign)
    transition = {'schema': 1, 'kind': 'post_calibration_verification_only', 'status': 'approved',
                  'from_commit': frozen_git['commit'], 'from_code_sha256': copy.deepcopy(from_code),
                  'to_code_sha256': copy.deepcopy(to_code), 'selection_receipt_sha256': file_hash(receipt_path),
                  'selected_config_file_sha256': file_hash(selected_path),
                  'selected_config_sha256': canonical_hash(selected),
                  'partition_sha256': receipt['partition_sha256'],
                  'plan_sha256': receipt['plan_sha256'], 'view_sha256': receipt['view_sha256'],
                  'unchanged_module_ast_sha256': _protected_digest(FROZEN_SOURCE),
                  'calibration_campaign': {'path': campaign_path.relative_to(root).as_posix(),
                                           'sha256': file_hash(campaign_path)}}
    _write(transition_path, transition)
    context = dict(root=root, selected=selected, selected_path=selected_path, receipt=receipt,
                   receipt_path=receipt_path, campaign=campaign, campaign_path=campaign_path,
                   transition=transition, transition_path=transition_path, from_code=from_code,
                   to_code=to_code, original_source=FROZEN_SOURCE, current_source=current_source,
                   git_calls=[])
    monkeypatch.setattr(calibration, 'REPO', root)
    monkeypatch.setattr(calibration, 'REFERENCE', reference)
    monkeypatch.setattr(calibration, 'SELECTED', selected_path)
    monkeypatch.setattr(calibration, 'FROZEN_RECEIPT', receipt_path)
    monkeypatch.setattr(calibration, 'git_metadata', lambda: {'commit': '4' * 40,
                      'working_tree_clean': True, 'code_sha256': context['to_code']})

    def frozen_source(command, **kwargs):
        assert command == ['git', 'show', f"{frozen_git['commit']}:{MODULE}"]
        assert kwargs == {'cwd': root}
        context['git_calls'].append(command)
        return context['original_source']

    monkeypatch.setattr(calibration.subprocess, 'check_output', frozen_source)
    return context


def _check(context):
    calibration.validate_selected_config(context['selected'])


def _save_transition(context):
    _write(context['transition_path'], context['transition'])


def _rewrite_current(context, source):
    context['current_source'] = source
    (context['root'] / MODULE).write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    context['to_code'][MODULE] = digest
    context['transition']['to_code_sha256'][MODULE] = digest
    _save_transition(context)


def _save_campaign(context):
    _write(context['campaign_path'], context['campaign'])
    context['transition']['calibration_campaign']['sha256'] = file_hash(context['campaign_path'])
    _save_transition(context)


def test_explicit_verification_only_transition_is_accepted(transition_context):
    _check(transition_context)
    assert len(transition_context['git_calls']) == 1


def test_unchanged_code_requires_no_transition(transition_context):
    context = transition_context
    context['transition_path'].unlink()
    context['to_code'] = copy.deepcopy(context['from_code'])
    _check(context)
    assert context['git_calls'] == []


def test_changed_code_requires_explicit_transition(transition_context):
    transition_context['transition_path'].unlink()
    with pytest.raises(ValueError, match='explicit post-calibration'):
        _check(transition_context)


@pytest.mark.parametrize('field', ['schema', 'kind', 'status', 'extra'])
def test_transition_schema_is_closed(transition_context, field):
    context = transition_context
    context['transition'][field] = True if field == 'schema' else 'different'
    _save_transition(context)
    with pytest.raises(ValueError, match='transition schema'):
        _check(context)


@pytest.mark.parametrize('side', ['from_code_sha256', 'to_code_sha256'])
def test_transition_must_enumerate_all_five_files(transition_context, side):
    context = transition_context
    context['transition'][side].pop('fusedspacefed_core.py')
    _save_transition(context)
    with pytest.raises(ValueError, match='all five'):
        _check(context)


@pytest.mark.parametrize('value', ['g' * 64, 'a' * 63, True])
def test_transition_hashes_must_be_exact_sha256(transition_context, value):
    context = transition_context
    context['transition']['to_code_sha256'][MODULE] = value
    _save_transition(context)
    with pytest.raises(ValueError, match='all five'):
        _check(context)


@pytest.mark.parametrize('side', ['from_code_sha256', 'to_code_sha256'])
def test_transition_from_to_bind_actual_metadata(transition_context, side):
    context = transition_context
    context['transition'][side][MODULE] = 'e' * 64
    _save_transition(context)
    with pytest.raises(ValueError, match='from/to hashes'):
        _check(context)


@pytest.mark.parametrize('filename', [name for name in CODE_FILES if name != MODULE])
def test_other_four_science_data_files_cannot_change(transition_context, filename):
    context = transition_context
    context['to_code'][filename] = 'f' * 64
    context['transition']['to_code_sha256'][filename] = 'f' * 64
    _save_transition(context)
    with pytest.raises(ValueError, match='may change only'):
        _check(context)


@pytest.mark.parametrize('field', ['selection_receipt_sha256', 'selected_config_file_sha256',
                                   'selected_config_sha256', 'partition_sha256', 'plan_sha256', 'view_sha256'])
def test_frozen_selection_configuration_data_are_bound(transition_context, field):
    context = transition_context
    context['transition'][field] = 'e' * 64
    _save_transition(context)
    with pytest.raises(ValueError, match='cannot change the frozen'):
        _check(context)


def test_receipt_bytes_cannot_be_rewritten(transition_context):
    context = transition_context
    context['receipt_path'].write_text(context['receipt_path'].read_text() + ' ')
    with pytest.raises(ValueError, match='cannot change the frozen'):
        _check(context)


def test_selected_configuration_bytes_cannot_be_rewritten(transition_context):
    context = transition_context
    context['selected_path'].write_text(context['selected_path'].read_text() + ' ')
    with pytest.raises(ValueError, match='cannot change the frozen'):
        _check(context)


def test_selected_tunable_cannot_be_changed(transition_context):
    context = transition_context
    context['selected']['training']['classifier_lr'] *= 2
    _write(context['selected_path'], context['selected'])
    with pytest.raises(ValueError, match='frozen selection'):
        _check(context)


@pytest.mark.parametrize('change', ['status', 'selection_sha256', 'plan_sha256', 'view_sha256', 'git'])
def test_calibration_must_already_be_completed_and_unchanged(transition_context, change):
    context = transition_context
    if change == 'git':
        context['campaign']['git']['commit'] = '5' * 40
    else:
        context['campaign'][change] = 'calibrating' if change == 'status' else 'e' * 64
    _save_campaign(context)
    with pytest.raises(ValueError, match='unchanged completed calibration'):
        _check(context)


def test_completed_campaign_bytes_must_match_artifact(transition_context):
    context = transition_context
    context['campaign_path'].write_text(context['campaign_path'].read_text() + ' ')
    with pytest.raises(ValueError, match='campaign checksum'):
        _check(context)


@pytest.mark.parametrize('path', ['../outside.json', '/tmp/campaign.json', ''])
def test_campaign_path_cannot_escape_repository(transition_context, path):
    context = transition_context
    context['transition']['calibration_campaign']['path'] = path
    _save_transition(context)
    with pytest.raises(ValueError, match='campaign path'):
        _check(context)


def test_from_commit_is_frozen(transition_context):
    context = transition_context
    context['transition']['from_commit'] = '5' * 40
    _save_transition(context)
    with pytest.raises(ValueError, match='commit differs'):
        _check(context)


def test_git_show_original_raw_hash_must_match(transition_context):
    context = transition_context
    context['original_source'] += b'\n# modified original\n'
    with pytest.raises(ValueError, match='source hashes'):
        _check(context)


def test_current_raw_hash_must_match(transition_context):
    context = transition_context
    (context['root'] / MODULE).write_bytes(context['current_source'] + b'\n# drift\n')
    with pytest.raises(ValueError, match='source hashes'):
        _check(context)


@pytest.mark.parametrize('suffix', [b'\nNEW_SETTING = 2\n', b'\nimport os\n',
                                    b'\ndef new_helper():\n    return 1\n'])
def test_no_top_level_additions_are_permitted(transition_context, suffix):
    context = transition_context
    _rewrite_current(context, context['current_source'] + suffix)
    with pytest.raises(ValueError, match='protected training/selection'):
        _check(context)


def test_changed_scientific_body_is_rejected_even_with_matching_target_hash(transition_context):
    context = transition_context
    _rewrite_current(context, context['current_source'].replace(
        b'return SCIENCE_SETTING', b'return SCIENCE_SETTING + 1'))
    with pytest.raises(ValueError, match='protected training/selection'):
        _check(context)


def test_declared_protected_ast_digest_must_match(transition_context):
    context = transition_context
    context['transition']['unchanged_module_ast_sha256'] = 'e' * 64
    _save_transition(context)
    with pytest.raises(ValueError, match='protected training/selection'):
        _check(context)


@pytest.mark.parametrize('change', ['duplicate', 'missing'])
def test_exactly_two_named_verification_definitions_required(transition_context, change):
    context = transition_context
    source = context['current_source']
    if change == 'duplicate':
        source += b'\ndef verify_definitive_result():\n    return None\n'
    else:
        source = source.replace(b'def verify_definitive_result', b'def different_verifier')
    _rewrite_current(context, source)
    with pytest.raises(ValueError, match='exactly the two'):
        _check(context)
