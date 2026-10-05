"""Synthetic checks of the five-run authorization; no training/GPU access."""
import copy
import json
from pathlib import Path

import pytest

from research.feature_shift_digits import launch_extension as launcher
from research.feature_shift_digits.data import atomic_json, canonical_hash
from research.feature_shift_digits.run_registered_seed import scientific_configuration, validate_extension


def configurations():
    original=json.loads(Path('research/feature_shift_digits/config.json').read_text())
    extended=json.loads(Path('research/feature_shift_digits/config_five.json').read_text())
    return original,extended


def test_extension_keeps_every_scientific_field():
    original,extended=configurations()
    validate_extension(original,extended)
    assert scientific_configuration(original)==scientific_configuration(extended)
    changed={name for name in original if original[name]!=extended[name]}
    assert changed=={'run_seeds','per_seed_device'}


@pytest.mark.parametrize('field,value',[('classifier_lr',0.1),('warmup_epochs',0),('rounds',301)])
def test_scientific_changes_are_refused(field,value):
    original,extended=configurations()
    extended=copy.deepcopy(extended);extended['training'][field]=value
    with pytest.raises(ValueError,match='only change'):
        validate_extension(original,extended)


def test_extension_scheduler_one_worker_per_gpu_and_three_fresh_outputs(tmp_path,monkeypatch):
    public=tmp_path/'public';private=tmp_path/'private'
    public.mkdir();private.mkdir()
    original,extended=configurations()
    atomic_json(public/'config.json',original);atomic_json(public/'config_five.json',extended)
    atomic_json(public/'five_run_authorization.json',{'extended_config_sha256':canonical_hash(extended),
                'scientific_config_sha256':canonical_hash(scientific_configuration(extended))})
    atomic_json(private/'campaign.json',{'status':'completed','runs':[{'seed':42,'exit_code':0},{'seed':43,'exit_code':0}]})
    monkeypatch.setattr(launcher,'PUBLIC',public);monkeypatch.setattr(launcher,'PRIVATE',private);monkeypatch.setattr(launcher,'REPO',tmp_path)
    monkeypatch.setattr(launcher,'verify',lambda root:{'partition_sha256':extended['partition_sha256']})
    def query(command,**kwargs):
        return '0, 49000, 100\n1, 49000, 100\n' if command[0]=='nvidia-smi' else 'operational_commit'
    monkeypatch.setattr(launcher.subprocess,'check_output',query)
    active=set();actions=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.seed=int(command[command.index('--seed')+1]);self.device=command[command.index('--device')+1]
            assert self.device not in active
            active.add(self.device);assert len(active)<=2
            self.pid=self.seed;self.returncode=0;actions.append(('start',self.seed))
        def poll(self):
            active.remove(self.device);actions.append(('end',self.seed));return 0
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    monkeypatch.setattr(launcher.time,'sleep',lambda seconds:None)
    assert launcher.main()==0
    record=json.loads((private/'campaign_extension.json').read_text())
    assert record['status']=='completed'
    assert [item['seed'] for item in record['runs']]==[44,45,46]
    assert actions[:2]==[('start',44),('start',45)]
    assert actions.index(('end',44))<actions.index(('start',46))
    assert record['external_actions']=='none' and not active
    with pytest.raises(FileExistsError):
        launcher.main()
