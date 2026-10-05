"""Synthetic adapter identity tests, not scientific CFD evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_dual_fno import MODULE, manifest_fixture


def fixture(tmp_path, monkeypatch):
    path, payload = manifest_fixture(tmp_path, monkeypatch)
    payload.update(kind=MODULE.P018_SYSTEM_KIND, status=MODULE.P018_MANIFEST_STATUS,
                   **MODULE._experiment_contract(MODULE.P018_SYSTEM_KIND)['extra'])
    payload['training_semantics'] = {'learning_rate':MODULE.P018_LEARNING_RATE}
    payload['aerodynamic']['metadata_kind'] = MODULE.P018_AERO_KIND
    protocol = Path(__file__).parents[1]/'docs/FC_P018_TRAINING_PROTOCOL_20261005.json'
    (tmp_path/'training_protocol.json').write_bytes(protocol.read_bytes())
    path.write_text(json.dumps(payload))
    return path,payload


def test_exact_profile_and_actual_protocol(tmp_path,monkeypatch):
    path,payload=fixture(tmp_path,monkeypatch)
    identity=MODULE.validate_dual_fno_manifest(path)
    assert identity.aerodynamic.metadata_kind==MODULE.P018_AERO_KIND
    assert identity.payload['actual_learning_rate']==1.5625e-7


@pytest.mark.parametrize('key,value',[
    ('actual_learning_rate',1e-5),('training_protocol_sha256','0'*64),
    ('training_protocol_file','../training_protocol.json'),('training_experiment','FC-P015'),
    ('status',MODULE.P015_MANIFEST_STATUS),('optimizer_steps',1368),
])
def test_mixed_identity_rejected(tmp_path,monkeypatch,key,value):
    path,payload=fixture(tmp_path,monkeypatch);payload[key]=value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):MODULE.validate_dual_fno_manifest(path)


@pytest.mark.parametrize('missing',[False,True])
def test_protocol_bytes_required(tmp_path,monkeypatch,missing):
    path,_=fixture(tmp_path,monkeypatch);protocol=tmp_path/'training_protocol.json'
    if missing:protocol.unlink()
    else:protocol.write_text('{}')
    with pytest.raises(ValueError):MODULE.validate_dual_fno_manifest(path)


def test_semantics_rate_not_only_top_level(tmp_path,monkeypatch):
    path,payload=fixture(tmp_path,monkeypatch)
    payload['training_semantics']['learning_rate']=1e-5;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):MODULE.validate_dual_fno_manifest(path)


def test_runtime_rechecks_protocol(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch);identity=MODULE.validate_dual_fno_manifest(path)
    (tmp_path/'training_protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='protocol SHA'):
        MODULE.validate_dual_runtime_files(identity,config_path=path,normalization_path=path)


@pytest.mark.parametrize('rate',[1.5625e-7,1e-5])
def test_loaded_metadata_rate_bound(tmp_path,monkeypatch,rate):
    path,_=fixture(tmp_path,monkeypatch)
    import fluid_control.calibrated_checkpoint as calibrated
    monkeypatch.setattr(calibrated,'validate_calibrated_epoch_zero',lambda *a,**k:None)
    monkeypatch.setattr(MODULE,'validate_runtime_precision',lambda:MODULE.PRECISION_PROTOCOL)
    def loader(directory,*,models,metadata_dict,device):
        if directory.name=='flow':return 0
        metadata_dict.update(status=MODULE.P018_AERO_KIND,checkpoint_epoch=1,
            flow_parent_model_sha256=MODULE.FLOW_MODEL_SHA256,
            flow_parent_state_sha256=MODULE.FLOW_STATE_SHA256,
            aerodynamic_initial_model_sha256=MODULE.FLOW_MODEL_SHA256,
            aerodynamic_initial_state_sha256=MODULE.FLOW_STATE_SHA256,
            **MODULE._experiment_contract(MODULE.P018_SYSTEM_KIND)['extra'],
            selection_performed=False,validation_accessed=False,frozen_test_accessed=False,ppo_executed=False)
        metadata_dict['actual_learning_rate']=rate
        return 1
    def run():return MODULE.load_dual_fno(path,SimpleNamespace(model=SimpleNamespace(**MODULE.ARCHITECTURE)),
        'cpu',build_model=lambda _:torch.nn.Linear(1,1),load_checkpoint=loader)
    if rate==MODULE.P018_LEARNING_RATE:assert run()[1].payload['kind']==MODULE.P018_SYSTEM_KIND
    else:
        with pytest.raises(ValueError,match='metadata'):run()
