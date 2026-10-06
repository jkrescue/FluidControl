import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


M = module('h25_review_loader', ROOT/'src/fluid_control/dual_fno.py')
PRODUCER = Path(os.environ.get('P064_H25_PRODUCER', str(ROOT/'scripts/train_p064_b_flow_h25_bounded.py')))
P = module('h25_actual_producer', PRODUCER)


class FakeModel:
    def cpu(self):
        return self

    def to(self, device):
        return self


@pytest.fixture
def produced(tmp_path, monkeypatch):
    parent = tmp_path/'parent'; parent.mkdir()
    roles = {}
    for role, epoch, kind in [('flow', 0, M.FLOW_KIND), ('aerodynamic', 1, M.P064_AERO_KIND['B'])]:
        directory=parent/role; directory.mkdir()
        model=directory/f'FNO.0.{epoch}.mdlus'; model.write_bytes((role+'model').encode())
        state=directory/f'checkpoint.0.{epoch}.pt'; state.write_bytes((role+'state').encode())
        roles[role]=dict(role=role, checkpoint_relative_directory=role, model_file=model.name,
            state_file=state.name, model_sha256=M.sha256(model), state_sha256=M.sha256(state),
            checkpoint_epoch=epoch, metadata_kind=kind, frozen=role=='flow')
    for constant, role, key in [('FLOW_MODEL_SHA256','flow','model_sha256'),
                               ('FLOW_STATE_SHA256','flow','state_sha256'),
                               ('P064_B_MODEL_SHA256','aerodynamic','model_sha256'),
                               ('P064_B_STATE_SHA256','aerodynamic','state_sha256')]:
        monkeypatch.setattr(M,constant,roles[role][key])
    payload=dict(schema_version=1, architecture=M.ARCHITECTURE,
        flow_architecture=M.ARCHITECTURE, aerodynamic_architecture=M.ARCHITECTURE,
        config_sha256=M.CONFIG_SHA256, normalization_sha256=M.NORMALIZATION_SHA256,
        precision_protocol=M.PRECISION_PROTOCOL, input_sha256={},
        history_input=M._p026_history_input(1), history_inventory=M._p026_inventory(),
        history_state_module_sha256=M.P026_HISTORY_STATE_SHA256,
        history_inference_module_sha256=M.P026_HISTORY_INFERENCE_SHA256,
        aerodynamic_initial_model_sha256=M.P028_AERO_PARENT_MODEL_SHA256,
        aerodynamic_initial_state_sha256=M.P028_AERO_PARENT_STATE_SHA256, **roles)
    identity=SimpleNamespace(payload=payload,aerodynamic=SimpleNamespace(directory=parent/'aerodynamic'))
    result=dict(fixed_scales={'field':.1,'force':.2}, flow_terminal_tensor_sha256='fixture',
                source_spec={'scales_receipt':{'sha256':'a'*64},'source_sha256':{}})
    output=tmp_path/'candidate';output.mkdir()
    saved={}

    def save(directory, *, models, optimizer, epoch, metadata):
        directory.mkdir()
        (directory/'FNO.0.1.mdlus').write_bytes(b'new trained flow')
        (directory/'checkpoint.0.1.pt').write_bytes(b'new flow state')
        saved.update(metadata)

    def producer_load(directory, *, models, metadata_dict, device):
        if directory.name=='flow':metadata_dict.update(saved)
        return 1

    P.save_terminal(output,FakeModel(),FakeModel(),None,None,identity,result,
                    lambda cfg:FakeModel(),save,producer_load,lambda model:'fixture')
    path=output/'dual_model_manifest.json'
    return path,json.loads(path.read_text()),saved


def test_actual_producer_manifest_and_loaded_metadata(produced,monkeypatch):
    path,payload,flow_metadata=produced
    identity=M.validate_dual_fno_manifest(path)
    assert identity.flow.epoch==1 and identity.aerodynamic.epoch==1
    assert payload['flow']['frozen'] is False and payload['aerodynamic']['frozen'] is True
    aero=dict(status=M.P064_AERO_KIND['B'],checkpoint_epoch=1,training_experiment='FC-P064',arm='B',
        history_profile='p026_k1',history_k=1,model_in_channels=6,
        training_protocol_sha256=M.P064_B_PROTOCOL_SHA256,training_protocol_file='training_protocol.json',
        history_state_module_sha256=M.P026_HISTORY_STATE_SHA256,
        history_inference_module_sha256=M.P026_HISTORY_INFERENCE_SHA256,
        training_windows=256,optimizer_steps=32,accumulation_windows=8,
        actual_learning_rate=M.P026_LEARNING_RATE,parent_sampler_order_sha256=M.P026_ORDER_SHA256,
        schedule_sha256=M.P064_SCHEDULE_SHA256['B'],parent_history_inventory=M._p026_inventory(),
        flow_parent_model_sha256=M.FLOW_MODEL_SHA256,flow_parent_state_sha256=M.FLOW_STATE_SHA256,
        aerodynamic_parent_model_sha256=M.P028_AERO_PARENT_MODEL_SHA256,
        aerodynamic_parent_state_sha256=M.P028_AERO_PARENT_STATE_SHA256,
        selection_performed=False,validation_accessed=False,frozen_test_accessed=False,ppo_executed=False)
    calls=[]

    def load(directory, *, models, metadata_dict, device):
        calls.append(directory.name)
        metadata_dict.update(flow_metadata if directory.name=='flow' else aero)
        return 1

    monkeypatch.setattr(M,'validate_runtime_precision',lambda:None)
    monkeypatch.setattr(M,'_runtime_architecture',lambda cfg:M.ARCHITECTURE)
    monkeypatch.setattr(M,'_p026_history_adapter_factory',lambda:lambda flow,aero,k:('K',k))
    cfg=SimpleNamespace(model=SimpleNamespace(in_channels=6))
    adapter,_=M.load_dual_fno(path,cfg,'cpu',build_model=lambda cfg:FakeModel(),load_checkpoint=load)
    assert adapter==('K',1) and calls==['flow','aerodynamic']
    aero['training_protocol_sha256']=payload['training_protocol_sha256']
    with pytest.raises(ValueError,match='metadata'):
        M.load_dual_fno(path,cfg,'cpu',build_model=lambda cfg:FakeModel(),load_checkpoint=load)


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(parent_manifest_sha256=M.P028_PARENT_MANIFEST_SHA256),
    lambda p:p['aerodynamic'].update(frozen=False),
    lambda p:p['history_input'].update(history_length=4),
    lambda p:p.update(optimizer_steps=171),
    lambda p:p.update(scientific_admission=True),
    lambda p:p.update(aerodynamic_parent_model_sha256=M.P028_AERO_PARENT_MODEL_SHA256),
])
def test_wrong_identity_rejected(produced,mutation):
    path,payload,_=produced
    mutation(payload);path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):M.validate_dual_fno_manifest(path)


@pytest.mark.parametrize('key,value',[('horizon',10),('horizon',25.0),('future_truth_inputs',0),('scale_horizon',25),('fixed_scales',{'field':True,'force':1.})])
def test_wrong_effective_protocol_rejected(produced,key,value):
    path,payload,_=produced
    protocol=copy.deepcopy(payload['training_semantics']);protocol[key]=value
    target=path.parent/'training_protocol.json';target.write_text(json.dumps(protocol))
    payload.update(training_semantics=protocol,training_protocol_sha256=M.sha256(target))
    if key=='fixed_scales':payload[key]=value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):M.validate_dual_fno_manifest(path)


def test_cli_only_identity_delta():
    candidate=(ROOT/'scripts/audit_dev30_validation_diagnostic_p064.py').read_text()
    candidate=candidate.replace('P064_H25_KIND = "FC_P064_B_H25_BOUNDED_CONTROL_AWARE_FLOW_REPAIR"\n','')
    candidate=candidate.replace(', P064_H25_KIND: P064_H25_KIND','').replace(', P064_H25_KIND','').replace('            P064_H25_KIND,\n','')
    assert hashlib.sha256(candidate.encode()).hexdigest() == 'e2e701cf138f5755468bdae6fc83e418d63069f0acd09c638715d32fb5c689dc'
