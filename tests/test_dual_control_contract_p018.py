"""Synthetic provenance tests, not scientific admission evidence."""
import json

import pytest

import fluid_control.dual_control_contract as c
from test_dual_control_contract_p015 import evidence as p015_evidence


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root, receipt, identity, _, run = p015_evidence.__wrapped__(tmp_path, monkeypatch)
    def load(p):
        return json.loads(p.read_text())
    def write(p, data):
        p.write_text(json.dumps(data))
    # Translate only synthetic fixture identities; production never translates evidence.
    for p in root.parent.rglob('*.json'):
        p.write_text(p.read_text().replace('P015', 'P018').replace('FC-P015', 'FC-P018')
                     .replace('fcp015_window_accumulation_dual_fno', 'fcp018_reduced_rate_dual_fno'))
    identity.payload['kind'] = c.P018_SYSTEM_KIND
    protocol = root.parent/'candidate/training_protocol.json'
    write(protocol, {'synthetic_protocol': True})
    monkeypatch.setattr(c, 'P018_PROTOCOL_SHA256', c.sha256(protocol))
    execution = root.parent/'execution_approval.json'
    write(execution, {'synthetic_execution': True})
    monkeypatch.setattr(c, 'P018_APPROVAL_SHA256', c.sha256(execution))
    extra = dict(actual_learning_rate=c.P018_LEARNING_RATE,
                 training_protocol_sha256=c.P018_PROTOCOL_SHA256)
    result_path = root.parent/'candidate/result.json'
    result = load(result_path)
    result.update(extra, status='FC_P018_REDUCED_RATE_TRAINING_COMPLETE_NOT_ADMISSION',
        training_protocol_file='training_protocol.json', dual_model_manifest_sha256='a'*64,
        config_sha256='f'*64, base_config_sha256='f'*64,
        official_pair_fresh_reload_verified=True, selection_performed=False,
        validation_accessed=False, frozen_test_accessed=False, ppo_executed=False,
        flow_tensor_sha256_before='1'*64, aerodynamic_tensor_sha256_before='1'*64)
    write(result_path, result)
    lineage = load(root/'lineage.json')
    lineage.update(extra, candidate_result_sha256=c.sha256(result_path),
        training_approval_sha256=c.P018_APPROVAL_SHA256,
        execution_observation_sha256=c.P018_OBSERVATION_SHA256, invocation_id=c.P018_INVOCATION)
    write(root/'lineage.json', lineage)
    write(root.parent/'candidate_audit.json', lineage)
    completion = {k:v for k,v in lineage.items() if k != 'checkpoint_epoch'}
    completion.update(status='FC_P018_TRAINING_COMPLETE_NOT_ADMISSION',
        training_protocol_file='training_protocol.json',
        unit='fluid-control-fcp018-reduced-rate-20261005.service', scientific_admission=False,
        ppo_executed=False, candidate_audit_sha256=c.sha256(root.parent/'candidate_audit.json'),
        terminal_unit=dict(LoadState='loaded', InvocationID=c.P018_INVOCATION,
            ActiveState='active', SubState='exited', MainPID='0', Result='success',
            ExecMainCode='1', ExecMainStatus='0'))
    write(root.parent/'completion_receipt.json', completion)
    reload_path = root.parent/'dual_reload_receipt.json'
    reloaded = load(reload_path)
    reloaded.update(extra, training_result_sha256=c.sha256(result_path),
                    training_protocol_file='training_protocol.json')
    write(reload_path, reloaded)
    def refresh():
        # Rebind outer hashes so negative tests reach semantic checks.
        approval_path = root/'evidence/formal_evaluation_approval.json'
        approval = load(approval_path)
        approval.update(extra, candidate_result_sha256=c.sha256(result_path),
            dual_reload_receipt_sha256=c.sha256(reload_path),
            candidate_completion_receipt_sha256=c.sha256(root.parent/'completion_receipt.json'))
        write(approval_path, approval)
        hashes = dict(lineage_sha256=c.sha256(root/'lineage.json'),
            formal_evaluation_approval_sha256=c.sha256(approval_path),
            precision_sha256=c.sha256(root/'precision.json'), posteval_chain_receipt_sha256='9'*64)
        for p in (root/'step_receipts').glob('*.json'):
            value=load(p); value.update(hashes); write(p,value)
        value=load(receipt); value.update(hashes)
        value['sha256']={str(p.relative_to(root)):c.sha256(p) for p in root.rglob('*') if p.is_file() and p!=receipt}
        write(receipt,value)
    refresh()
    return root, receipt, identity, refresh, run


def test_p018_binding_is_explicit_and_not_control_success(evidence):
    root, _, _, refresh, run = evidence
    result=run()
    assert result['training_experiment']=='FC-P018'
    assert result['actual_learning_rate']==1.5625e-7
    assert result['training_protocol_sha256']==c.P018_PROTOCOL_SHA256
    assert result['policy_trained'] is False
    assert result['real_cfd_control_validated'] is False
    for name in ('force_window/result.json','development_gate.json'):
        (root/name).write_text(json.dumps({'status':'DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL'}))
    refresh()
    with pytest.raises(ValueError, match='not passed'): run()


@pytest.mark.parametrize('relative,key,value', [
    ('lineage.json','actual_learning_rate',1e-5),
    ('lineage.json','training_protocol_sha256','0'*64),
    ('lineage.json','execution_observation_sha256','0'*64),
    ('lineage.json','invocation_id','wrong'),
    ('lineage.json','optimizer_steps',1368),
    ('../candidate/result.json','actual_learning_rate',1e-5),
    ('../candidate/result.json','training_protocol_file','../training_protocol.json'),
    ('../candidate/result.json','flow_tensor_sha256_before','0'*64),
    ('../completion_receipt.json','status','FC_P015_TRAINING_COMPLETE_NOT_ADMISSION'),
    ('../completion_receipt.json','checkpoint_sha256','0'*64),
    ('../completion_receipt.json','execution_observation_sha256','0'*64),
    ('../completion_receipt.json','terminal_unit',{}),
    ('../dual_reload_receipt.json','status','FC_P015_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION'),
    ('../dual_reload_receipt.json','actual_learning_rate',1e-5),
    ('../dual_reload_receipt.json','gpu_used',True),
    ('../dual_reload_receipt.json','training_protocol_file','wrong'),
    ('evidence/formal_evaluation_approval.json','formal_evaluation_authorized',False),
    ('step_receipts/dynamic6.json','status','FC_P015_POSTEVAL_STEP_COMPLETE'),
])
def test_semantic_proof_mismatch_rejected(evidence, relative, key, value):
    root, _, _, refresh, run=evidence
    path=root/relative; payload=json.loads(path.read_text()); payload[key]=value
    path.write_text(json.dumps(payload)); refresh()
    with pytest.raises(ValueError): run()


@pytest.mark.parametrize('relative', ['candidate/training_protocol.json','execution_approval.json',
    'completion_receipt.json','candidate_audit.json','dual_reload_receipt.json'])
def test_missing_proof_rejected(evidence, relative):
    root, _, _, _, run=evidence
    (root.parent/relative).unlink()
    with pytest.raises((ValueError,FileNotFoundError)): run()


@pytest.mark.parametrize('relative', ['candidate/training_protocol.json','execution_approval.json'])
def test_actual_pinned_bytes_required(evidence, relative):
    root, _, _, _, run=evidence
    (root.parent/relative).write_text('{}')
    with pytest.raises(ValueError): run()


@pytest.mark.parametrize('kind', [c.SYSTEM_KIND,c.P015_SYSTEM_KIND])
def test_p018_receipt_cannot_enter_legacy_profile(evidence, kind):
    _, _, identity, _, run=evidence
    identity.payload['kind']=kind
    with pytest.raises(ValueError,match='receipt identity'): run()
