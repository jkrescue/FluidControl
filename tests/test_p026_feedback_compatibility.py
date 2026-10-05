"""Synthetic evidence only; no PPO/model/CFD execution or real artifact loads."""
import copy
import json
from pathlib import Path

import pytest
from test_candidate_openfoam_readiness_adapter import MODULE as exporter, fixture, dump
from test_full40_canonical_openfoam_feedback import MODULE as feedback


def p026_fixture(tmp_path, k):
    paths = fixture(tmp_path)
    approved = json.loads(paths['approved_preflight_path'].read_text())
    audit = json.loads(paths['canonical_audit_path'].read_text())
    contract = approved['command_contract']
    identity = approved['candidate_readiness']['candidate_identity']
    ready = approved['canonical_preflight']
    kind = f'FC_P026_K{k}_HISTORY_FORCE_FNO'
    dual = dict(status='DUAL_CONTROL_IDENTITY_VERIFIED_NOT_CONTROL_SUCCESS',
                canonical_endpoint_window_dynamic_gates_still_required=True,
                policy_trained=False, real_cfd_control_validated=False,
                dual_system_kind=kind, training_experiment='FC-P026',
                checkpoint_sha256='a'*64, checkpoint_state_sha256='b'*64,
                training_config_sha256='c'*64, normalization_sha256='d'*64,
                dual_manifest_sha256='e'*64, flow_model_sha256='f'*64,
                flow_state_sha256='1'*64, posteval_receipt_sha256='2'*64)
    runtime = dict(schema_version=1, profile=f'p026_k{k}', history_length=k,
                   flow_input_channels=6, aerodynamic_input_channels=6 if k == 1 else 18,
                   left_padding='trajectory_frame0', autoregressive_state_source='frozen_flow_prediction',
                   future_state_inputs=False, future_force_inputs=False, manifest_kind=kind,
                   dual_manifest_sha256='e'*64, flow_model_sha256='f'*64, flow_state_sha256='1'*64,
                   aerodynamic_model_sha256='a'*64, aerodynamic_state_sha256='b'*64,
                   config_sha256='c'*64, normalization_sha256='d'*64,
                   training_protocol_sha256='3'*64,
                   history_state_module_sha256=exporter.P026_HISTORY_STATE_SHA256,
                   history_inference_module_sha256=exporter.P026_HISTORY_INFERENCE_SHA256)
    dual['fno_history_runtime'] = runtime
    contract.update(candidate_kind=kind, full40_normalization_sha256='d'*64)
    identity.update(candidate_kind=kind, normalization_sha256='d'*64)
    for obj in (contract, identity, ready, audit):
        obj['dual_control_binding'] = copy.deepcopy(dual)
    # Actual low-level readiness has nested runtime, but no top-level copy.
    for obj in (contract, identity, audit):
        obj['fno_history_runtime'] = copy.deepcopy(runtime)
    audit['training_executed'] = True
    dump(paths['approved_preflight_path'], approved)
    dump(paths['canonical_audit_path'], audit)
    bound = json.loads(paths['binding_receipt_path'].read_text())
    bound.update(command_contract=contract,
                 candidate_readiness_sha256=exporter.sha256(paths['approved_preflight_path']),
                 canonical_audit_sha256=exporter.sha256(paths['canonical_audit_path']))
    dump(paths['binding_receipt_path'], bound)
    return paths


@pytest.mark.parametrize('k', [1, 4])
def test_normal_export_reaches_existing_identity_artifact_gate(tmp_path, k):
    paths = p026_fixture(tmp_path, k)
    ready, audit = exporter.adapt(**paths)
    assert ready['dual_control_binding'] == audit['dual_control_binding']
    assert audit['iterations'][-1]['checkpoint'] == str(paths['policy_path'])
    with pytest.raises(ValueError, match='requires VecNormalize path'):
        feedback.validate_dual_policy_contract(audit=audit, readiness=ready, gate={},
            policy=paths['policy_path'], normalization_sha256='d'*64,
            vecnormalize=None, expected_vecnormalize_sha256=None)


@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('side', ['audit', 'readiness', 'gate'])
@pytest.mark.parametrize('marker', ['kind', 'profile', 'runtime'])
def test_marker_only_cannot_downgrade(k, side, marker):
    evidence = dict(audit={}, readiness={}, gate={})
    evidence[side] = {'kind': f'FC_P026_K{k}_HISTORY_FORCE_FNO'} if marker == 'kind' else (
        {'profile': f'p026_k{k}'} if marker == 'profile' else {'fno_history_runtime': {}})
    with pytest.raises(ValueError, match='dual_control_binding'):
        feedback.validate_dual_policy_contract(**evidence, policy=Path('never_read'),
            normalization_sha256='d'*64, vecnormalize=None, expected_vecnormalize_sha256=None)


@pytest.mark.parametrize('field,value', [('profile', 'p026_k4'), ('history_length', 4),
    ('aerodynamic_model_sha256', '9'*64), ('future_state_inputs', True),
    ('history_state_module_sha256', None), ('history_state_module_sha256', '9'*64),
    ('history_inference_module_sha256', '8'*64), ('training_protocol_sha256', '7'*64),
    ('future_force_inputs', 0), ('history_length', True), ('schema_version', True),
    ('flow_input_channels', 6.0)])
def test_conflicting_history_rejected_before_artifact_load(tmp_path, field, value):
    paths = p026_fixture(tmp_path, 1)
    ready, audit = exporter.adapt(**paths)
    audit['dual_control_binding']['fno_history_runtime'][field] = value
    ready['dual_control_binding'] = copy.deepcopy(audit['dual_control_binding'])
    with pytest.raises(ValueError, match='P026 history'):
        feedback.validate_dual_policy_contract(audit=audit, readiness=ready, gate={},
            policy=Path('never_read'), normalization_sha256='d'*64,
            vecnormalize=None, expected_vecnormalize_sha256=None)


@pytest.mark.parametrize('mutation', ['missing', 'conflicting_runtime'])
def test_export_rejects_incomplete_p026(tmp_path, mutation):
    paths = p026_fixture(tmp_path, 4)
    approved = json.loads(paths['approved_preflight_path'].read_text())
    if mutation == 'missing':
        del approved['command_contract']['dual_control_binding']
    else:
        approved['command_contract']['fno_history_runtime']['profile'] = 'p026_k1'
    dump(paths['approved_preflight_path'], approved)
    with pytest.raises(ValueError, match='dual control bindings|P026 history'):
        exporter.adapt(**paths)


def test_malformed_nested_binding_is_value_error(tmp_path):
    paths = p026_fixture(tmp_path, 1)
    ready, audit = exporter.adapt(**paths)
    with pytest.raises(ValueError, match='nested binding'):
        exporter.validate_p026_history_binding(ready['dual_control_binding'],
            {'dual_control_binding': None, 'fno_history_runtime': audit['fno_history_runtime']})
