"""Software fixtures only; no scientific admission, PPO, or CFD execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import test_p015_candidate_ppo_profiles as old
from test_full40_canonical_ppo_dry_run import MODULE as trainer
from test_full40_canonical_openfoam_feedback import MODULE as feedback
from test_p013_canonical_endpoint_path_view import evidence

P = old.P
M = P.READY.MODULE
KIND = "fcp018_reduced_rate_dual_fno"


def fixture(tmp_path, monkeypatch):
    args = old.fixture(tmp_path, monkeypatch)
    args['profile'] = 'P018'
    root = args['candidate_root']
    protocol = root / 'candidate/training_protocol.json'
    P.READY.write_json(protocol, {'software_fixture': 'protocol'})
    observation = args['repo_root'] / 'docs/FC_P018_RUNNING_EXECUTION_20261005.json'
    P.READY.write_json(observation, {'software_fixture': 'external observation'})
    # Substitute only immutable byte pins for synthetic files, not validators.
    monkeypatch.setattr(M, 'FC_P018_PROTOCOL_SHA', P.READY.digest(protocol))
    monkeypatch.setattr(M, 'FC_P018_APPROVAL_SHA', P.READY.digest(root / 'execution_approval.json'))
    monkeypatch.setattr(M, 'FC_P018_OBSERVATION_SHA', P.READY.digest(observation))
    identity = dict(training_experiment='FC-P018', actual_learning_rate=1.5625e-7,
                    training_protocol_sha256=M.FC_P018_PROTOCOL_SHA,
                    training_protocol_file='training_protocol.json')
    args['lineage'].update(identity, candidate_kind=KIND,
        execution_observation_sha256=M.FC_P018_OBSERVATION_SHA,
        status='FC_P018_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION')
    P.READY.write_json(root / 'candidate_audit.json', args['lineage'])
    completion = M.load(root / 'completion_receipt.json')
    completion.update(identity, status='FC_P018_TRAINING_COMPLETE_NOT_ADMISSION',
        execution_observation_sha256=M.FC_P018_OBSERVATION_SHA,
        candidate_audit_sha256=P.READY.digest(root / 'candidate_audit.json'))
    P.READY.write_json(root / 'completion_receipt.json', completion)
    approval_path = args['receipt_path'].parent / 'evidence/formal_evaluation_approval.json'
    approval = M.load(approval_path)
    approval.update(candidate_kind=KIND,
        candidate_completion_receipt_sha256=P.READY.digest(root / 'completion_receipt.json'))
    P.READY.write_json(approval_path, approval)
    precision_path = args['receipt_path'].parent / 'precision.json'
    precision = M.load(precision_path)
    precision['status'] = 'FC_P018_FORMAL_EVALUATION_DEFAULT_TF32_HIGH'
    P.READY.write_json(precision_path, precision)
    # P018 must not depend on the obsolete candidate-relative observation file.
    (root / 'running_execution_evidence.json').unlink()
    return args


def test_p018_actual_terminal_layout(tmp_path, monkeypatch):
    assert M.p013_candidate_identity(**fixture(tmp_path, monkeypatch))['epoch'] == 1


@pytest.mark.parametrize('key,value', [
    ('actual_learning_rate', 1e-5), ('training_protocol_sha256', '0'*64),
    ('training_protocol_file', '../protocol.json'), ('training_experiment', 'FC-P015'),
    ('optimizer_steps', 1368), ('training_windows', 171), ('accumulation_windows', 1),
    ('candidate_kind', old.KIND), ('status', 'FC_P015_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION')])
def test_reject_cross_profile_and_protocol(tmp_path, monkeypatch, key, value):
    args = fixture(tmp_path, monkeypatch)
    args['lineage'][key] = value
    with pytest.raises(ValueError):
        M.p013_candidate_identity(**args)


@pytest.mark.parametrize('relative', ['candidate/training_protocol.json', 'completion_receipt.json',
                                    'dual_reload_receipt.json', 'execution_approval.json'])
def test_mutated_evidence_rejected(tmp_path, monkeypatch, relative):
    args = fixture(tmp_path, monkeypatch)
    P.READY.write_json(args['candidate_root'] / relative, {'changed': True})
    with pytest.raises(ValueError):
        M.p013_candidate_identity(**args)


def test_mutated_external_observation_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    P.READY.write_json(args['repo_root'] / 'docs/FC_P018_RUNNING_EXECUTION_20261005.json', {})
    with pytest.raises(ValueError, match='external execution'):
        M.p013_candidate_identity(**args)


@pytest.mark.parametrize('mode', ['dry-run', 'execute'])
def test_dual_command_preserves_separate_configs(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(P, 'KIND', KIND)
    P.test_dual_command_has_all_identities_and_vecnormalize(tmp_path, mode)


@pytest.mark.parametrize('key', ['endpoint_gate_path', 'window_gate_path', 'dynamic_gate_path', 'development_gate_path'])
def test_all_original_numeric_failures_still_reject(tmp_path, monkeypatch, key):
    monkeypatch.setattr(P, 'KIND', KIND)
    P.test_p013_does_not_bypass_any_legacy_numeric_gate(tmp_path, monkeypatch, key)


def test_host_export_keeps_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'KIND', KIND)
    P.test_export_preserves_binding_and_translates_only_policy(tmp_path)


@pytest.mark.parametrize('marker', ['FC_P018_REDUCED_RATE_FORCE_FNO',
    'FC_P018_POSTEVAL_COMPLETE', KIND, 'FC-P018'])
def test_no_single_model_fallback(marker):
    with pytest.raises(ValueError, match='complete dual arguments'):
        trainer.require_single_model_identity({'metadata': {'training_experiment': marker}})


def test_actual_checkpoint_marker_rejects_single_model(tmp_path):
    torch.save({'metadata': {'training_experiment': 'FC-P018'}}, tmp_path / 'checkpoint.0.1.pt')
    with pytest.raises(ValueError, match='complete dual arguments'):
        trainer.verify_single_model_selection(SimpleNamespace(checkpoint_dir=tmp_path), {})


@pytest.mark.parametrize('side', ['audit', 'readiness', 'gate'])
@pytest.mark.parametrize('marker', ['FC_P018_REDUCED_RATE_FORCE_FNO', KIND, 'FC-P018'])
def test_feedback_missing_binding_rejects(side, marker):
    values = dict(audit={}, readiness={}, gate={})
    values[side] = {'metadata': {'marker': marker}}
    with pytest.raises(ValueError, match='dual_control_binding'):
        feedback.validate_dual_policy_contract(**values, policy=Path('not_read.zip'),
            normalization_sha256='a'*64, vecnormalize=None, expected_vecnormalize_sha256=None)


def test_endpoint_identity_label_only(evidence):
    args, gate, _ = evidence
    path = args['dual_options']['dual_posteval_receipt']
    receipt = trainer.read_json(path)
    receipt['status'] = 'FC_P018_POSTEVAL_COMPLETE'
    path.write_text(json.dumps(receipt))
    before = {name: args[name].read_bytes() for name in ('validation_report', 'validation_segments', 'validation_gate')}
    actual, mapping = trainer.recompute_p013_endpoint_view(**args)
    assert actual == gate and mapping['numerical_evidence_changed'] is False
    assert mapping['status'] == 'P018_ENDPOINT_PATH_VIEW_IDENTITY_VERIFIED'
    assert all(args[name].read_bytes() == value for name, value in before.items())
