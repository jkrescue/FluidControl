"""Synthetic JSON/byte fixtures for identity checks, never scientific results."""
import json
from types import SimpleNamespace

import pytest

from fluid_control.dual_control_contract import PROTOCOL, check_receipt_identity, verify_receipt_files
from fluid_control.dual_fno import sha256


def test_receipt_requires_dual_identity_and_does_not_accept_legacy_completion():
    expected = {'dual_manifest_sha256': 'a' * 64, 'flow_model_sha256': 'b' * 64,
                'flow_state_sha256': 'c' * 64, 'checkpoint_sha256': 'd' * 64,
                'checkpoint_state_sha256': 'e' * 64}
    value = {'status': 'FC_P013_POSTEVAL_COMPLETE', 'candidate_kind': 'fcp013_independent_force_dual_fno',
             'checkpoint_epoch': 1, 'protocol': PROTOCOL, 'frozen_test_accessed': False,
             'ppo_auto_launched': False, **expected}
    check_receipt_identity(value, expected)
    for key in ('status', 'dual_manifest_sha256', 'flow_state_sha256', 'checkpoint_sha256'):
        broken = dict(value)
        broken[key] = 'unrelated'
        with pytest.raises(ValueError):
            check_receipt_identity(broken, expected)


def file_fixture(root):
    names = ('lineage.json', 'precision.json', 'development_gate.json', 'force_window/result.json',
             'validation10/evaluation.json', 'validation10/segments.json', 'validation10/endpoint_gate.json',
             'validation10/diagnostic.json', 'dynamic6/diagnostic.json', 'evidence/formal_evaluation_approval.json',
             'dynamic6/evaluation.json', 'dynamic6/segments.json', 'step_receipts/validation10.json',
             'step_receipts/dynamic6.json', 'step_receipts/force_window.json')
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'software_fixture_only': True}))
    return {name: sha256(root / name) for name in names}


def test_files_cannot_be_missing_tampered_or_outside_receipt_directory(tmp_path):
    root = tmp_path / 'posteval'
    files = file_fixture(root)
    verify_receipt_files(root, files)
    missing = dict(files)
    del missing['development_gate.json']
    with pytest.raises(ValueError):
        verify_receipt_files(root, missing)
    outside = tmp_path / 'outside.json'
    outside.write_text('outside')
    with pytest.raises(ValueError):
        verify_receipt_files(root, {**files, '../outside.json': sha256(outside)})
    (root / 'dynamic6/evaluation.json').write_text('tampered')
    with pytest.raises(ValueError):
        verify_receipt_files(root, files)


def test_composed_binding_recomputes_gate_and_requires_same_step_identity(tmp_path, monkeypatch):
    import fluid_control.dual_control_contract as contract
    root = tmp_path / 'posteval'
    files = file_fixture(root)
    expected = {'dual_manifest_sha256': 'a' * 64, 'flow_model_sha256': 'b' * 64,
                'flow_state_sha256': 'c' * 64, 'checkpoint_sha256': 'd' * 64,
                'checkpoint_state_sha256': 'e' * 64}
    identity = SimpleNamespace(
        aerodynamic=SimpleNamespace(directory=tmp_path / 'aero', model_sha256='d' * 64, state_sha256='e' * 64),
        flow=SimpleNamespace(model_sha256='b' * 64, state_sha256='c' * 64),
        manifest_sha256='a' * 64, payload={'config_sha256': 'f' * 64, 'normalization_sha256': '0' * 64},
    )
    # Lower-level manifest/runtime verification has its own real-file tests.
    monkeypatch.setattr(contract, 'validate_dual_fno_manifest', lambda *a, **k: identity)
    monkeypatch.setattr(contract, 'validate_dual_runtime_files', lambda *a, **k: None)
    for name in ('validation10', 'dynamic6', 'force_window'):
        (root / 'step_receipts' / f'{name}.json').write_text(json.dumps({
            'status': 'FC_P013_POSTEVAL_STEP_COMPLETE', 'step': name, **expected}))
    auditor = tmp_path / 'software_gate_fixture.py'
    auditor.write_text('import json\ndef audit(path, checkpoint):\n return json.loads(path.read_text())\n')
    monkeypatch.setattr(contract, 'DEVELOPMENT_AUDITOR_SHA', sha256(auditor))
    receipt = root / 'receipt.json'

    def refresh_receipt():
        receipt.write_text(json.dumps({
            'status': 'FC_P013_POSTEVAL_COMPLETE', 'candidate_kind': 'fcp013_independent_force_dual_fno',
            'checkpoint_epoch': 1, 'protocol': PROTOCOL, 'frozen_test_accessed': False,
            'ppo_auto_launched': False, **expected,
            'sha256': {name: sha256(root / name) for name in files},
        }))

    def run():
        return contract.verify_dual_control_binding(
            manifest_path=tmp_path / 'manifest.json', expected_manifest_sha256='a' * 64,
            training_config=tmp_path / 'training.yaml', normalization_path=tmp_path / 'normalization.json',
            checkpoint_dir=identity.aerodynamic.directory, expected_checkpoint_sha256='d' * 64,
            posteval_receipt=receipt, expected_posteval_receipt_sha256=sha256(receipt), development_auditor=auditor)

    for name in ('force_window/result.json', 'development_gate.json'):
        (root / name).write_text(json.dumps({'status': 'DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL'}))
    refresh_receipt()
    with pytest.raises(ValueError, match='not passed'):
        run()
    for name in ('force_window/result.json', 'development_gate.json'):
        (root / name).write_text(json.dumps({'status': 'DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS'}))
    refresh_receipt()
    binding = run()
    assert binding['canonical_endpoint_window_dynamic_gates_still_required']
    assert not binding['policy_trained'] and not binding['real_cfd_control_validated']
    step = root / 'step_receipts/dynamic6.json'
    value = json.loads(step.read_text())
    value['flow_state_sha256'] = '9' * 64
    step.write_text(json.dumps(value))
    refresh_receipt()
    with pytest.raises(ValueError, match='same complete dual system'):
        run()
