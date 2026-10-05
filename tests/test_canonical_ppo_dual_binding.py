"""CPU identity-routing fixtures; no policy training or scientific acceptance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fluid_control.dual_control_contract as contract

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('canonical_dual_fixture', ROOT / 'scripts/train_full40_hydrogym_ppo_canonical.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def full_args():
    return SimpleNamespace(
        dual_fno_manifest=Path('/fixture/manifest.json'), expected_dual_fno_manifest_sha256='a' * 64,
        dual_training_config=Path('/fixture/training.yaml'), dual_posteval_receipt=Path('/fixture/posteval/receipt.json'),
        expected_dual_posteval_receipt_sha256='b' * 64,
        data=Path('/fixture/data'), checkpoint_dir=Path('/fixture/aerodynamic'),
        train_only_smoke=False, allow_calibrated_epoch_zero=False,
        vecnormalize_output=Path('/fixture/new_policy/vecnormalize.pkl'),
    )


def ready():
    return {'status': 'FULL40_CANONICAL_PPO_EXECUTION_READY', 'checkpoint_sha256': 'c' * 64, 'blockers': []}


def test_default_single_model_readiness_unchanged():
    old = ready()
    assert module.attach_dual_readiness(SimpleNamespace(), old) == old
    assert 'dual_control_binding' not in old


def test_partial_dual_options_block_instead_of_falling_back_to_single_model():
    value = module.attach_dual_readiness(SimpleNamespace(dual_fno_manifest=Path('/fixture')), ready())
    assert value['status'] == 'FULL40_CANONICAL_PPO_EXECUTION_BLOCKED'
    assert any('dual_identity_invalid' in item for item in value['blockers'])


@pytest.mark.parametrize('field,value', [('train_only_smoke', True), ('allow_calibrated_epoch_zero', True),
                                       ('vecnormalize_output', None)])
def test_no_smoke_single_calibration_or_missing_normalization_bypass(field, value):
    args = full_args()
    setattr(args, field, value)
    result = module.attach_dual_readiness(args, ready())
    assert result['status'] == 'FULL40_CANONICAL_PPO_EXECUTION_BLOCKED'


def test_original_canonical_blockers_cannot_be_overridden(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError('blocked canonical gates must not reach dual acceptance')
    monkeypatch.setattr(contract, 'verify_dual_control_binding', forbidden)
    old = {'status': 'FULL40_CANONICAL_PPO_EXECUTION_BLOCKED', 'blockers': ['window_failed']}
    result = module.attach_dual_readiness(full_args(), old)
    assert result['status'] == old['status'] and 'window_failed' in result['blockers']


def test_binding_routes_actual_primary_and_manifest_identities(monkeypatch):
    seen = {}
    def verify(**kwargs):
        seen.update(kwargs)
        return {'software_fixture_only': True}
    monkeypatch.setattr(contract, 'verify_dual_control_binding', verify)
    args = full_args()
    result = module.attach_dual_readiness(args, ready())
    assert result['dual_control_binding'] == {'software_fixture_only': True}
    assert seen['expected_checkpoint_sha256'] == ready()['checkpoint_sha256']
    assert seen['normalization_path'] == args.data / 'normalization.json'
    assert seen['development_auditor'] == ROOT / 'scripts/audit_dynamic_fno_development_gates.py'


def test_execute_rechecks_binding_and_uses_dual_loader_without_changing_ppo_hyperparameters():
    source = (ROOT / 'scripts/train_full40_hydrogym_ppo_canonical.py').read_text()
    assert 'runtime_dual = dual_binding_from_args(args, readiness)' in source
    assert 'readiness.get("dual_control_binding") != runtime_dual' in source
    assert 'network, dual_identity = load_dual_fno(' in source
    assert 'expected_manifest_sha256=args.expected_dual_fno_manifest_sha256' in source
    assert 'n_steps=128' in source and 'batch_size=256' in source


@pytest.mark.parametrize('source', ['checkpoint', 'validation_report', 'validation_gate',
                                    'validation_segments', 'window_gate', 'dynamic_gate', 'readiness'])
def test_omitting_all_dual_flags_cannot_use_p013_identity(tmp_path, source):
    import torch

    args = SimpleNamespace(checkpoint_dir=tmp_path)
    old = ready()
    if source == 'checkpoint':
        torch.save({'metadata': {'status': 'FC_P013_INDEPENDENT_FORCE_FNO_AERODYNAMIC_CHECKPOINT'}},
                   tmp_path / 'checkpoint.0.1.pt')
    elif source == 'readiness':
        old['dual_control_binding'] = {'dual_manifest_sha256': 'a' * 64}
    else:
        path = tmp_path / f'{source}.json'
        path.write_text(json.dumps({'lineage': {'dual_manifest_sha256': 'a' * 64}}))
        setattr(args, source, path)
    result = module.attach_dual_readiness(args, old)
    assert result['status'] == 'FULL40_CANONICAL_PPO_EXECUTION_BLOCKED'
    assert 'complete dual arguments' in result['blockers'][-1]


def test_genuine_legacy_checkpoint_metadata_keeps_single_route(tmp_path):
    import torch

    torch.save({'metadata': {'status': 'LEGACY_TRAIN_COMPLETE'},
                'optimizer_state_dict': {'fixture': torch.ones(2)}},
               tmp_path / 'checkpoint.0.2.pt')
    args = SimpleNamespace(checkpoint_dir=tmp_path)
    assert module.attach_dual_readiness(args, ready()) == ready()


def test_unreadable_checkpoint_identity_fails_closed(tmp_path):
    (tmp_path / 'checkpoint.0.1.pt').write_bytes(b'invalid checkpoint')
    result = module.attach_dual_readiness(SimpleNamespace(checkpoint_dir=tmp_path), ready())
    assert result['status'] == 'FULL40_CANONICAL_PPO_EXECUTION_BLOCKED'
    assert 'cannot verify selected checkpoint metadata' in result['blockers'][-1]


@pytest.mark.parametrize('name', ['expected_calibrated_model_sha256', 'expected_calibrated_state_sha256'])
def test_stray_single_calibration_shas_conflict_with_dual_selection(name):
    args = full_args()
    setattr(args, name, 'e' * 64)
    result = module.attach_dual_readiness(args, ready())
    assert result['status'] == 'FULL40_CANONICAL_PPO_EXECUTION_BLOCKED'


def test_loaded_checkpoint_metadata_also_rejects_p013_without_dual():
    with pytest.raises(ValueError, match='complete dual arguments'):
        module.require_single_model_identity({'status': 'FC_P013_INDEPENDENT_FORCE_FNO_AERODYNAMIC_CHECKPOINT'})
