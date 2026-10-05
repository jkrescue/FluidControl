import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fcp013_validator_fixture', ROOT / 'scripts/validate_fcp008_posteval.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def lineage():
    return {'dual_manifest_sha256': 'a' * 64, 'flow_model_sha256': 'b' * 64,
            'flow_state_sha256': 'c' * 64, 'checkpoint_sha256': 'd' * 64,
            'checkpoint_state_sha256': 'e' * 64}


def test_dual_profile_is_distinct_and_legacy_profile_restores():
    v.configure_profile('p013')
    assert v.DUAL_PROFILE and v.TRAINED_PROFILE
    assert v.CANDIDATE_KIND == 'fcp013_independent_force_dual_fno'
    assert v.calibrated_kwargs(lineage()) == {}
    assert len(v.dual_identity_fields(lineage())) == 3
    v.configure_profile('p008')
    assert not v.DUAL_PROFILE and not v.TRAINED_PROFILE
    assert v.dual_identity_fields({}) == {}
    assert v.calibrated_kwargs(lineage())['allow_calibrated_epoch_zero']


def report(window=False):
    ident = lineage()
    if window:
        return {'dual_fno_manifest_sha256': ident['dual_manifest_sha256'],
                'flow_model_sha256': ident['flow_model_sha256'],
                'flow_state_sha256': ident['flow_state_sha256'],
                'model_sha256': ident['checkpoint_sha256'],
                'aerodynamic_state_sha256': ident['checkpoint_state_sha256']}
    return {'checkpoint_metadata': {'dual_fno': True,
            'manifest_sha256': ident['dual_manifest_sha256'],
            'flow_model_sha256': ident['flow_model_sha256'],
            'flow_state_sha256': ident['flow_state_sha256'],
            'aerodynamic_model_sha256': ident['checkpoint_sha256'],
            'aerodynamic_state_sha256': ident['checkpoint_state_sha256']}}


@pytest.mark.parametrize('window', [False, True])
def test_each_dual_component_report_identity_is_required(window):
    v.configure_profile('p013')
    value = report(window)
    v.validate_dual_report(value, lineage(), force_window=window)
    target = value if window else value['checkpoint_metadata']
    for key in list(target):
        broken = json.loads(json.dumps(value))
        (broken if window else broken['checkpoint_metadata'])[key] = 'tampered'
        with pytest.raises(ValueError):
            v.validate_dual_report(broken, lineage(), force_window=window)


def test_missing_dual_lineage_identity_rejected():
    v.configure_profile('p013')
    for key in ('dual_manifest_sha256', 'flow_model_sha256', 'flow_state_sha256'):
        value = lineage()
        del value[key]
        with pytest.raises(ValueError):
            v.dual_identity_fields(value)


def test_runner_preserves_protocol_and_freezes_all_dual_dependencies():
    source = (ROOT / 'scripts/run_fcp008_posteval_spark.sh').read_text()
    for token in ('src/fluid_control/dual_fno.py', 'src/fluid_control/calibrated_checkpoint.py',
                  'scripts/audit_fcp011_candidate.py', 'scripts/train_fcp013_independent_force_fno.py',
                  'scripts/evaluate_fcp013_fixed_train_windows.py', 'training_config.yaml'):
        assert token in source
    assert '--segment-stride 25 --evaluation-batch-size 4' in source
    assert '--segment-stride 1 --evaluation-batch-size 8' in source
    assert source.count('"${dual_args[@]}"') == 3
    assert '--dual-training-config /workspace/training_config.yaml' in source
    assert 'container_checkpoint=/workspace/dual/aerodynamic' in source
    assert 'numerical_commit="7216214b545fbbd50b2fb5ed866f231039b06b18"' in source
    for name in ('run_fcp008_posteval_spark.sh', 'run_fcp013_posteval_spark.sh'):
        subprocess.run(['bash', '-n', str(ROOT / 'scripts' / name)], check=True)


def test_overlay_receipt_names_and_hashes_are_mandatory(tmp_path):
    v.configure_profile('p013')
    root = tmp_path / 'chain'
    source = root / 'numerical_source'
    names = ('src/fluid_control/dual_fno.py', 'src/fluid_control/calibrated_checkpoint.py',
             'scripts/evaluate_tandem_fno.py', 'scripts/diagnose_fno_force_window.py')
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('software fixture only')
    receipt = root / 'receipt.json'
    value = {'status': v.CHAIN_STATUS, 'git_commit': 'a' * 40, 'git_tree': 'b' * 40,
             'numerical_source_commit': '7216214b545fbbd50b2fb5ed866f231039b06b18',
             'numerical_source_tree': 'c' * 40, 'overlay_source_commit': 'a' * 40,
             'sha256': {str(path.relative_to(root)): v.sha256(path) for path in root.rglob('*') if path.is_file()},
             'numerical_source_overlays': {name: v.sha256(source / name) for name in names}}
    receipt.write_text(json.dumps(value))
    v.validate_chain_receipt(receipt, source)
    value['numerical_source_overlays'][names[0]] = '0' * 64
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='overlay identities'):
        v.validate_chain_receipt(receipt, source)
    value['numerical_source_overlays'][names[0]] = v.sha256(source / names[0])
    value['numerical_source_commit'] = 'f' * 40
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='numerical source lineage'):
        v.validate_chain_receipt(receipt, source)
