"""Temporary P013 path views: synthetic CPU evidence, no scientific admission."""
import json
from types import SimpleNamespace

import pytest

from test_full40_canonical_ppo_dry_run import MODULE
import fluid_control.dual_control_contract as contract


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / 'posteval'
    (root / 'validation10').mkdir(parents=True)
    config = tmp_path / 'evaluation.yaml'
    config.write_text('official evaluation configuration fixture')
    binding = {key: str(index) * 64 for index, key in enumerate((
        'dual_manifest_sha256', 'flow_model_sha256', 'flow_state_sha256',
        'checkpoint_sha256', 'checkpoint_state_sha256', 'posteval_receipt_sha256'))}
    report = {
        'checkpoint_dir': '/workspace/dual/aerodynamic', 'evaluation_data': '/workspace/devdata',
        'normalization_data': '/workspace/devdata', 'checkpoint_epoch': 1,
        'checkpoint_metadata': {'dual_fno': True, 'manifest_sha256': binding['dual_manifest_sha256'],
            'flow_model_sha256': binding['flow_model_sha256'], 'flow_state_sha256': binding['flow_state_sha256'],
            'aerodynamic_model_sha256': binding['checkpoint_sha256'],
            'aerodynamic_state_sha256': binding['checkpoint_state_sha256']},
        'cases': [{'case': 'synthetic', 'horizons': {'100': {'metric': 0.123456789}}}],
    }
    report_path = root / 'validation10/evaluation.json'
    report_path.write_text(json.dumps(report))
    segments = root / 'validation10/segments.json'
    segments.write_text(json.dumps({'synthetic_segments': [1, 2, 3]}))
    gate = {'checkpoint_sha256': binding['checkpoint_sha256'],
            'checkpoint_dir': report['checkpoint_dir'], 'report_sha256': MODULE.sha256(report_path),
            'model_config_sha256': MODULE.sha256(config),
            'status': MODULE.VALIDATION_GATE_STATUS, 'joint_terminal_readiness': True,
            'h100_force_gate': {'beats_persistence': True, 'metric': 0.123456789}}
    gate_path = root / 'validation10/endpoint_gate.json'
    gate_path.write_text(json.dumps(gate))
    receipt = root / 'receipt.json'
    paths = [report_path, segments, gate_path]
    receipt.write_text(json.dumps({'sha256': {str(p.relative_to(root)): MODULE.sha256(p) for p in paths}}))
    calls = []
    monkeypatch.setattr(contract, 'verify_dual_control_binding', lambda **kwargs: (calls.append(kwargs) or binding))
    def audit(view_path, actual_segments, predecl, checkpoint, data, actual_config, image):
        view = MODULE.read_json(view_path)
        originals = dict(view)
        for key in ('checkpoint_dir', 'evaluation_data', 'normalization_data'):
            originals[key] = report[key]
        assert originals == report
        assert view['checkpoint_dir'] == str(checkpoint.resolve())
        assert view['evaluation_data'] == view['normalization_data'] == str(data.resolve())
        assert actual_segments == segments and actual_config == config
        assert calls, 'identity must be verified before numerical audit'
        calls.append(view_path)
        return {**gate, 'checkpoint_dir': str(checkpoint), 'report_sha256': MODULE.sha256(view_path)}
    args = dict(dual_options=dict(zip(MODULE.DUAL_OPTION_NAMES, [tmp_path / 'manifest.json', '0' * 64,
        tmp_path / 'training.yaml', receipt, '5' * 64], strict=True)), gate_module=SimpleNamespace(audit=audit),
        validation_gate=gate_path, validation_report=report_path, validation_segments=segments,
        predeclaration=tmp_path / 'predecl.json', checkpoint_dir=tmp_path / 'candidate/aerodynamic',
        data=tmp_path / 'full40', config=config, image_id='fixture')
    return args, gate, calls


def test_only_three_paths_change_and_original_gate_and_sources_survive(evidence):
    args, gate, calls = evidence
    before = {name: args[name].read_bytes() for name in ('validation_gate', 'validation_report', 'validation_segments')}
    result, mapping = MODULE.recompute_p013_endpoint_view(**args)
    assert result == gate
    assert mapping['numerical_evidence_changed'] is False
    assert mapping['original_paths']['checkpoint_dir'] == '/workspace/dual/aerodynamic'
    assert mapping['runtime_paths']['checkpoint_dir'] == str(args['checkpoint_dir'])
    assert not calls[-1].exists(), 'temporary path view must be removed'
    assert all(args[name].read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('name', ['validation_gate', 'validation_report', 'validation_segments'])
def test_source_hash_tampering_rejected_before_temp_view(evidence, name, monkeypatch):
    args, _, _ = evidence
    args[name].write_text(args[name].read_text() + ' ')
    monkeypatch.setattr(MODULE.tempfile, 'TemporaryDirectory', lambda **kwargs: pytest.fail('no rewrite before SHA validation'))
    with pytest.raises(ValueError, match='source SHA'):
        MODULE.recompute_p013_endpoint_view(**args)


@pytest.mark.parametrize('fault', ['alias', 'dual_metadata', 'outside_receipt', 'config', 'partial_options'])
def test_identity_or_alias_failures_reject_before_temp_view(evidence, fault, monkeypatch):
    args, _, _ = evidence
    if fault in ('alias', 'dual_metadata'):
        report = MODULE.read_json(args['validation_report'])
        if fault == 'alias':
            report['evaluation_data'] = '/arbitrary/path'
        else:
            report['checkpoint_metadata']['flow_model_sha256'] = '9' * 64
        args['validation_report'].write_text(json.dumps(report))
        receipt = args['dual_options']['dual_posteval_receipt']
        value = MODULE.read_json(receipt)
        value['sha256']['validation10/evaluation.json'] = MODULE.sha256(args['validation_report'])
        receipt.write_text(json.dumps(value))
    elif fault == 'outside_receipt':
        other = args['validation_segments'].parent.parent / 'copied_segments.json'
        other.write_bytes(args['validation_segments'].read_bytes())
        args['validation_segments'] = other
    elif fault == 'config':
        args['config'].write_text('different')
    else:
        args['dual_options'].pop('dual_training_config')
    monkeypatch.setattr(MODULE.tempfile, 'TemporaryDirectory', lambda **kwargs: pytest.fail('no rewrite before identity validation'))
    with pytest.raises(ValueError):
        MODULE.recompute_p013_endpoint_view(**args)


def test_scientific_failure_is_returned_unchanged(evidence):
    args, gate, _ = evidence
    gate.update(status='FULL40_VALIDATION_SURROGATE_READINESS_FAIL', joint_terminal_readiness=False)
    gate['h100_force_gate']['beats_persistence'] = False
    result, _ = MODULE.recompute_p013_endpoint_view(**args)
    assert result['status'].endswith('FAIL')
    assert result['joint_terminal_readiness'] is False
    assert result['h100_force_gate'] == gate['h100_force_gate']


def test_dual_verifier_failure_prevents_any_view(evidence, monkeypatch):
    args, _, _ = evidence
    def rejected(**kwargs):
        raise ValueError('receipt identity rejected')
    monkeypatch.setattr(contract, 'verify_dual_control_binding', rejected)
    monkeypatch.setattr(MODULE.tempfile, 'TemporaryDirectory', lambda **kwargs: pytest.fail('identity rejected'))
    with pytest.raises(ValueError, match='receipt identity rejected'):
        MODULE.recompute_p013_endpoint_view(**args)


@pytest.mark.parametrize('fault', [None, 'promotion', 'numerical_change'])
def test_preflight_requires_promotion_and_exact_original_gate(evidence, monkeypatch, tmp_path, fault):
    args, gate, _ = evidence
    data = args['data']
    data.mkdir()
    (data / 'manifest.json').write_text(json.dumps({'trajectory_counts': MODULE.EXPECTED_COUNTS}))
    for split, cases in (('train', MODULE.TRAIN_CASES), ('validation', MODULE.VALIDATION_CASES)):
        (data / split).mkdir()
        for case in cases:
            (data / split / f'{case}.h5').touch()
    for name in ('window', 'dynamic', 'baselines'):
        (tmp_path / f'{name}.json').write_text('{}')
    gate.update(data_manifest_sha256='6' * 64, normalization_sha256='7' * 64)
    args['validation_gate'].write_text(json.dumps(gate))
    calls = []
    def promotion(*a, **kw):
        if fault == 'promotion':
            raise ValueError('promotion missing')
        return {'full40_manifest_sha256': '6' * 64, 'full40_normalization_sha256': '7' * 64}
    def view(**kwargs):
        calls.append(kwargs)
        value = json.loads(json.dumps(gate))
        if fault == 'numerical_change':
            value['h100_force_gate']['metric'] += 1e-12
        return value, {'fixture_path_view': True}
    monkeypatch.setattr(MODULE, 'validate_full40_action_contract', lambda *a: {})
    monkeypatch.setattr(MODULE, 'validate_dev30_promotion_receipt', promotion)
    monkeypatch.setattr(MODULE, 'recompute_p013_endpoint_view', view)
    monkeypatch.setattr(MODULE, 'validate_evidence_gate', lambda *a, **kw: {})
    monkeypatch.setattr(MODULE, 'validate_baselines', lambda *a: {})
    result = MODULE.preflight(**{k: v for k, v in args.items() if k != 'gate_module'},
        dev30_data=tmp_path / 'dev30', promotion_receipt=tmp_path / 'promotion.json',
        window_gate=tmp_path / 'window.json', dynamic_gate=tmp_path / 'dynamic.json',
        baselines=tmp_path / 'baselines.json', runtime_image_id=MODULE.HYDROGYM_RUNTIME_IMAGE_ID,
        episode_steps=100)
    if fault is None:
        assert result['status'] == 'FULL40_CANONICAL_PPO_EXECUTION_READY'
        assert result['endpoint_path_view'] == {'fixture_path_view': True}
    else:
        assert result['status'] == 'FULL40_CANONICAL_PPO_EXECUTION_BLOCKED'
    if fault == 'promotion':
        assert not calls
    if fault == 'numerical_change':
        assert any('stored validation gate differs' in message for message in result['blockers'])
