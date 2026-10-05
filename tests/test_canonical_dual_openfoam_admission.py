"""CPU-only admission fixtures: no policies trained and no CFD executed."""
import json

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from test_full40_canonical_openfoam_feedback import MODULE, fixture


class ObservationFixture(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (69,), dtype=np.float32)
    action_space = gym.spaces.Box(-0.75, 0.75, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(69, dtype=np.float32), {}

    def step(self, action):
        raise AssertionError('admission must never step an environment')


def save_vec(path, *, norm_obs=False, norm_reward=False):
    raw = DummyVecEnv([ObservationFixture])
    try:
        VecNormalize(raw, norm_obs=norm_obs, norm_reward=norm_reward).save(path)
    finally:
        raw.close()


@pytest.fixture
def dual(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'new_policy'
    (output / 'checkpoints').mkdir(parents=True)
    policy = output / 'checkpoints/ppo_final.zip'
    policy.write_bytes(value['policy'].read_bytes())
    value['policy'] = policy
    vec = output / 'vecnormalize.pkl'
    save_vec(vec)
    value['vecnormalize'] = vec
    audit = MODULE.load_json(value['audit'])
    readiness = MODULE.load_json(value['readiness'])
    binding = {
        'status': 'DUAL_CONTROL_IDENTITY_VERIFIED_NOT_CONTROL_SUCCESS',
        'canonical_endpoint_window_dynamic_gates_still_required': True,
        'policy_trained': False, 'real_cfd_control_validated': False,
        'checkpoint_sha256': audit['physicsnemo_checkpoint_sha256'],
        'normalization_sha256': MODULE.sha256(value['data'] / 'normalization.json'),
        **{key: 'b' * 64 for key in ('checkpoint_state_sha256', 'dual_manifest_sha256',
            'flow_model_sha256', 'flow_state_sha256', 'posteval_receipt_sha256', 'training_config_sha256')},
    }
    audit.update(dual_control_binding=binding, training_executed=True,
                 vecnormalize_sha256=MODULE.sha256(vec),
                 vecnormalize_contract='identity: norm_obs=false, norm_reward=false; preserves legacy PPO numerics')
    audit['iterations'][-1]['checkpoint'] = str(policy)
    readiness['dual_control_binding'] = binding
    value['audit'].write_text(json.dumps(audit))
    value['readiness'].write_text(json.dumps(readiness))
    return value


def validate(value, **overrides):
    kwargs = dict(policy=value['policy'], ppo_audit=value['audit'], ppo_readiness=value['readiness'],
                  fno_gate=value['gate'], data=value['data'], vecnormalize=value['vecnormalize'],
                  expected_vecnormalize_sha256=MODULE.sha256(value['vecnormalize']))
    kwargs.update(overrides)
    return MODULE.validate_policy_and_gates(**kwargs)


def test_complete_dual_identity_and_saved_identity_vec_are_bound(dual):
    result = validate(dual)
    assert result['dual_control_binding']['dual_manifest_sha256'] == 'b' * 64
    assert result['vecnormalize_host_path'] == str(dual['vecnormalize'].resolve())
    assert result['vecnormalize_sha256'] == MODULE.sha256(dual['vecnormalize'])


@pytest.mark.parametrize('side', ['audit', 'readiness'])
@pytest.mark.parametrize('change', ['missing', 'different_flow', 'missing_hash'])
def test_missing_or_conflicting_dual_binding_rejected(dual, side, change):
    document = MODULE.load_json(dual[side])
    if change == 'missing':
        del document['dual_control_binding']
    elif change == 'different_flow':
        document['dual_control_binding']['flow_model_sha256'] = 'c' * 64
    else:
        del document['dual_control_binding']['posteval_receipt_sha256']
    dual[side].write_text(json.dumps(document))
    with pytest.raises(ValueError, match='dual_control_binding'):
        validate(dual)


@pytest.mark.parametrize('kwargs', [dict(vecnormalize=None), dict(expected_vecnormalize_sha256=None),
                                   dict(expected_vecnormalize_sha256='0' * 64)])
def test_dual_requires_explicit_paired_artifact_and_hash(dual, kwargs):
    with pytest.raises(ValueError, match='VecNormalize'):
        validate(dual, **kwargs)


@pytest.mark.parametrize('flags', [dict(norm_obs=True), dict(norm_reward=True)])
def test_even_sha_bound_nonidentity_vec_is_rejected(dual, flags):
    save_vec(dual['vecnormalize'], **flags)
    audit = MODULE.load_json(dual['audit'])
    audit['vecnormalize_sha256'] = MODULE.sha256(dual['vecnormalize'])
    dual['audit'].write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='not canonical identity'):
        validate(dual)


def test_normalization_from_another_output_is_rejected(dual, tmp_path):
    other = tmp_path / 'historical_vecnormalize.pkl'
    other.write_bytes(dual['vecnormalize'].read_bytes())
    with pytest.raises(ValueError, match='host path'):
        validate(dual, vecnormalize=other)


def test_tampered_artifact_rejected_before_deserialization(dual, monkeypatch):
    expected = MODULE.sha256(dual['vecnormalize'])
    dual['vecnormalize'].write_bytes(b'changed')
    monkeypatch.setattr(MODULE.pickle, 'loads', lambda *a: pytest.fail('must check hash before pickle'))
    with pytest.raises(ValueError, match='SHA differs'):
        validate(dual, expected_vecnormalize_sha256=expected)


def test_existing_single_policy_path_unchanged(tmp_path, monkeypatch):
    value = fixture(tmp_path, monkeypatch)
    result = MODULE.validate_policy_and_gates(policy=value['policy'], ppo_audit=value['audit'],
        ppo_readiness=value['readiness'], fno_gate=value['gate'], data=value['data'])
    assert result == value['lineage']


def test_missing_both_bindings_but_dual_model_marker_is_rejected(dual):
    for key in ('audit', 'readiness'):
        document = MODULE.load_json(dual[key])
        del document['dual_control_binding']
        if key == 'audit':
            document['calibrated_checkpoint_identity'] = {'dual_model_system': True}
        dual[key].write_text(json.dumps(document))
    with pytest.raises(ValueError, match='dual_control_binding'):
        validate(dual)


@pytest.mark.parametrize('change', ['missing_hash', 'different_model', 'different_normalization'])
def test_agreeing_but_invalid_dual_bindings_are_rejected(dual, change):
    for side in ('audit', 'readiness'):
        document = MODULE.load_json(dual[side])
        binding = document['dual_control_binding']
        if change == 'missing_hash':
            del binding['flow_state_sha256']
        else:
            key = 'checkpoint_sha256' if change == 'different_model' else 'normalization_sha256'
            binding[key] = 'c' * 64
        dual[side].write_text(json.dumps(document))
    with pytest.raises(ValueError, match='dual control identity|dual policy model'):
        validate(dual)


def test_dual_preflight_requires_new_lineage_and_explicit_artifact(dual, tmp_path):
    arguments = dict(policy=dual['policy'], ppo_audit=dual['audit'], ppo_readiness=dual['readiness'],
        fno_gate=dual['gate'], data=dual['data'], predeclaration=dual['predeclaration'],
        predeclaration_sha256=MODULE.sha256(dual['predeclaration']), output=tmp_path / 'result',
        resources=dual['resources'])
    blocked = MODULE.build_preflight(**arguments)
    assert blocked['status'] == 'CANONICAL_REAL_OPENFOAM_FEEDBACK_BLOCKED'
    assert any('requires VecNormalize' in item for item in blocked['blockers'])
    declaration = MODULE.load_json(dual['predeclaration'])
    declaration['lineage_sha256'] = validate(dual)
    dual['predeclaration'].write_text(json.dumps(declaration))
    arguments.update(predeclaration_sha256=MODULE.sha256(dual['predeclaration']),
        vecnormalize=dual['vecnormalize'], expected_vecnormalize_sha256=MODULE.sha256(dual['vecnormalize']))
    result = MODULE.build_preflight(**arguments)
    assert result['status'] == 'CANONICAL_REAL_OPENFOAM_FEEDBACK_READY'
    assert result['lineage']['vecnormalize_host_path'] == str(dual['vecnormalize'])
