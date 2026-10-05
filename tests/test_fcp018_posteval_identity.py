import pytest
import validate_fcp008_posteval as v


@pytest.fixture(autouse=True)
def reset_profile():
    yield
    v.configure_profile('p008')


def fields():
    return dict(training_experiment='FC-P018',optimizer_steps=171,training_windows=1368,
        accumulation_windows=8,actual_learning_rate=1.5625e-7,training_protocol_file='training_protocol.json',
        training_protocol_sha256='310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d')


def test_explicit_p018_profile_and_unchanged_protocol():
    protocol=list(v.PROTOCOL);precision=dict(v.PRECISION)
    v.configure_profile('p018');v.validate_training_experiment(fields())
    assert v.DUAL_PROFILE and v.REDUCED_RATE_PROFILE and v.ACCUMULATION_PROFILE
    assert v.CANDIDATE_KIND=='fcp018_reduced_rate_dual_fno'
    assert v.CHAIN_STATUS=='FC_P018_IMMUTABLE_POSTEVAL_CHAIN_STAGED'
    assert v.PROTOCOL==protocol and v.PRECISION==precision


@pytest.mark.parametrize('key,value',[('training_experiment','FC-P015'),('actual_learning_rate',1e-5),
    ('training_protocol_sha256','bad'),('training_protocol_file','other.json'),('optimizer_steps',1368)])
def test_reject_wrong_effective_training_identity(key,value):
    v.configure_profile('p018');data=fields();data[key]=value
    with pytest.raises(ValueError):v.validate_training_experiment(data)


def test_legacy_profile_state_reset():
    v.configure_profile('p018');v.configure_profile('p015')
    assert not v.REDUCED_RATE_PROFILE
    v.validate_training_experiment(dict(training_experiment='FC-P015',optimizer_steps=171,
        training_windows=1368,accumulation_windows=8))
    with pytest.raises(ValueError):v.validate_training_experiment(fields())
    v.configure_profile('p013')
    assert not v.ACCUMULATION_PROFILE
    v.validate_training_experiment({'optimizer_steps':1368})
