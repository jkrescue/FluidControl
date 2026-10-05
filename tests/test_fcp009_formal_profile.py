import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_validator():
    path = ROOT / "scripts/validate_fcp008_posteval.py"
    spec = importlib.util.spec_from_file_location("shared_posteval_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identities(module):
    return (
        module.CANDIDATE_KIND,
        module.LINEAGE_STATUS,
        module.STEP_STATUS,
        module.COMPLETE_STATUS,
        module.CALIBRATED_KIND,
        module.DIAGNOSTIC_KIND,
        module.PRECISION_STATUS,
        module.CHAIN_STATUS,
    )


def test_profile_switch_is_complete_and_reversible():
    module = load_validator()
    original = identities(module)
    module.configure_profile("p009")
    assert identities(module) == (
        "joint_h1_free_ar_force_row_recalibration",
        "FC_P009_CANDIDATE_LINEAGE_PASS",
        "FC_P009_POSTEVAL_STEP_COMPLETE",
        "FC_P009_POSTEVAL_COMPLETE",
        "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE",
        "fc_p009_joint_force_row_calibrated_epoch0",
        "FC_P009_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
        "FC_P009_IMMUTABLE_POSTEVAL_CHAIN_STAGED",
    )
    module.configure_profile("p008")
    assert identities(module) == original


def test_p008_historical_signature_omits_new_kind():
    module = load_validator()
    module.configure_profile("p008")
    kwargs = module.calibrated_kwargs({
        "checkpoint_sha256": "a" * 64,
        "checkpoint_state_sha256": "b" * 64,
    })
    assert "expected_calibrated_kind" not in kwargs
    module.configure_profile("p009")
    assert module.calibrated_kwargs({
        "checkpoint_sha256": "a" * 64,
        "checkpoint_state_sha256": "b" * 64,
    })["expected_calibrated_kind"] == "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"


def test_runner_keeps_kind_flag_profile_conditional():
    text = (ROOT / "scripts/run_fcp008_posteval_spark.sh").read_text()
    assert 'if [[ "$profile" == p009 ]]; then kind_args=' in text
    assert 'numerical_commit="ab7b9fe5e442c044319d3d686cb1e1d2fc0b4a82"' in text
    assert 'numerical_commit="7216214b545fbbd50b2fb5ed866f231039b06b18"' in text
    wrapper = (ROOT / "scripts/run_fcp009_posteval_spark.sh").read_text()
    assert "FCP_POSTEVAL_PROFILE=p009" in wrapper
    assert "run_fcp008_posteval_spark.sh" in wrapper
