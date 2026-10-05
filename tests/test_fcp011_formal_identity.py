import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).parents[1]


def load_module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
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
        module.TRAINED_PROFILE,
    )


def test_positive_epoch_profiles_are_exact_and_reversible():
    module = load_module("scripts/validate_fcp008_posteval.py", "validator_profiles")
    original = identities(module)
    module.configure_profile("p011_head_only")
    assert identities(module) == (
        "fcp011_head_only_epoch1",
        "FC_P011_HEAD_ONLY_CANDIDATE_LINEAGE_PASS",
        "FC_P011_HEAD_ONLY_POSTEVAL_STEP_COMPLETE",
        "FC_P011_HEAD_ONLY_POSTEVAL_COMPLETE",
        None,
        "dev30_h20_development",
        "FC_P011_HEAD_ONLY_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
        "FC_P011_HEAD_ONLY_IMMUTABLE_POSTEVAL_CHAIN_STAGED",
        True,
    )
    assert module.calibrated_kwargs({}) == {}
    module.configure_profile("p011_decoder_tail")
    assert module.TRAINED_PROFILE is True
    module.configure_profile("p008")
    assert identities(module) == original


def test_shared_runner_uses_positive_epoch_without_calibration_flags():
    text = (ROOT / "scripts/run_fcp008_posteval_spark.sh").read_text()
    assert 'p011_head_only|p011_decoder_tail)' in text
    assert 'checkpoint_epoch=1' in text
    assert 'checkpoint_relative_expected="final"' in text
    assert 'checkpoint_args=()' in text
    assert '--checkpoint-epoch "$checkpoint_epoch"' in text
    assert '"$candidate_kind" "$checkpoint_epoch"' in text
    wrapper = (ROOT / "scripts/run_fcp011_posteval_spark.sh").read_text()
    assert 'FCP_POSTEVAL_PROFILE="p011_${scope}"' in wrapper


def write_model(path: Path, state: dict) -> None:
    torch = pytest.importorskip("torch")
    import io

    stream = io.BytesIO()
    torch.save(state, stream)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.pt", stream.getvalue())
        archive.writestr("args.json", "{}\n")
        archive.writestr("metadata.json", "{}\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_fixture(tmp_path: Path, monkeypatch, scope: str = "head_only"):
    torch = pytest.importorskip("torch")
    module = load_module("scripts/audit_fcp011_candidate.py", f"audit_{scope}")
    repo = tmp_path
    candidate = repo / "artifacts" / module.PROFILES[scope]["root"]
    final = candidate / "final"
    final.mkdir(parents=True)
    parent_dir = repo / "artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate"
    parent_dir.mkdir(parents=True)
    parent = {
        module.FINAL_WEIGHT: torch.zeros((7, 2)),
        module.FINAL_BIAS: torch.zeros(7),
        module.HIDDEN_WEIGHT: torch.zeros((2, 2)),
        module.HIDDEN_BIAS: torch.zeros(2),
        "encoder.weight": torch.ones(1),
    }
    changed = {key: value.clone() for key, value in parent.items()}
    changed[module.FINAL_WEIGHT][6] = 1
    changed[module.FINAL_BIAS][6] = 1
    if scope == "decoder_tail":
        changed[module.HIDDEN_WEIGHT][0, 0] = 1
        changed[module.HIDDEN_BIAS][0] = 1
    parent_model = parent_dir / "FNO.0.0.mdlus"
    model = final / "FNO.0.1.mdlus"
    write_model(parent_model, parent)
    write_model(model, changed)
    monkeypatch.setattr(module, "PARENT_MODEL_SHA", sha(parent_model))
    state = final / "checkpoint.0.1.pt"
    torch.save(
        {
            "metadata": {
                "status": module.CHECKPOINT_STATUS,
                "scope": scope,
                "optimizer_steps": 1368,
                "parent_model_sha256": module.PARENT_MODEL_SHA,
                "parent_state_sha256": module.PARENT_STATE_SHA,
                "selection_performed": False,
                "validation_accessed": False,
                "frozen_test_accessed": False,
                "ppo_executed": False,
            }
        },
        state,
    )
    launch = {
        "status": "FC_P011_TRAINING_LAUNCH_VERIFIED",
        "scope": scope,
        "trainer_sha256": module.TRAINER_SHA,
        "approval_sha256": module.TRAINING_APPROVAL_SHA,
        "config_sha256": module.CONFIG_SHA,
        "parent_model_sha256": module.PARENT_MODEL_SHA,
        "parent_state_sha256": module.PARENT_STATE_SHA,
        "image_id": module.IMAGE_ID,
        "expected_updates": 1368,
        "resource_probe": False,
        "validation_or_frozen_mounted": False,
    }
    (candidate / "launch_receipt.json").write_text(json.dumps(launch))
    identities = [
        {
            "case": f"case-{index}",
            "start": index,
            "dataset_index": 0 if index < 720 else (1 if index < 1128 else 2),
            "split": "train",
            "rollout_steps": 100,
        }
        for index in range(1368)
    ]
    records = [
        {
            "step": index + 1,
            "identity": identity,
            "total_loss": 1.0,
            "decoder_hidden_gradient_norm": 0.0 if scope == "head_only" else 1.0,
        }
        for index, identity in enumerate(identities)
    ]
    expected_changed = sorted(module.PROFILES[scope]["changed"])
    expected_scope = {
        "optimizer_parameter_names": expected_changed,
        "optimizer_tensor_elements": 17415 if scope == "decoder_tail" else 903,
        "effective_trainable_coefficients": 16641 if scope == "decoder_tail" else 129,
    }
    result = {
        "status": module.RESULT_STATUS,
        "scope": scope,
        "parent_model_sha256": module.PARENT_MODEL_SHA,
        "parent_state_sha256": module.PARENT_STATE_SHA,
        "model_state_before_sha256": "before",
        "model_state_after_sha256": "after",
        "fresh_reload_tensor_sha256": "after",
        "train_order_sha256": module.ORDER_SHA,
        "expected_train_order_sha256": module.ORDER_SHA,
        "family_window_counts": {"base20": 720, "train8": 408, "train16": 240},
        "train_identities": identities,
        "training_records": records,
        "trainable_scope": expected_scope,
        "changed_tensor_names": expected_changed,
        "optimizer_steps": 1368,
        "batch_size": 1,
        "loss_contract": {
            "field_weight": 1.0,
            "force_weight": 0.2,
            "equal_four_share": 0.5,
            "rear_cl_share": 0.5,
            "effective_force_channel_weights": [0.125, 0.125, 0.125, 0.625],
            "rollout_steps": 100,
            "step_weights": "uniform",
            "teacher_forcing": 0.0,
        },
        "diagnostic_steps_predeclared": [0, 32, 128, 512, 1368],
        "train_only_diagnostics": {str(step): {"finite": True} for step in (0, 32, 128, 512, 1368)},
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "terminal_model_sha256": sha(model),
        "terminal_state_sha256": sha(state),
        "config_sha256": module.CONFIG_SHA,
        "input_sha256": module.INPUT_SHA,
        "implementation_sha256": module.TRAINER_SHA,
    }
    (candidate / "result.json").write_text(json.dumps(result))
    completion = {
        "status": module.COMPLETION_STATUS,
        "scope": scope,
        "training_exit_code": 0,
        "trainer_sha256": module.TRAINER_SHA,
        "approval_sha256": module.TRAINING_APPROVAL_SHA,
        "config_sha256": module.CONFIG_SHA,
        "parent_model_sha256": module.PARENT_MODEL_SHA,
        "parent_state_sha256": module.PARENT_STATE_SHA,
        "optimizer_steps": 1368,
        "train_order_sha256": module.ORDER_SHA,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "sha256": {
            "launch_receipt.json": sha(candidate / "launch_receipt.json"),
            "result.json": sha(candidate / "result.json"),
            "final/FNO.0.1.mdlus": sha(model),
            "final/checkpoint.0.1.pt": sha(state),
        },
    }
    (candidate / "completion_receipt.json").write_text(json.dumps(completion))
    return module, repo, candidate


@pytest.mark.parametrize("scope", ["head_only", "decoder_tail"])
def test_auditor_accepts_only_exact_positive_epoch_scope(tmp_path, monkeypatch, scope):
    module, repo, candidate = candidate_fixture(tmp_path, monkeypatch, scope)
    value = module.validate_candidate(repo, candidate, scope)
    assert value["checkpoint_epoch"] == 1
    assert value["training_performed"] is True
    assert value["training_scope"] == scope


def test_auditor_rejects_scope_and_metadata_tamper(tmp_path, monkeypatch):
    module, repo, candidate = candidate_fixture(tmp_path, monkeypatch)
    result = json.loads((candidate / "result.json").read_text())
    result["changed_tensor_names"].append(module.HIDDEN_WEIGHT)
    (candidate / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        module.validate_candidate(repo, candidate, "head_only")
