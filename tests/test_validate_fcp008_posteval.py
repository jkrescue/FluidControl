from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "validate_fcp008_posteval.py"
)
SPEC = importlib.util.spec_from_file_location("validate_fcp008_posteval", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def candidate_lineage(tmp_path: Path) -> tuple[Path, dict]:
    candidate = tmp_path / "candidate_root"
    checkpoint = candidate / "candidate"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "FNO.0.0.mdlus"
    state = checkpoint / "checkpoint.0.0.pt"
    model.write_bytes(b"official-model")
    state.write_bytes(b"official-state")
    source = candidate / "source_snapshot/scripts"
    source.mkdir(parents=True)
    legacy = source / "validate_fcp003c_posteval.py"
    legacy.write_text(
        """
def _check(candidate):
    assert (candidate / 'best/FNO.0.0.mdlus').read_bytes() == b'official-model'
    assert (candidate / 'source_snapshot/scripts/validate_fcp003c_posteval.py').is_file()
def validate_validation(repo, candidate, out, checkpoint): _check(candidate)
def validate_dynamic(repo, candidate, out, checkpoint): _check(candidate)
def validate_force(candidate, out, checkpoint): _check(candidate)
""",
        encoding="utf-8",
    )
    lineage = {
        "status": validator.LINEAGE_STATUS,
        "candidate_kind": validator.CANDIDATE_KIND,
        "official_image_id": validator.IMAGE_ID,
        "formal_protocol": validator.PROTOCOL,
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "training_performed": False,
        "validation_or_frozen_accessed": False,
        "ppo_auto_launch": False,
        "checkpoint_relative_directory": "candidate",
        "checkpoint_model_file": model.name,
        "checkpoint_state_file": state.name,
        "checkpoint_epoch": 0,
        "checkpoint_sha256": validator.sha256(model),
        "checkpoint_state_sha256": validator.sha256(state),
    }
    return candidate, lineage


def test_lineage_requires_exact_pair_epoch_and_nontraining_scope(tmp_path: Path) -> None:
    candidate, lineage = candidate_lineage(tmp_path)
    lineage_path = tmp_path / "lineage.json"
    write_json(lineage_path, lineage)
    assert validator.validate_lineage(candidate, lineage_path)[0] == lineage

    for key, bad in (
        ("checkpoint_epoch", 2),
        ("training_performed", True),
        ("validation_or_frozen_accessed", True),
    ):
        changed = {**lineage, key: bad}
        write_json(lineage_path, changed)
        with pytest.raises(ValueError):
            validator.validate_lineage(candidate, lineage_path)

    write_json(lineage_path, lineage)
    (candidate / "candidate/checkpoint.0.0.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="state differs"):
        validator.validate_lineage(candidate, lineage_path)


@pytest.mark.parametrize("step", ["validation10", "dynamic6", "force_window"])
def test_science_steps_delegate_to_reviewed_numerical_validator(
    tmp_path: Path, step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, lineage = candidate_lineage(tmp_path)
    source = candidate / "source_snapshot"
    out = tmp_path / "out"
    expected = {
        "validation_diagnostic": {"kind": "validation"},
        "endpoint_gate": {"kind": "endpoint"},
        "dynamic6": {"kind": "dynamic"},
        "development": {"kind": "development"},
    }
    write_json(out / "validation10/diagnostic.json", expected["validation_diagnostic"])
    write_json(out / "validation10/endpoint_gate.json", expected["endpoint_gate"])
    write_json(out / "dynamic6/diagnostic.json", expected["dynamic6"])
    write_json(out / "development_gate.json", expected["development"])

    class FakeModule:
        def __init__(self, value: dict):
            self.value = value

        def audit(self, *args, **kwargs):
            if self.value["kind"] in {"validation", "endpoint"}:
                assert kwargs == {
                    "allow_calibrated_epoch_zero": True,
                    "expected_calibrated_model_sha256": lineage[
                        "checkpoint_sha256"
                    ],
                    "expected_calibrated_state_sha256": lineage[
                        "checkpoint_state_sha256"
                    ],
                }
            return self.value

    def fake_module(path: Path, name: str):
        if "dev30" in path.name:
            return FakeModule(expected["validation_diagnostic"])
        if "full40_validation_gate" in path.name:
            return FakeModule(expected["endpoint_gate"])
        if "dynamic6" in path.name:
            return FakeModule(expected["dynamic6"])
        return FakeModule(expected["development"])

    monkeypatch.setattr(validator, "module", fake_module)
    validator.validate_science_step(
        tmp_path,
        candidate,
        out,
        lineage,
        step,
        source,
    )


def make_complete_output(tmp_path: Path) -> tuple[Path, Path, dict, Path]:
    candidate, lineage = candidate_lineage(tmp_path)
    out = tmp_path / "posteval"
    out.mkdir()
    lineage_path = out / "lineage.json"
    write_json(lineage_path, lineage)
    lineage_sha = validator.sha256(lineage_path)
    lineage = {
        **lineage,
        "posteval_chain_receipt_sha256": "c" * 64,
        "formal_evaluation_approval_sha256": "d" * 64,
    }
    write_json(
        out / "precision.json",
        {
            "status": "FC_P008_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
            "official_image_id": validator.IMAGE_ID,
            **validator.PRECISION,
        },
    )
    for names in validator.EXPECTED.values():
        for name in names:
            write_json(out / name, {"artifact": name})
    write_json(
        out / "development_gate.json",
        {
            "artifact": "development_gate.json",
            "status": "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL",
            "ppo_authorized": False,
            "frozen_test_accessed": False,
        },
    )
    identity = {
        "candidate_kind": validator.CANDIDATE_KIND,
        "checkpoint_epoch": lineage["checkpoint_epoch"],
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "checkpoint_state_sha256": lineage["checkpoint_state_sha256"],
        "lineage_sha256": lineage_sha,
        "posteval_chain_receipt_sha256": lineage["posteval_chain_receipt_sha256"],
        "formal_evaluation_approval_sha256": lineage[
            "formal_evaluation_approval_sha256"
        ],
        "precision_sha256": validator.sha256(out / "precision.json"),
    }
    for step, names in validator.EXPECTED.items():
        write_json(
            out / "step_receipts" / f"{step}.json",
            {
                "status": validator.STEP_STATUS,
                "step": step,
                **identity,
                "sha256": {name: validator.sha256(out / name) for name in names},
            },
        )
    files = {
        str(path.relative_to(out)): validator.sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    write_json(
        out / "receipt.json",
        {
            "status": validator.COMPLETE_STATUS,
            **identity,
            "official_image_id": validator.IMAGE_ID,
            "protocol": validator.PROTOCOL,
            "frozen_test_accessed": False,
            "ppo_auto_launched": False,
            "sha256": files,
        },
    )
    return candidate, out, lineage, lineage_path


def test_complete_receipt_binds_state_epoch_lineage_and_all_artifacts(
    tmp_path: Path,
) -> None:
    candidate, out, lineage, lineage_path = make_complete_output(tmp_path)
    validator.validate_complete(candidate, out, lineage_path, lineage)

    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    receipt["checkpoint_state_sha256"] = "0" * 64
    write_json(out / "receipt.json", receipt)
    with pytest.raises(ValueError, match="identity differs"):
        validator.validate_complete(candidate, out, lineage_path, lineage)


def test_step_receipt_rejects_tampered_science_artifact(tmp_path: Path) -> None:
    _, out, lineage, lineage_path = make_complete_output(tmp_path)
    lineage = {**lineage, "lineage_sha256": validator.sha256(lineage_path)}
    (out / "dynamic6/evaluation.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hashes differ"):
        validator.validate_step_receipt(
            out / "step_receipts/dynamic6.json", out, "dynamic6", lineage
        )


def test_precision_requires_historical_default_tf32_high(tmp_path: Path) -> None:
    path = tmp_path / "precision.json"
    write_json(
        path,
        {
            "status": "FC_P008_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
            "official_image_id": validator.IMAGE_ID,
            **validator.PRECISION,
        },
    )
    validator.validate_precision(path)
    value = json.loads(path.read_text())
    value["float32_matmul_precision"] = "highest"
    write_json(path, value)
    with pytest.raises(ValueError, match="precision differs"):
        validator.validate_precision(path)


def test_chain_receipt_binds_independent_numerical_source(tmp_path: Path) -> None:
    root = tmp_path / "chain"
    numerical = root / "numerical_source/scripts"
    numerical.mkdir(parents=True)
    (numerical / "validate_fcp003c_posteval.py").write_text("# reviewed\n")
    (root / "scripts").mkdir()
    (root / "scripts/run_fcp008_posteval_spark.sh").write_text("# reviewed\n")
    files = {
        str(path.relative_to(root)): validator.sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    receipt = root / "receipt.json"
    write_json(
        receipt,
        {
            "status": "FC_P008_IMMUTABLE_POSTEVAL_CHAIN_STAGED",
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "numerical_source_commit": "3" * 40,
            "numerical_source_tree": "4" * 40,
            "sha256": files,
        },
    )
    assert validator.validate_chain_receipt(receipt, root / "numerical_source")
    (numerical / "validate_fcp003c_posteval.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="files differ"):
        validator.validate_chain_receipt(receipt, root / "numerical_source")
