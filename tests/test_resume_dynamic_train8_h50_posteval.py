from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/resume_dynamic_train8_h50_posteval.sh"


def test_resume_is_evaluation_only_and_uses_canonical_devdata_mount():
    text = RUNNER.read_text()
    assert "train_tandem_fno_rollout.py" not in text
    assert "src=$base,dst=/workspace/devdata,readonly" in text
    assert "--data /workspace/devdata --normalization-data /workspace/devdata" in text
    assert "--data /workspace/dynamic --normalization-data /workspace/devdata" in text
    assert "tandem_cylinders_matched_start_full40_dev30_v1" in text
    assert "tandem_cylinders_full40_dynamic_validation_v1" in text


def test_resume_is_fail_closed_and_never_mounts_frozen_data():
    text = RUNNER.read_text()
    assert "EXECUTE_REVIEWED_DYNAMIC_H50_POSTEVAL_RESUME" in text
    assert "refusing existing resume output" in text
    assert "frozen_test_accessed':False" in text
    assert "frozen_test" not in "\n".join(
        line for line in text.splitlines() if line.lstrip().startswith("--mount")
    )
    assert "DYNAMIC_FNO_POSTEVAL_RESUME_COMPLETE" in text
    assert "'model_sha256':model_sha" in text

