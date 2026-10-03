import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/audit_control_gap_v4_action_coverage.py"
)
SPEC = importlib.util.spec_from_file_location("action_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_omega(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("omega", data=np.asarray(values).reshape(-1, 1))


def make_profile(tmp_path):
    root = tmp_path / "profile"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "control_gap_v4",
                "frames_per_trajectory": 4,
                "trajectory_counts": {"train": 2, "validation": 1, "test": 99},
            }
        )
    )
    write_omega(root / "train/a.h5", [0.0, 0.375, 0.75, 0.8])
    write_omega(root / "train/b.h5", [-3.0, -2.0, 2.0, 3.0])
    write_omega(root / "validation/c.h5", [-0.75, 0.75, 0.0, 0.1])
    test_dir = root / "test"
    test_dir.mkdir()
    (test_dir / "must_not_be_opened.h5").write_text("not an HDF5 file")
    return root


def test_report_reads_train_and_validation_omega_but_not_test(tmp_path):
    report = MODULE.build_report(make_profile(tmp_path))
    assert report["read_contract"]["flow_fields_read"] is False
    assert report["read_contract"]["audited_splits"] == ["train", "validation"]
    train = report["splits"]["train"]
    assert train["frame_count"] == 8
    assert train["target_range_abs_omega_le_0p75"]["frame_count"] == 3
    assert train["frame_fraction_abs_omega_gt_2"] == pytest.approx(0.25)
    validation = report["splits"]["validation"]
    assert validation["target_range_abs_omega_le_0p75"][
        "entirely_target_range_trajectory_count"
    ] == 1


def test_target_boundary_is_inclusive_and_full_trajectory_is_distinct():
    omega = np.asarray([-0.75, 0.0, 0.75, 0.751])
    row = MODULE.trajectory_summary(Path("x.h5"), omega)
    assert row["target_range_frame_count"] == 3
    assert row["entire_trajectory_within_target_range"] is False
    assert row["majority_frames_within_target_range"] is True


def test_manifest_count_mismatch_fails_closed(tmp_path):
    root = make_profile(tmp_path)
    (root / "train/b.h5").unlink()
    with pytest.raises(ValueError, match="manifest/file count mismatch"):
        MODULE.build_report(root)


def test_nonfinite_omega_fails_closed(tmp_path):
    path = tmp_path / "bad.h5"
    write_omega(path, [0.0, 1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="invalid omega"):
        MODULE.read_omega_only(path, 4)


def test_atomic_writer_refuses_overwrite(tmp_path):
    output = tmp_path / "result.json"
    MODULE.write_json_exclusive_atomic(output, {"value": 1})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MODULE.write_json_exclusive_atomic(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}
