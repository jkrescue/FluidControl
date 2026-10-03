from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("dynamic6_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_accepts_explicit_intermediate_checkpoint_identity(tmp_path: Path) -> None:
    module = load_module()
    data = tmp_path / "data"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = {
        "trajectory_counts": {"train": 0, "validation": 6, "frozen_test": 0},
        "training_access": "FORBIDDEN",
    }
    write_json(data / "manifest.json", manifest)
    module.MANIFEST_SHA = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    model = checkpoint / "FNO.0.1.mdlus"
    model.write_bytes(b"fixed-epoch-one")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    cases = sorted(module.CASES)
    report = {
        "checkpoint_epoch": 1,
        "split": "validation",
        "action_mode": "observed",
        "force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "cases": [
            {
                "case": case,
                "horizons": {
                    horizon: {
                        "stable": True,
                        "failed_segments": 0,
                        "segments": 1,
                        "total_drag_rmse": 0.01,
                        "total_drag_target_rms": 1.0,
                    }
                    for horizon in module.HORIZONS
                },
            }
            for case in cases
        ],
    }
    segments = {
        "segments": [
            {
                "case": case,
                "horizon": 100,
                "start": 0,
                "target_total_drag": 1.0,
                "predicted_total_drag": 1.0,
            }
            for case in cases
        ]
    }
    physical = {
        "phases": {
            phase: {"true_cfd_total_cd_ranking": ["minus", "zero", "plus"]}
            for phase in ("b01", "b05")
        }
    }
    report_path = tmp_path / "report.json"
    segments_path = tmp_path / "segments.json"
    physical_path = tmp_path / "physical.json"
    write_json(report_path, report)
    write_json(segments_path, segments)
    write_json(physical_path, physical)
    result = module.audit(
        SimpleNamespace(
            data=data,
            checkpoint=checkpoint,
            checkpoint_epoch=1,
            expected_model_sha=model_sha,
            report=report_path,
            segments=segments_path,
            physical_qc=physical_path,
        )
    )
    assert result["checkpoint_epoch"] == 1
    assert result["checkpoint_sha256"] == model_sha
    assert result["frozen_test_accessed"] is False
