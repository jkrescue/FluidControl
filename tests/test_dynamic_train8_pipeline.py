from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "cfd/tandem_cylinders/make_dynamic_train8_panel.py"
CURATOR = ROOT / "cfd/tandem_cylinders/curate_dynamic_train8.py"
AUDIT = ROOT / "cfd/tandem_cylinders/audit_dynamic_train8_panel.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_schedules_are_deterministic_bounded_and_train_only() -> None:
    module = load(GENERATOR, "dynamic_train8_generator")
    first, second = module.schedules(), module.schedules()
    assert first == second
    assert set(first) == {0, 2, 4, 6}
    for phase, profiles in first.items():
        assert set(profiles) == {"prbs", "multisine"}
        assert profiles["prbs"] != profiles["multisine"]
        for points in profiles.values():
            metrics = module.validate_points(points)
            assert metrics["max_abs_omega"] <= 0.75
            assert metrics["max_abs_delta_omega_per_0p1"] <= 0.1 + 1e-10
            assert abs(metrics["mean_omega"]) <= 0.03
            assert len(points) == 201


def test_predeclaration_never_uses_validation_or_frozen(monkeypatch) -> None:
    module = load(GENERATOR, "dynamic_train8_predecl")
    cases = {
        f"dynamic_train8_b{phase:02d}_{profile}": {
            "split": "train", "phase_bin": phase, "profile": profile
        }
        for phase in (0, 2, 4, 6)
        for profile in ("prbs", "multisine")
    }
    monkeypatch.setattr(module, "load_contract", lambda: ({"phase_manifest_sha256": "a" * 64}, cases))
    payload = module.predeclaration()
    assert payload["matrix"]["case_count"] == 8
    assert all(row["split"] == "train" for row in payload["cases"].values())
    assert payload["curation_contract"]["validation_or_frozen_access"] == "forbidden"
    assert payload["solver_contract"]["maximum_total_raw_GiB"] == 30


def test_generation_and_runner_bind_reviewed_sha_and_curator_stays_disabled() -> None:
    generator = load(GENERATOR, "dynamic_train8_disabled")
    curator = load(CURATOR, "dynamic_train8_curator_disabled")
    runner = (ROOT / "cfd/tandem_cylinders/run_dynamic_train8_case.sh").read_text()
    digest = "4cf4e7c9b9da27b71e58db2e94b0750736b7f79f09a2aebc3ffa97729e882c5a"
    assert generator.APPROVED_PREDECLARATION_SHA256 == digest
    assert curator.PREDECL_SHA == digest
    assert curator.EXECUTION_REVIEWED is False
    assert f'predecl_sha="{digest}"' in runner
    assert "DYNAMIC_TRAIN8_APPROVAL_TOKEN" in runner
    assert "active < 4" in runner
    assert "Spark free disk below 230 GiB launch floor" in runner
    assert "{ pgrep -x pimpleFoam || true; }" in runner
    assert "--cpus 1" in runner


def test_curator_matrix_and_exclusive_receipts(tmp_path) -> None:
    module = load(CURATOR, "dynamic_train8_curator")
    assert module.EXPECTED == {
        f"dynamic_train8_b{phase:02d}_{profile}"
        for phase in (0, 2, 4, 6)
        for profile in ("prbs", "multisine")
    }
    target = tmp_path / "receipt.json"
    module.exclusive(target, {"status": "FIRST"})
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module.exclusive(target, {"status": "SECOND"})
    assert json.loads(target.read_text()) == {"status": "FIRST"}


def test_authorization_rejects_unreviewed_sha(tmp_path) -> None:
    module = load(CURATOR, "dynamic_train8_auth")
    path = tmp_path / module.PREDECL
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed predeclaration SHA differs"):
        module.make_authorization(tmp_path)


def test_curator_uses_official_source_filter_sink_contract() -> None:
    source = CURATOR.read_text()
    assert ".filter(base.NumericalQualityFilter(0.75))" in source
    assert ".write(base.TrajectoryHDF5Sink" in source
    assert "base.run_pipeline(" in source
    assert '"split": "train"' in source
    assert '"validation": 0, "frozen_test": 0' in source
    assert "validate_matched_start_source_force" in source
    assert '"max_abs_omega": 0.75' in source
    assert "shutil.copyfile(source_normalization, temporary)" in source
    assert "sha256(target_normalization) != sha256(source_normalization)" in source


def test_raw_audit_is_exact_eight_case_fail_closed() -> None:
    module = load(AUDIT, "dynamic_train8_audit")
    assert module.EXPECTED == {
        f"dynamic_train8_b{phase:02d}_{profile}"
        for phase in (0, 2, 4, 6)
        for profile in ("prbs", "multisine")
    }
    source = AUDIT.read_text()
    assert "values.shape[0] != 4000" in source
    assert 'config.get("split") != "train"' in source
    assert 'marker.get("solver_log_sha256") != sha256(log_path)' in source
    assert 'sha256(case / "source_restart_provenance" / field)' in source
    assert 'omega_table(case / f"{start:g}" / "U")' in source
    assert '"validation_or_frozen_accessed": False' in source
