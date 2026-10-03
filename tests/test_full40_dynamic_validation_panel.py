from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "cfd/tandem_cylinders/make_full40_dynamic_validation_panel.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("dynamic_panel", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profiles_obey_canonical_action_and_rate_bounds() -> None:
    module = load_generator()
    assert set(module.PROFILES) == {"minus", "zero", "plus"}
    for points in module.PROFILES.values():
        metrics = module.validate_points([list(row) for row in points])
        assert metrics["max_abs_omega"] <= 0.75
        assert metrics["max_abs_delta_omega_per_0p1"] <= 0.1
        assert points[0] == (0.0, 0.0)
        assert points[-1] == (20.0, 0.0)
    assert module.PROFILES["minus"] == tuple(
        (time, -omega) for time, omega in module.PROFILES["plus"]
    )


def test_predeclaration_is_validation_only_and_counterfactual_scope_is_strict() -> None:
    module = load_generator()
    full40, cases = module.load_cases()
    payload = module.predeclaration(full40, cases)
    assert payload["frozen_test_accessed"] is False
    assert payload["matrix"] == {
        "phase_bins": [1, 5],
        "profiles": ["minus", "zero", "plus"],
        "case_count": 6,
        "duration_D_over_U": 20.0,
    }
    assert all(row["split"] == "validation" for row in payload["cases"].values())
    contract = payload["fno_validation_contract"]
    assert contract["horizons_frames"] == [1, 10, 50, 100]
    assert contract["strict_counterfactual_pairs"] == "only start_frame=0 within each phase"
    assert contract["terminal_total_cd_pooled_nrmse_max"] == 0.10
    assert contract["strict_start0_delta_total_cd_mae_max"] == 0.023


def test_execution_and_qc_are_fail_closed() -> None:
    runner = (ROOT / "cfd/tandem_cylinders/run_full40_dynamic_validation_case.sh").read_text(
        encoding="utf-8"
    )
    audit = (ROOT / "scripts/audit_full40_dynamic_validation_panel.py").read_text(
        encoding="utf-8"
    )
    assert "--preflight-only" in runner
    assert "FULL40_DYNAMIC_VALIDATION_APPROVAL_TOKEN" in runner
    assert "MemAvailable is below 40 GiB" in runner
    assert "CFD/Curator activity detected" in runner
    assert "pgrep -af" not in runner
    assert '$1 == "pimpleFoam"' in runner
    assert '$1 ~ /^python(3)?$/' in runner
    assert "check_dynamic6_qs1_coexistence.py" in runner
    assert "expected_repo=\"/workspace/fluid_control\"" in runner
    assert "Worker temporary copies require a separate reviewed transfer protocol" in runner
    assert "refusing existing solver output" in runner
    assert "dst=/workspace/frozen" not in runner
    assert "frozen_test_accessed\") is not False" in runner
    assert "write_exclusive" in audit
    assert "Only frame 0 within each phase is a strict counterfactual" in audit
    assert "not closed-loop PPO" in audit


def test_qs1_coexistence_is_exact_and_resource_bounded() -> None:
    guard = (
        ROOT / "scripts/check_dynamic6_qs1_coexistence.py"
    ).read_text(encoding="utf-8")
    assert "fluid-control-dev30-quickscreen-pipeline-qs1.service" not in guard
    assert "fluid-control-dev30-quickscreen-h20-qs1.service" in guard
    assert 'host.get("Memory") == 64 * 1024**3' in guard
    assert 'host.get("NanoCpus") == 8_000_000_000' in guard
    assert 'row.get("DeviceIDs") == ["0"]' in guard
    assert 'mem_available_gib() < 60' in guard
    assert 'mounts.get("/workspace/devdata") == (str(DATA), False)' in guard
    assert '"/workspace/frozen" not in mounts' in guard
    assert "tracked <= container_pids(matches[0])" in guard


def test_qs1_container_contract_rejects_mutated_resource_or_output() -> None:
    path = ROOT / "scripts/check_dynamic6_qs1_coexistence.py"
    spec = importlib.util.spec_from_file_location("dynamic6_coexistence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = module.ALLOWED["fluid-control-dev30-quickscreen-h20-qs1.service"]
    payload = {
        "State": {"Running": True},
        "Image": module.IMAGE_ID,
        "Config": {
            "Cmd": [
                "python",
                f"/workspace/scripts/{contract['trainer']}",
                "--config-name",
                contract["config"],
            ]
        },
        "HostConfig": {
            "NetworkMode": "none",
            "Memory": 64 * 1024**3,
            "NanoCpus": 8_000_000_000,
            "ReadonlyRootfs": True,
            "AutoRemove": True,
            "DeviceRequests": [{"DeviceIDs": ["0"]}],
        },
        "Mounts": [
            {
                "Destination": "/workspace/devdata",
                "Source": str(module.DATA),
                "RW": False,
            },
            {
                "Destination": "/workspace/output",
                "Source": str(contract["output"]),
                "RW": True,
            },
        ],
    }
    assert module.valid_container(payload, contract)
    payload["HostConfig"]["Memory"] = 65 * 1024**3
    assert not module.valid_container(payload, contract)
    payload["HostConfig"]["Memory"] = 64 * 1024**3
    payload["Mounts"][1]["Source"] = "/tmp/unreviewed-output"
    assert not module.valid_container(payload, contract)


def test_serial_runner_is_fixed_fail_stop_and_lock_guarded() -> None:
    runner = (ROOT / "scripts/run_full40_dynamic6_serial_spark.sh").read_text(
        encoding="utf-8"
    )
    expected = [
        "full40_dynamic_validation_b01_zero",
        "full40_dynamic_validation_b01_minus",
        "full40_dynamic_validation_b01_plus",
        "full40_dynamic_validation_b05_zero",
        "full40_dynamic_validation_b05_minus",
        "full40_dynamic_validation_b05_plus",
    ]
    positions = [runner.index(case) for case in expected]
    assert positions == sorted(positions)
    assert "set -euo pipefail" in runner
    assert "flock -n 9" in runner
    assert 'bash "$runner" "$case" --preflight-only' in runner
    assert 'bash "$runner" "$case" --execute' in runner
    assert "EXECUTE_REVIEWED_FULL40_DYNAMIC6_SERIAL" in runner
    assert '--execute &\n' not in runner
