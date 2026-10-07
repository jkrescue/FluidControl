#!/usr/bin/env python3
"""Exact frozen canonical-joint reward replay for six saved B action branches.

This is a CPU-only descriptive diagnostic.  It uses the real OpenFOAM causal
prehistory at q0 and the existing saved H1--H5 truth/prediction arrays.  It does
not load a model, choose thresholds, or claim controller admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

PHASES = {"b01": 130.0, "b05": 102.0}
ROLES = ("minus", "zero", "plus")
EXPECTED_SOURCE_SHA256 = "ce4dab24d07fef642be9d43faf9897519271eb0b1281c71e7b0488c06598e6df"
EXPECTED_REWARD_SHA256 = "138ab2b49ebddcbed2a24486c995b85a3a5ae226ee936ff2ed5de318496ee8bd"
EXPECTED_HISTORY_SHA256 = "ec8720581e0f362c308a2bf82fb1f05585eaf876cf17956a6c8e7df0ca14c763"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def coefficient_rows(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 5:
            rows.append((float(fields[0]), float(fields[1]), float(fields[4])))
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"invalid coefficient table: {path}")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"non-increasing coefficient clock: {path}")
    return values


def validation_baseline(case: Path, end: float) -> tuple[dict, dict]:
    sources = {}
    arrays = []
    for obj in ("forceFront", "forceRear"):
        paths = sorted(case.glob(f"postProcessing/{obj}/*/coefficient.dat"))
        if len(paths) != 1:
            raise ValueError(f"expected one {obj} table in {case}")
        table = coefficient_rows(paths[0])
        selected = table[(table[:, 0] >= end - 60.0 - 1e-10) & (table[:, 0] <= end + 1e-10)]
        if selected.shape != (12001, 3):
            raise ValueError(f"final60 grid differs for {case}/{obj}: {selected.shape}")
        expected = end - 60.0 + 0.005 * np.arange(12001)
        if float(np.max(np.abs(selected[:, 0] - expected))) > 2e-8:
            raise ValueError(f"final60 clock differs for {case}/{obj}")
        arrays.append(selected)
        sources[obj] = {"path": str(paths[0]), "sha256": sha256(paths[0])}
    if not np.array_equal(arrays[0][:, 0], arrays[1][:, 0]):
        raise ValueError("front/rear baseline clocks differ")
    total_drag = float(np.mean(arrays[0][:, 1] + arrays[1][:, 1]))
    rear_cl = arrays[1][:, 2]
    rear_fluctuation = float(np.sqrt(np.mean(np.square(rear_cl - np.mean(rear_cl)))))
    return {
        "total_drag": total_drag,
        "rear_cl_fluctuation_rms": rear_fluctuation,
        "source": f"{case}:predeclared_final_60D/U_validation_same_phase_zero",
    }, sources


def total_cost(components: dict[str, float]) -> float:
    expected = {
        "drag_screen", "drag_gate_violation", "rear_cl_fluctuation_gate_violation",
        "rear_cl_mean_bias_gate_violation", "actuation", "rate",
    }
    if set(components) != expected:
        raise ValueError("canonical component schema differs")
    return float(sum(float(components[key]) for key in sorted(expected)))


def execute(root: Path, source_result: Path, output: Path) -> dict:
    frozen = root / "artifacts/exploratory_h5_ppo_source_20261006_immutable/src/fluid_control"
    reward_path = frozen / "canonical_joint_v1.py"
    history_path = frozen / "openfoam_force_history.py"
    if sha256(source_result) != EXPECTED_SOURCE_SHA256:
        raise ValueError("saved action-sequence result SHA differs")
    if sha256(reward_path) != EXPECTED_REWARD_SHA256:
        raise ValueError("frozen canonical reward SHA differs")
    if sha256(history_path) != EXPECTED_HISTORY_SHA256:
        raise ValueError("frozen causal-history loader SHA differs")
    reward = load_module("p064_frozen_canonical_joint_v1", reward_path)
    history = load_module("p064_frozen_openfoam_force_history", history_path)
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if source.get("status") != "SAVED_B_SHORT_ACTION_SEQUENCE_DIAGNOSTIC_NOT_ADMISSION":
        raise ValueError("saved action-sequence result status differs")
    if set(source.get("phases", {})) != set(PHASES):
        raise ValueError("saved phases differ")
    baseline_artifact = root / "artifacts/matched_start_full40_extension/train20_physics_summary.json"
    baseline_document = json.loads(baseline_artifact.read_text(encoding="utf-8"))
    if baseline_document.get("scope", {}).get("phase_bins") != [0, 2, 4, 6]:
        raise ValueError("B training baseline artifact scope differs")

    phases = {}
    for phase, restart in PHASES.items():
        phase_source = source["phases"][phase]
        if not math.isclose(float(phase_source["source_restart_time"]), restart, abs_tol=1e-12):
            raise ValueError(f"{phase} restart differs")
        source_case = root / "cfd/tandem_cylinders/cases/tandem_backward_dt005"
        times, forces, history_sources = history.actual_causal_prehistory(
            source_case, restart, provenance_root=root, control_dt=0.1, sample_count=62
        )
        past_t = np.asarray(times, dtype=np.float64)
        past_f = np.asarray(forces, dtype=np.float64)
        if past_f.shape != (62, 4) or not np.isclose(past_t[-1], restart, atol=1e-12):
            raise ValueError(f"{phase} causal prehistory differs")
        hdf = root / (
            "data/curated/tandem_cylinders_full40_dynamic_validation_v1/validation/"
            f"full40_dynamic_validation_{phase}_minus.h5"
        )
        import h5py
        with h5py.File(hdf, "r") as handle:
            q0_force = np.asarray(handle["force"][0], dtype=np.float32)
            q0_time = float(np.asarray(handle["time"][0]).reshape(()))
        if not np.array_equal(past_f[-1].astype(np.float32), q0_force):
            raise ValueError(f"{phase} real prehistory does not end at original q0 force")
        if not np.isclose(q0_time, restart, rtol=0.0, atol=2e-5):
            raise ValueError(f"{phase} original q0 clock differs")
        branch_hdf_sha = str(phase_source["branches"]["minus"]["hdf_sha256"])
        if sha256(hdf) != branch_hdf_sha:
            raise ValueError(f"{phase} q0 HDF SHA differs from saved-array binding")
        baseline_case = root / f"cfd/tandem_cylinders/cases/matched_start_acquisition_validation_{phase}_zero"
        baseline, baseline_sources = validation_baseline(baseline_case, restart + 80.0)
        branches = {}
        for role in ROLES:
            row = phase_source["branches"][role]
            truth = np.asarray(row["truth"], dtype=np.float64)
            prediction = np.asarray(row["prediction"], dtype=np.float64)
            omega = np.asarray(row["omega"], dtype=np.float64)
            future_times = np.asarray(row["times"], dtype=np.float64)
            if (truth.shape != (5, 4) or prediction.shape != (5, 4)
                    or omega.shape != (6,) or future_times.shape != (5,)):
                raise ValueError(f"{phase}/{role} saved shape differs")
            horizons = []
            for h in range(1, 6):
                clock = np.concatenate((past_t[h:], future_times[:h]))
                applied = float(omega[h])
                delta = float(omega[h] - omega[h - 1])
                kinds = {}
                for kind, future in (("prediction", prediction), ("truth", truth)):
                    values = np.concatenate((past_f[h:], future[:h]), axis=0)
                    ledger = reward.causal_window_ledger(clock, values, baseline, window_seconds=6.15)
                    components = reward.canonical_joint_cost_components(
                        ledger, omega=applied, delta_omega=delta
                    )
                    kinds[kind] = {
                        "ledger": ledger,
                        "components": components,
                        "total_cost": total_cost(components),
                    }
                horizons.append({"horizon": h, "omega": applied, "delta_omega": delta, **kinds})
            branches[role] = horizons
        ranking = []
        for h in range(1, 6):
            predicted = {role: branches[role][h - 1]["prediction"]["total_cost"] for role in ROLES}
            truth = {role: branches[role][h - 1]["truth"]["total_cost"] for role in ROLES}
            pred_min = min(predicted.values())
            truth_min = min(truth.values())
            pred_choices = [role for role in ROLES if predicted[role] == pred_min]
            truth_choices = [role for role in ROLES if truth[role] == truth_min]
            predicted_returns = {
                role: float(sum((0.99 ** j) * (-0.1) * branches[role][j]["prediction"]["total_cost"]
                                for j in range(h))) for role in ROLES
            }
            truth_returns = {
                role: float(sum((0.99 ** j) * (-0.1) * branches[role][j]["truth"]["total_cost"]
                                for j in range(h))) for role in ROLES
            }
            pred_return_max = max(predicted_returns.values())
            truth_return_max = max(truth_returns.values())
            pred_return_choices = [role for role in ROLES if predicted_returns[role] == pred_return_max]
            truth_return_choices = [role for role in ROLES if truth_returns[role] == truth_return_max]
            ranking.append({
                "horizon": h,
                "endpoint_window_cost": {
                    "predicted": predicted, "truth": truth,
                    "predicted_minimizers_exact": pred_choices,
                    "truth_minimizers_exact": truth_choices,
                    "minimizer_sets_overlap": bool(set(pred_choices) & set(truth_choices)),
                    "truth_regret_by_predicted_minimizer": {
                        role: float(truth[role] - truth_min) for role in pred_choices
                    },
                },
                "truncated_discounted_reward_no_bootstrap": {
                    "gamma": 0.99, "reward_per_step": "-0.1 * canonical_cost",
                    "predicted": predicted_returns, "truth": truth_returns,
                    "predicted_maximizers_exact": pred_return_choices,
                    "truth_maximizers_exact": truth_return_choices,
                    "maximizer_sets_overlap": bool(set(pred_return_choices) & set(truth_return_choices)),
                    "truth_return_regret_by_predicted_maximizer": {
                        role: float(truth_return_max - truth_returns[role]) for role in pred_return_choices
                    },
                },
            })
        phases[phase] = {
            "restart_time": restart,
            "causal_history": {"sample_count": 62, "start": times[0], "end": times[-1], "sources": history_sources},
            "q0_hdf": {"path": str(hdf.relative_to(root)), "sha256": sha256(hdf),
                       "time": q0_time, "force": q0_force.tolist(), "last_force_float32_exact": True},
            "baseline": baseline,
            "baseline_sources": baseline_sources,
            "branches": branches,
            "ranking": ranking,
        }
    result = {
        "status": "P064_B_CANONICAL_REWARD_SEQUENCE_REPLAY_COMPLETE_NOT_ADMISSION",
        "source_sha256": sha256(Path(__file__)),
        "inputs": {
            str(source_result.relative_to(root)): sha256(source_result),
            str(reward_path.relative_to(root)): sha256(reward_path),
            str(history_path.relative_to(root)): sha256(history_path),
            str(baseline_artifact.relative_to(root)): sha256(baseline_artifact),
        },
        "baseline_scope_guard": (
            "B PPO training artifact contains b00/b02/b04/b06 only; b01/b05 use their existing "
            "same-phase validation-zero final60 sources under the unchanged canonical formula."
        ),
        "phases": phases,
        "model_forward_calls": 0,
        "optimizer_steps": 0,
        "new_cfd": False,
        "scientific_admission": False,
        "limitations": [
            "six saved branches and two fixed validation phases only; no statistical generalization",
            "mixed windows contain real pre-q0 history plus at most five saved future samples",
            "uses stored float32 future clocks; this is arithmetic reconstruction, not byte-identical online stepping",
            "truth reward comparison does not repair surrogate state or force prediction error",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    source = args.source_result.resolve()
    output = args.output.resolve()
    if not source.is_relative_to(root) or not output.is_relative_to(root) or output.exists():
        raise ValueError("input/output path contract differs")
    if not args.execute:
        print("P064_B_CANONICAL_REWARD_REPLAY_PREFLIGHT_PASS_NOT_RUNNING")
        return 0
    result = execute(root, source, output)
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
