#!/usr/bin/env python3
"""Audit the predeclared validation-only dynamic6 PhysicsNeMo diagnostic."""

import argparse, hashlib, json, math, os, tempfile
from pathlib import Path

CASES = {
    f"full40_dynamic_validation_b{b:02d}_{p}"
    for b in (1, 5)
    for p in ("minus", "zero", "plus")
}
HORIZONS = {"1", "10", "50", "100"}
MAX_NRMSE = 0.10
MAX_DELTA_MAE = 0.023
MODEL_SHA = "a66779c18e4c6c0724f903dd4e767eee643d0867180ad8c6cab95587353537ae"
MANIFEST_SHA = "bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    y = float(x)
    if not math.isfinite(y):
        raise ValueError("non-finite metric")
    return y


def audit(args):
    manifest = load(args.data / "manifest.json")
    report = load(args.report)
    segments = load(args.segments)
    physical = load(args.physical_qc)
    if (
        sha(args.data / "manifest.json") != MANIFEST_SHA
        or manifest.get("trajectory_counts")
        != {"train": 0, "validation": 6, "frozen_test": 0}
        or manifest.get("training_access") != "FORBIDDEN"
    ):
        raise ValueError("dynamic6 dataset identity differs")
    if sha(args.checkpoint / "FNO.0.5.mdlus") != MODEL_SHA:
        raise ValueError("e5 model SHA differs")
    if (
        report.get("checkpoint_epoch") != 5
        or report.get("split") != "validation"
        or report.get("action_mode") != "observed"
        or report.get("force_channels")
        != ["front_cd", "front_cl", "rear_cd", "rear_cl"]
    ):
        raise ValueError("evaluation contract differs")
    rows = {r.get("case"): r for r in report.get("cases", [])}
    if set(rows) != CASES or any(
        set(r.get("horizons", {})) != HORIZONS for r in rows.values()
    ):
        raise ValueError("case/horizon matrix differs")
    h100 = []
    for name, row in rows.items():
        for h, metric in row["horizons"].items():
            if metric.get("stable") is not True or metric.get("failed_segments") != 0:
                raise ValueError(f"unstable rollout: {name}/H{h}")
        m = row["horizons"]["100"]
        h100.append(
            (
                int(m["segments"]),
                finite(m["total_drag_rmse"]),
                finite(m["total_drag_target_rms"]),
            )
        )
    pooled = math.sqrt(
        sum(n * e * e for n, e, _ in h100) / sum(n * t * t for n, _, t in h100)
    )
    starts = {
        r["case"]: r
        for r in segments.get("segments", [])
        if r.get("horizon") == 100 and r.get("start") == 0
    }
    if set(starts) != CASES:
        raise ValueError("strict H100 start0 matrix differs")
    differences = []
    ordering = {}
    for b in (1, 5):
        prefix = f"full40_dynamic_validation_b{b:02d}_"
        zero = starts[prefix + "zero"]
        phase = []
        for p in ("minus", "plus"):
            row = starts[prefix + p]
            td = finite(row["target_total_drag"]) - finite(zero["target_total_drag"])
            pd = finite(row["predicted_total_drag"]) - finite(
                zero["predicted_total_drag"]
            )
            differences.append(abs(pd - td))
            phase.append(
                {
                    "profile": p,
                    "true_delta_cd": td,
                    "predicted_delta_cd": pd,
                    "absolute_error": abs(pd - td),
                }
            )
        ordering[f"b{b:02d}"] = phase
    delta_mae = sum(differences) / len(differences)
    passed = pooled <= MAX_NRMSE and delta_mae <= MAX_DELTA_MAE
    return {
        "status": (
            "DYNAMIC6_FNO_DIAGNOSTIC_PASS" if passed else "DYNAMIC6_FNO_DIAGNOSTIC_FAIL"
        ),
        "scope": "validation-only diagnostic; no training, frozen access, PPO authorization, or control-benefit claim",
        "checkpoint_epoch": 5,
        "checkpoint_sha256": MODEL_SHA,
        "dataset_manifest_sha256": MANIFEST_SHA,
        "horizons": [1, 10, 50, 100],
        "pooled_h100_total_cd_nrmse": pooled,
        "pooled_h100_limit": MAX_NRMSE,
        "strict_start0_h100_delta_total_cd_mae": delta_mae,
        "strict_delta_limit": MAX_DELTA_MAE,
        "strict_pairs": ordering,
        "true_cfd_window_rankings": {
            k: v["true_cfd_total_cd_ranking"] for k, v in physical["phases"].items()
        },
        "frozen_test_accessed": False,
        "ppo_authorized": False,
        "passes_predeclared_joint_diagnostic": passed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--physical-qc", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=a.output.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(result, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, a.output)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
