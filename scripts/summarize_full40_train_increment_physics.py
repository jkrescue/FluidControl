#!/usr/bin/env python3
"""Summarize the exact first four full40 train-only open-loop branches.

The comparison is deliberately narrow: b00/b02 at omega=+/-0.375 are
compared with their already-QC-passed same-phase zero branches over the
predeclared final 60 D/U.  Validation and frozen-test cases are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

PHASE_MANIFEST_SHA256 = (
    "6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603"
)
FULL40_PREDECLARATION_SHA256 = (
    "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
)
FULL40_AUTHORIZATION_SHA256 = (
    "da8bccaf18a86666ac78804e775c1608d8fc390bfca1484cbd1eaabe93c17151"
)
PHASES = (0, 2)
ACTIONS = (-0.375, 0.375)
ACTION_LABELS = {-0.375: "m0375", 0.375: "p0375"}
FORCE_DT = 0.005
EXPECTED_ANALYSIS_SAMPLES = 12001


def increment_case_name(phase_bin: int, action: float) -> str:
    return (
        f"matched_start_acquisition_train_b{phase_bin:02d}_"
        f"{ACTION_LABELS[action]}"
    )


def zero_case_name(phase_bin: int) -> str:
    return f"matched_start_acquisition_train_b{phase_bin:02d}_zero"


EXACT_INCREMENT_CASES = tuple(
    increment_case_name(phase, action) for phase in PHASES for action in ACTIONS
)
EXACT_ZERO_CASES = tuple(zero_case_name(phase) for phase in PHASES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_force_rows(case: Path, force_name: str) -> np.ndarray:
    samples: dict[float, np.ndarray] = {}
    paths = sorted(case.glob(f"postProcessing/{force_name}/*/coefficient.dat"))
    if not paths:
        raise FileNotFoundError(f"missing {force_name} output: {case}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            values = np.asarray([float(value) for value in line.split()], dtype=float)
            if len(values) < 8 or not np.isfinite(values).all():
                raise ValueError(f"invalid force record: {path}")
            key = round(float(values[0]), 8)
            if key in samples and not np.allclose(
                samples[key], values, rtol=0, atol=1e-10
            ):
                raise ValueError(f"conflicting force record at t={key}: {case}")
            samples[key] = values
    return np.asarray([samples[key] for key in sorted(samples)], dtype=float)


def select_exact_window(rows: np.ndarray, begin: float, end: float) -> np.ndarray:
    selected = rows[(rows[:, 0] >= begin - 1e-9) & (rows[:, 0] <= end + 1e-9)]
    expected = np.linspace(begin, end, EXPECTED_ANALYSIS_SAMPLES)
    if len(selected) != EXPECTED_ANALYSIS_SAMPLES:
        raise ValueError(
            f"analysis window requires {EXPECTED_ANALYSIS_SAMPLES} samples, "
            f"got {len(selected)}"
        )
    if not np.allclose(selected[:, 0], expected, rtol=0, atol=1e-8):
        raise ValueError("analysis timestamps do not match the fixed 0.005 grid")
    return selected


def branch_metrics(front: np.ndarray, rear: np.ndarray) -> dict[str, float | int]:
    if not np.array_equal(front[:, 0], rear[:, 0]):
        raise ValueError("front/rear force timelines differ")
    rear_cl = rear[:, 4]
    rear_cl_mean = float(np.mean(rear_cl))
    return {
        "sample_count": len(front),
        "mean_cd_front": float(np.mean(front[:, 1])),
        "mean_cd_rear": float(np.mean(rear[:, 1])),
        "mean_cd_total": float(np.mean(front[:, 1] + rear[:, 1])),
        "mean_cl_rear": rear_cl_mean,
        "rms_cl_rear_fluctuation": float(
            np.sqrt(np.mean((rear_cl - rear_cl_mean) ** 2))
        ),
    }


def comparison(control: dict, zero: dict) -> dict[str, float | bool]:
    drag_reduction = 1.0 - control["mean_cd_total"] / zero["mean_cd_total"]
    fluctuation_ratio = (
        control["rms_cl_rear_fluctuation"] / zero["rms_cl_rear_fluctuation"]
    )
    mean_bias_ratio = (
        abs(control["mean_cl_rear"]) / zero["rms_cl_rear_fluctuation"]
    )
    return {
        "total_drag_reduction_fraction_positive_is_better": float(drag_reduction),
        "rear_cl_fluctuation_rms_ratio_to_zero": float(fluctuation_ratio),
        "abs_mean_rear_cl_over_zero_fluctuation_rms": float(mean_bias_ratio),
        "canonical_drag_check_ge_0p02": bool(drag_reduction >= 0.02),
        "canonical_rear_cl_fluctuation_check_le_1p05": bool(fluctuation_ratio <= 1.05),
        "canonical_mean_rear_cl_check_le_0p10_zero_clprime_rms": bool(
            mean_bias_ratio <= 0.10
        ),
        "canonical_joint_diagnostic_pass": bool(
            drag_reduction >= 0.02
            and fluctuation_ratio <= 1.05
            and mean_bias_ratio <= 0.10
        ),
    }


def _validate_config(config: dict, name: str, phase: int, action: float) -> None:
    if any(token in name for token in ("validation", "frozen", "test")):
        raise ValueError(f"non-train identity is forbidden: {name}")
    begin, end = (float(value) for value in config.get("analysis_window", []))
    if (
        config.get("case") != name
        or config.get("split") != "train"
        or config.get("phase_bin") != phase
        or float(config.get("action_target")) != action
        or config.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
        or not np.isclose(begin, float(config.get("start_time")) + 20.0)
        or not np.isclose(end, float(config.get("end_time")))
        or not np.isclose(end - begin, 60.0)
    ):
        raise ValueError(f"config violates fixed train/window contract: {name}")


def load_and_validate_prerequisites(
    repo: Path,
    cases_dir: Path,
    increment_receipt_dir: Path,
    zero_receipt_dir: Path,
    aggregate_qc: Path,
) -> dict[str, tuple[Path, dict]]:
    aggregate = read_json(aggregate_qc)
    frozen_nine = {
        f"matched_start_acquisition_train_b{phase:02d}_{label}"
        for phase in (0, 2, 4)
        for label in ("m075", "zero", "p075")
    }
    aggregate_rows = {row.get("case"): row for row in aggregate.get("cases", [])}
    if (
        aggregate.get("status") != "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS"
        or aggregate.get("complete_nine_case_panel") is not True
        or aggregate.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
        or set(aggregate_rows) != frozen_nine
        or set(aggregate.get("transfer", {})) != frozen_nine
    ):
        raise ValueError("nine-case QC is not the exact reviewed train panel")

    panel: dict[str, tuple[Path, dict]] = {}
    for phase in PHASES:
        zero_name = zero_case_name(phase)
        zero_receipt = read_json(zero_receipt_dir / f"{zero_name}.json")
        zero_case = cases_dir / zero_name
        zero_config = read_json(zero_case / "case_config.json")
        _validate_config(zero_config, zero_name, phase, 0.0)
        if (
            zero_receipt.get("status") != "RAW_TRANSFER_VERIFIED"
            or zero_receipt.get("case") != zero_name
            or zero_receipt.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
            or aggregate_rows[zero_name].get("source_state_sha256")
            != zero_config.get("source_state_sha256")
        ):
            raise ValueError(f"invalid same-phase zero provenance: {zero_name}")
        panel[zero_name] = zero_case, zero_config

        for action in ACTIONS:
            name = increment_case_name(phase, action)
            receipt_path = increment_receipt_dir / f"{name}.json"
            receipt = read_json(receipt_path)
            case = cases_dir / name
            config = read_json(case / "case_config.json")
            _validate_config(config, name, phase, action)
            manifest = repo / str(receipt.get("worker_raw_manifest", ""))
            if (
                receipt.get("status") != "FULL40_RAW_TRANSFER_VERIFIED"
                or receipt.get("case") != name
                or receipt.get("split") != "train"
                or receipt.get("phase_bin") != phase
                or float(receipt.get("action_target")) != action
                or receipt.get("full40_predeclaration_sha256")
                != FULL40_PREDECLARATION_SHA256
                or receipt.get("full40_extension_authorization_sha256")
                != FULL40_AUTHORIZATION_SHA256
                or config.get("full40_predeclaration_sha256")
                != FULL40_PREDECLARATION_SHA256
                or config.get("full40_extension_authorization_sha256")
                != FULL40_AUTHORIZATION_SHA256
                or receipt.get("source_state_sha256")
                != zero_config.get("source_state_sha256")
                or config.get("source_state_sha256")
                != zero_config.get("source_state_sha256")
                or not manifest.is_file()
                or sha256(manifest) != receipt.get("worker_raw_manifest_sha256")
            ):
                raise ValueError(f"invalid full40 train receipt/provenance: {name}")
            panel[name] = case, config
    if set(panel) != set(EXACT_INCREMENT_CASES) | set(EXACT_ZERO_CASES):
        raise AssertionError("internal exact-case allowlist mismatch")
    return panel


def build_report(
    repo: Path,
    cases_dir: Path,
    increment_receipt_dir: Path,
    zero_receipt_dir: Path,
    aggregate_qc: Path,
) -> dict:
    panel = load_and_validate_prerequisites(
        repo, cases_dir, increment_receipt_dir, zero_receipt_dir, aggregate_qc
    )
    phases = {}
    all_results = []
    for phase in PHASES:
        zero_name = zero_case_name(phase)
        zero_case, zero_config = panel[zero_name]
        begin, end = (float(value) for value in zero_config["analysis_window"])
        zero = branch_metrics(
            select_exact_window(read_force_rows(zero_case, "forceFront"), begin, end),
            select_exact_window(read_force_rows(zero_case, "forceRear"), begin, end),
        )
        controls = {}
        for action in ACTIONS:
            name = increment_case_name(phase, action)
            case, config = panel[name]
            control_begin, control_end = (
                float(value) for value in config["analysis_window"]
            )
            if (control_begin, control_end) != (begin, end):
                raise ValueError(f"same-phase analysis windows differ: {name}")
            metrics = branch_metrics(
                select_exact_window(
                    read_force_rows(case, "forceFront"), control_begin, control_end
                ),
                select_exact_window(
                    read_force_rows(case, "forceRear"), control_begin, control_end
                ),
            )
            result = comparison(metrics, zero)
            all_results.append(result)
            controls[ACTION_LABELS[action]] = {
                "case": name,
                "action_target": action,
                "metrics": metrics,
                "same_phase_zero_comparison": result,
            }
        phases[f"b{phase:02d}"] = {
            "split": "train",
            "analysis_window": [begin, end],
            "same_phase_zero": {"case": zero_name, "metrics": zero},
            "increment_branches": controls,
        }

    return {
        "status": "FULL40_FIRST4_TRAIN_INCREMENT_PHYSICS_SUMMARY",
        "scope": {
            "increment_cases": list(EXACT_INCREMENT_CASES),
            "zero_reference_cases": list(EXACT_ZERO_CASES),
            "excluded_splits": ["validation", "frozen_test"],
        },
        "provenance": {
            "phase_manifest_sha256": PHASE_MANIFEST_SHA256,
            "full40_predeclaration_sha256": FULL40_PREDECLARATION_SHA256,
            "full40_extension_authorization_sha256": FULL40_AUTHORIZATION_SHA256,
            "nine_case_aggregate_qc": str(aggregate_qc),
            "nine_case_aggregate_qc_sha256": sha256(aggregate_qc),
        },
        "metric_contract": {
            "window": "each case_config predeclared final 60D/U; no subwindow selection",
            "force_sample_interval": FORCE_DT,
            "samples_per_branch": EXPECTED_ANALYSIS_SAMPLES,
            "total_drag": "mean(Cd_front + Cd_rear)",
            "canonical_joint_gate": (
                "drag reduction >=2%; rear Cl-prime RMS <=1.05x same-phase zero; "
                "abs(mean rear Cl) <=0.10x same-phase zero Cl-prime RMS"
            ),
        },
        "phases": phases,
        "all_four_canonical_joint_diagnostics_pass": all(
            row["canonical_joint_diagnostic_pass"] for row in all_results
        ),
        "interpretation_guards": [
            "These are prescribed open-loop branches in the train split only.",
            "This summary is not validation, frozen-test, or generalization evidence.",
            "This summary is not learned control or online closed-loop evidence.",
            "Results must not stop, reorder, or tune the predeclared acquisition panel.",
        ],
    }


def write_json_exclusive_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-dir", type=Path, default=repo / "cfd/tandem_cylinders/cases"
    )
    parser.add_argument(
        "--increment-receipt-dir",
        type=Path,
        default=repo / "artifacts/matched_start_full40_extension/transfer_verified",
    )
    parser.add_argument(
        "--zero-receipt-dir",
        type=Path,
        default=repo / "artifacts/matched_start_acquisition/transfer_verified",
    )
    parser.add_argument(
        "--aggregate-qc",
        type=Path,
        default=repo / "artifacts/matched_start_acquisition/aggregate_qc/result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo
            / "artifacts/matched_start_full40_extension/train_increment_physics"
            / "first4_b00_b02_pm0375.json"
        ),
    )
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((repo / "artifacts").resolve()):
        parser.error("output must be inside the repository artifacts directory")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = build_report(
        repo,
        args.cases_dir,
        args.increment_receipt_dir,
        args.zero_receipt_dir,
        args.aggregate_qc,
    )
    write_json_exclusive_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": len(report["scope"]["increment_cases"]),
                "all_four_joint_pass": report[
                    "all_four_canonical_joint_diagnostics_pass"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
