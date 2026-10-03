#!/usr/bin/env python3
"""Predeclare and atomically stage one reviewed full40 remainder case.

The default/list and predeclaration modes never create CFD cases.  Case
generation is deliberately one-at-a-time and is disabled until the exact
predeclaration artifact SHA-256 is hard-bound below after review/commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CASES = ROOT / "cases"
SOURCE = CASES / "tandem_backward_dt005"
sys.path.insert(0, str(ROOT))
from make_expanded_control_dataset import replace_once, replace_rear_patch

PHASE_MANIFEST = (
    REPO
    / "artifacts/tandem_cylinders/"
    "matched_start_phase_restart_predeclared_v3_20261003.json"
)
PHASE_MANIFEST_SHA256 = (
    "6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603"
)
PREDECLARATION = (
    REPO
    / "artifacts/tandem_cylinders/"
    "matched_start_full40_predeclared_20261003.json"
)
APPROVED_PREDECLARATION_SHA256 = (
    "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
)
EXTENSION_AUTHORIZATION = (
    REPO
    / "artifacts/tandem_cylinders/"
    "matched_start_full40_extension_authorized_20261003.json"
)
# Filled only after the nine-case aggregate passes and the authorization is committed.
APPROVED_EXTENSION_AUTHORIZATION_SHA256 = "REVIEW_REQUIRED_AFTER_NINE_CASE_AGGREGATE"
GENERATION_TOKEN = "GENERATE_REVIEWED_FULL40_REMAINDER"
STATE_FIELDS = ("U", "U_0", "p", "phi", "phi_0")
PHASE_BINS = tuple(range(8))
ACTIONS = (-0.75, -0.375, 0.0, 0.375, 0.75)
ACTION_LABELS = {
    -0.75: "m075",
    -0.375: "m0375",
    0.0: "zero",
    0.375: "p0375",
    0.75: "p075",
}
SPLIT_BINS = {
    "train": (0, 2, 4, 6),
    "validation": (1, 5),
    "frozen_test": (3, 7),
}
COMMISSIONING_BINS = (0, 2, 4)
COMMISSIONING_ACTIONS = (-0.75, 0.0, 0.75)
DURATION = 80.0
DELTA_T = 0.005
WRITE_INTERVAL = 0.1
RAMP_RATE = 1.0
IMAGE = (
    "opencfd/openfoam-default@sha256:"
    "33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
)


@dataclass(frozen=True)
class Full40Case:
    name: str
    phase_bin: int
    split: str
    source_time: float
    action: float
    source_state_sha256: dict[str, str]
    disposition: str


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def split_for_bin(phase_bin: int) -> str:
    matches = [split for split, bins in SPLIT_BINS.items() if phase_bin in bins]
    if len(matches) != 1:
        raise ValueError(f"phase bin {phase_bin} has no unique split")
    return matches[0]


def case_name(phase_bin: int, split: str, action: float) -> str:
    return f"matched_start_acquisition_{split}_b{phase_bin:02d}_{ACTION_LABELS[action]}"


def action_points(start: float, action: float) -> list[list[float]]:
    end = start + DURATION
    if action == 0.0:
        return [[start, 0.0], [end, 0.0]]
    return [[start, 0.0], [start + abs(action) / RAMP_RATE, action], [end, action]]


def validate_action(points: list[list[float]]) -> dict[str, float]:
    if points[0][1] != 0.0:
        raise ValueError("action must start at zero")
    rates = []
    for (t0, a0), (t1, a1) in pairwise(points):
        if t1 <= t0:
            raise ValueError("action timestamps are not strictly increasing")
        rates.append(abs(a1 - a0) / (t1 - t0))
    metrics = {
        "max_abs_omega": max(abs(row[1]) for row in points),
        "max_abs_domega_dt": max(rates),
    }
    if metrics["max_abs_omega"] > 0.75 + 1e-12 or metrics["max_abs_domega_dt"] > 1.0 + 1e-12:
        raise ValueError("action violates full40 bounds")
    return metrics


def load_plan() -> tuple[dict, list[Full40Case]]:
    if sha256(PHASE_MANIFEST) != PHASE_MANIFEST_SHA256:
        raise ValueError("phase manifest differs from the reviewed v3 SHA-256")
    manifest = json.loads(PHASE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "MATCHED_START_PHASE_RESTART_AUDIT_COMPLETE"
        or manifest.get("decision") != "GO_9_CASE_COMMISSIONING"
    ):
        raise ValueError("v3 phase audit is not complete")
    if manifest.get("full_seed_matrix") != {
        "actions": list(ACTIONS),
        "case_count": 40,
        "authorization": "not authorized by this audit",
    }:
        raise ValueError("v3 full_seed_matrix contract differs")
    force = manifest.get("signal_qc", {})
    force_path = REPO / force.get("force_path", "")
    if not force_path.is_file() or sha256(force_path) != force.get("force_sha256"):
        raise ValueError("phase-source force provenance differs")

    selections = {row.get("phase_bin"): row for row in manifest.get("selections", [])}
    if set(selections) != set(PHASE_BINS):
        raise ValueError("phase manifest must contain exactly bins 0..7")
    specs = []
    for phase_bin in PHASE_BINS:
        row = selections[phase_bin]
        split = split_for_bin(phase_bin)
        if row.get("split") != split or row.get("coverage_pass") is not True:
            raise ValueError(f"phase bin {phase_bin} split/coverage differs")
        selected = row["selected"]
        source_time = float(selected["restart_time"])
        source = SOURCE / f"{source_time:g}"
        names = {path.name for path in source.iterdir() if path.is_file()}
        if names != set(STATE_FIELDS):
            raise ValueError(f"incomplete source state for phase bin {phase_bin}")
        source_hashes = {field: sha256(source / field) for field in STATE_FIELDS}
        if (
            source_hashes["U"] != selected["u_sha256"]
            or source_hashes["p"] != selected["p_sha256"]
        ):
            raise ValueError(f"source U/p differs for phase bin {phase_bin}")
        for action in ACTIONS:
            commissioned = phase_bin in COMMISSIONING_BINS and action in COMMISSIONING_ACTIONS
            specs.append(
                Full40Case(
                    name=case_name(phase_bin, split, action),
                    phase_bin=phase_bin,
                    split=split,
                    source_time=source_time,
                    action=action,
                    source_state_sha256=source_hashes,
                    disposition=(
                        "existing_nine_case_commissioning"
                        if commissioned
                        else "planned_new_remainder_case"
                    ),
                )
            )
    if len(specs) != 40 or len({spec.name for spec in specs}) != 40:
        raise ValueError("full40 plan is not exactly 40 unique cases")
    return manifest, specs


def predeclaration(manifest: dict, specs: list[Full40Case]) -> dict:
    cases = {}
    for spec in specs:
        points = action_points(spec.source_time, spec.action)
        cases[spec.name] = {
            "phase_bin": spec.phase_bin,
            "split": spec.split,
            "source_restart_case": SOURCE.name,
            "source_restart_time": spec.source_time,
            "source_state_sha256": spec.source_state_sha256,
            "action_target": spec.action,
            "action_points": points,
            "action_metrics": validate_action(points),
            "run_window": [spec.source_time, spec.source_time + DURATION],
            "analysis_window": [spec.source_time + 20.0, spec.source_time + DURATION],
            "disposition": spec.disposition,
        }
    split_counts = {
        split: sum(spec.split == split for spec in specs) for split in SPLIT_BINS
    }
    return {
        "status": "MATCHED_START_FULL40_PREDECLARED_NO_NEW_CASES_GENERATED",
        "scope": "eight real phase restarts by five prescribed constant rear-cylinder rotations",
        "phase_manifest": str(PHASE_MANIFEST.relative_to(REPO)),
        "phase_manifest_sha256": PHASE_MANIFEST_SHA256,
        "phase_force_source_sha256": manifest["signal_qc"]["force_sha256"],
        "matrix": {
            "phase_bins": list(PHASE_BINS),
            "actions": list(ACTIONS),
            "case_count": 40,
            "already_generated_commissioning_cases": 9,
            "planned_new_remainder_cases": 31,
            "split_case_counts": split_counts,
        },
        "split_roles": {
            "train": "model fitting only",
            "validation": "model selection only; never train fitting",
            "frozen_test": "one-time final evaluation only; never training or tuning",
        },
        "cases": cases,
        "solver_contract": {
            "image": IMAGE,
            "network": "none",
            "delta_t": DELTA_T,
            "duration": DURATION,
            "field_write_interval": WRITE_INTERVAL,
            "force_write_interval": DELTA_T,
            "expected_steps": 16000,
            "expected_field_frames": 801,
            "expected_raw_force_samples": 16000,
            "expected_aligned_force_samples_with_source_t0": 16001,
            "analysis_window": "fixed final 60D/U; no post-hoc subwindow selection",
        },
        "generation_guards": {
            "one_case_per_invocation": True,
            "exclusive_atomic_case_directory": True,
            "existing_nine_cases_are_validate_only_and_never_overwritten_or_rerun": True,
            "new_case_directory_must_not_exist": True,
            "predeclaration_exact_sha256_must_be_hard_bound_after_review_commit": True,
            "no_case_generation_authorized_by_this_artifact_alone": True,
        },
        "interpretation_guards": [
            "This matrix contains prescribed constant rotations and is not closed-loop control.",
            "Coverage of eight phases and five actions is not evidence of training sufficiency.",
            "All starts come from one real baseline limit cycle and are not independent flows.",
            (
                "The nine commissioning cases retain their original provenance and "
                "are not regenerated."
            ),
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


def validate_existing_nine(specs: list[Full40Case]) -> None:
    existing_specs = [
        spec for spec in specs if spec.disposition == "existing_nine_case_commissioning"
    ]
    if len(existing_specs) != 9:
        raise ValueError("existing commissioning subset is not exactly nine")
    for spec in existing_specs:
        case = CASES / spec.name
        config = json.loads((case / "case_config.json").read_text(encoding="utf-8"))
        if (
            config.get("case") != spec.name
            or config.get("split") != spec.split
            or config.get("phase_bin") != spec.phase_bin
            or float(config.get("action_target")) != spec.action
            or config.get("source_state_sha256") != spec.source_state_sha256
            or config.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
        ):
            raise ValueError(f"existing commissioning case differs: {spec.name}")


def validate_approved_predeclaration(manifest: dict, specs: list[Full40Case]) -> None:
    if sha256(PREDECLARATION) != APPROVED_PREDECLARATION_SHA256:
        raise ValueError("full40 predeclaration SHA-256 differs from reviewed artifact")
    actual = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    if actual != predeclaration(manifest, specs):
        raise ValueError("full40 predeclaration content differs from current sources")


def validate_extension_authorization(specs: list[Full40Case]) -> dict:
    if len(APPROVED_EXTENSION_AUTHORIZATION_SHA256) != 64:
        raise ValueError("full40 extension authorization is not reviewed and hard-bound")
    if sha256(EXTENSION_AUTHORIZATION) != APPROVED_EXTENSION_AUTHORIZATION_SHA256:
        raise ValueError("full40 extension authorization SHA-256 differs")
    authorization = json.loads(EXTENSION_AUTHORIZATION.read_text(encoding="utf-8"))
    expected = sorted(
        spec.name for spec in specs if spec.disposition == "planned_new_remainder_case"
    )
    expected_nine = sorted(
        spec.name
        for spec in specs
        if spec.disposition == "existing_nine_case_commissioning"
    )
    resources = {
        "worker_start_mem_available_gib_at_least": 64,
        "worker_running_mem_available_gib_at_least": 40,
        "worker_free_disk_gib_at_least": 100,
        "spark_free_disk_gib_at_least": 250,
    }
    if (
        authorization.get("status") != "MATCHED_START_FULL40_EXTENSION_AUTHORIZED"
        or authorization.get("phase_manifest_sha256") != PHASE_MANIFEST_SHA256
        or authorization.get("full40_predeclaration_sha256")
        != APPROVED_PREDECLARATION_SHA256
        or authorization.get("authorized_cases") != expected
        or authorization.get("maximum_parallel_cases") != 4
        or authorization.get("resource_guards") != resources
        or len(authorization.get("nine_case_aggregate_qc_sha256", "")) != 64
        or sorted(authorization.get("nine_raw_transfer_receipt_sha256", {}))
        != expected_nine
    ):
        raise ValueError("full40 extension authorization content differs")
    return authorization


def generate(spec: Full40Case) -> Path:
    if spec.disposition != "planned_new_remainder_case":
        raise ValueError("the nine commissioning cases are validate-only and cannot be generated")
    target = CASES / spec.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {target}")
    source_time = SOURCE / f"{spec.source_time:g}"
    actual_hashes = {field: sha256(source_time / field) for field in STATE_FIELDS}
    if actual_hashes != spec.source_state_sha256:
        raise ValueError("source state changed after plan validation")
    points = action_points(spec.source_time, spec.action)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.", dir=CASES) as temporary:
        stage = Path(temporary) / spec.name
        stage.mkdir()
        shutil.copytree(SOURCE / "constant", stage / "constant")
        shutil.copytree(SOURCE / "system", stage / "system")
        shutil.copytree(source_time, stage / f"{spec.source_time:g}")
        shutil.copytree(source_time, stage / "source_restart_provenance")
        velocity = stage / f"{spec.source_time:g}" / "U"
        velocity.write_text(
            replace_rear_patch(velocity.read_text(encoding="utf-8"), points),
            encoding="utf-8",
        )
        control_path = stage / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        control = replace_once(
            control, "startTime 0;", f"startTime {spec.source_time:g};", control_path
        )
        control = replace_once(
            control,
            "endTime 160;",
            f"endTime {spec.source_time + DURATION:g};",
            control_path,
        )
        control = replace_once(control, "writeInterval 2;", "writeInterval 0.1;", control_path)
        control_path.write_text(control, encoding="utf-8")
        config = {
            "case": spec.name,
            "panel": "matched_start_acquisition_full40_v1",
            "split": spec.split,
            "phase_bin": spec.phase_bin,
            "source_restart_case": SOURCE.name,
            "source_restart_time": spec.source_time,
            "source_state_sha256": spec.source_state_sha256,
            "source_state_provenance_dir": "source_restart_provenance",
            "source_force_sha256": {
                force: sha256(SOURCE / f"postProcessing/{force}/0/coefficient.dat")
                for force in ("forceFront", "forceRear")
            },
            "phase_manifest": str(PHASE_MANIFEST.relative_to(REPO)),
            "phase_manifest_sha256": PHASE_MANIFEST_SHA256,
            "full40_predeclaration": str(PREDECLARATION.relative_to(REPO)),
            "full40_predeclaration_sha256": APPROVED_PREDECLARATION_SHA256,
            "full40_extension_authorization": str(
                EXTENSION_AUTHORIZATION.relative_to(REPO)
            ),
            "full40_extension_authorization_sha256": (
                APPROVED_EXTENSION_AUTHORIZATION_SHA256
            ),
            "action_target": spec.action,
            "action_points": points,
            "action_metrics": validate_action(points),
            "start_time": spec.source_time,
            "end_time": spec.source_time + DURATION,
            "analysis_window": [spec.source_time + 20.0, spec.source_time + DURATION],
            "delta_t": DELTA_T,
            "expected_solver_steps": 16000,
            "field_write_interval": WRITE_INTERVAL,
            "expected_field_frames": 801,
            "force_write_interval": DELTA_T,
            "expected_raw_solver_force_samples": 16000,
            "expected_aligned_force_samples_with_source_t0": 16001,
            "openfoam_image": IMAGE,
            "interpretation_guard": (
                "prescribed constant-rotation full40 acquisition; not closed-loop evidence"
            ),
        }
        (stage / "case_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-predeclaration", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    manifest, specs = load_plan()
    if args.write_predeclaration:
        if args.case or args.approval_token:
            parser.error("predeclaration mode cannot generate a case")
        write_json_exclusive_atomic(PREDECLARATION, predeclaration(manifest, specs))
        print(PREDECLARATION)
        return
    if args.list:
        for spec in specs:
            print(spec.name, spec.disposition)
        return
    if not args.case or args.approval_token != GENERATION_TOKEN:
        parser.error("one --case and the reviewed generation approval token are required")
    by_name = {spec.name: spec for spec in specs}
    if args.case not in by_name:
        parser.error("case is outside the frozen full40 matrix")
    validate_approved_predeclaration(manifest, specs)
    validate_extension_authorization(specs)
    validate_existing_nine(specs)
    print(generate(by_name[args.case]))


if __name__ == "__main__":
    main()
