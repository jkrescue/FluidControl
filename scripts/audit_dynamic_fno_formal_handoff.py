#!/usr/bin/env python3
"""Bind the existing endpoint Gate to a dynamic-candidate lineage receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def audit(
    repo: Path,
    candidate_root: Path,
    lineage_path: Path,
    endpoint_gate_path: Path,
    development_gate_path: Path | None = None,
) -> dict:
    repo = repo.resolve()
    lineage_module = module(
        repo / "scripts/audit_dynamic_fno_candidate_lineage.py",
        "dynamic_candidate_lineage",
    )
    recomputed = lineage_module.build(repo, candidate_root)
    stored = load(lineage_path)
    if stored != recomputed:
        raise ValueError("stored candidate lineage differs from recomputation")
    endpoint = load(endpoint_gate_path)
    if endpoint.get("checkpoint_sha256") != recomputed["checkpoint_sha256"]:
        raise ValueError("endpoint Gate checkpoint differs from candidate")
    if endpoint.get("data_manifest_sha256") != recomputed["data_lineage"][
        "formal_full40_manifest_sha256"
    ]:
        raise ValueError("endpoint Gate full40 manifest differs")
    if endpoint.get("normalization_sha256") != recomputed["data_lineage"][
        "normalization_sha256"
    ]:
        raise ValueError("endpoint Gate normalization differs")
    if endpoint.get("frozen_test_accessed") is not False:
        raise ValueError("endpoint Gate frozen-test contract differs")
    endpoint_pass = (
        endpoint.get("status") == "FULL40_VALIDATION_SURROGATE_READINESS_PASS"
        and endpoint.get("joint_terminal_readiness") is True
    )
    development = load(development_gate_path) if development_gate_path else None
    if development is not None:
        if (
            development.get("checkpoint_sha256") != recomputed["checkpoint_sha256"]
            or development.get("frozen_test_accessed") is not False
            or development.get("ppo_authorized") is not False
        ):
            raise ValueError("development Gate lineage differs")
    development_pass = bool(
        development
        and development.get("status") == "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS"
        and development.get("window_gate", {}).get("status") == "PASS"
        and development.get("dynamic_action_gate", {}).get("status") == "PASS"
    )
    missing = {} if development is not None else {
        "canonical_window_gate": {
            "status": "MISSING_CANDIDATE_SPECIFIC_DEVELOPMENT_EVIDENCE",
            "existing_scientific_quantities": [
                "2 percent final real-CFD drag reduction",
                "rear Cl-prime ratio <= 1.05",
                "absolute rear mean Cl / zero rear Cl-prime RMS <= 0.10",
            ],
            "historical_limitation": (
                "surrogate-vs-CFD paired window-error metrics and numerical pass "
                "thresholds were not part of the original predeclaration"
            ),
            "producer_created": True,
        },
        "dynamic_action_gate": {
            "status": "MISSING_CANDIDATE_SPECIFIC_DEVELOPMENT_EVIDENCE",
            "existing_contract": {
                "max_abs_omega": 0.75,
                "max_delta_omega": 0.1,
                "minimum_horizon_steps": 100,
            },
            "historical_limitation": (
                "the action constraints were fixed, but numerical predicted-response "
                "thresholds were not part of the original predeclaration"
            ),
            "producer_created": True,
        },
    }
    development_standard = {
        "status": "ADOPTED_2026-10-04_DEVELOPMENT_ADMISSION_NOT_ORIGINAL_GATE",
        "basis": "reserve half of each physical acceptance margin for surrogate discrimination",
        "thresholds": {
            "paired_total_drag_change_error": "<= 1 percent of zero total Cd",
            "rear_cl_prime_error": "<= 2.5 percent of same-window zero rear Cl-prime RMS",
            "rear_mean_cl_error": "<= 2.5 percent of same-window zero rear Cl-prime RMS",
            "dynamic_response": (
                "for fixed b01/b05 minus/zero/plus sequences: pooled total-Cd NRMSE "
                "<= 0.10, zero-relative delta-Cd MAE <= 0.023, non-tie sign accuracy "
                "= 1 and cross-action ordering accuracy = 1"
            ),
        },
        "warning": (
            "This is a development admission standard, not an original formal Gate or "
            "the final dense-60D/U real-CFD acceptance. It must be recomputed from the "
            "candidate's stepwise evidence rather than asserted by boolean placeholders."
        ),
    }
    return {
        "status": (
            "DYNAMIC_FNO_ENDPOINT_GATE_PASS_PPO_STILL_BLOCKED"
            if endpoint_pass
            else "DYNAMIC_FNO_ENDPOINT_GATE_FAIL_PPO_BLOCKED"
        ),
        "candidate_kind": recomputed["candidate_kind"],
        "checkpoint_sha256": recomputed["checkpoint_sha256"],
        "candidate_lineage_sha256": sha256(lineage_path),
        "endpoint_gate_sha256": sha256(endpoint_gate_path),
        "endpoint_gate_status": endpoint.get("status"),
        "endpoint_gate_pass": endpoint_pass,
        "development_gate_sha256": (
            sha256(development_gate_path) if development_gate_path else None
        ),
        "development_gate_pass": development_pass,
        "field_precision_interpretation": (
            "force/total-Cd accuracy and field-relative-L2 are separate metrics; "
            "a small drag error must not be described as percent-level full-field accuracy"
        ),
        "missing_required_gates": missing,
        "development_admission_standard": development_standard,
        "surrogate_ppo_episode_steps_required": 100,
        "surrogate_ppo_episode_duration_D_over_U": 10.0,
        "episode_reset": "each episode resets to a real curated frame-0 train/validation zero state",
        "real_cfd_final_evaluation_steps": 800,
        "real_cfd_final_evaluation_duration_D_over_U": 80.0,
        "candidate_specific_ppo_preflight_status": (
            "READY_FOR_SEPARATE_REVIEWED_PPO_LAUNCHER_IMPLEMENTATION"
            if endpoint_pass and development_pass
            else "BLOCKED"
        ),
        "ppo_authorized": False,
        "frozen_test_accessed": False,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--endpoint-gate", type=Path, required=True)
    parser.add_argument("--development-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = audit(
        args.repo,
        args.candidate_root,
        args.lineage,
        args.endpoint_gate,
        args.development_gate,
    )
    write_exclusive(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
