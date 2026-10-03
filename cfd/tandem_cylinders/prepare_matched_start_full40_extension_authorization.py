#!/usr/bin/env python3
"""Create the review candidate authorizing the 31-case full40 extension.

This can succeed only after the nine-case aggregate and all nine raw-transfer
receipts exist.  The emitted JSON does not enable execution until it is
independently committed and its exact SHA-256 is bound in execution code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import make_matched_start_full40 as full40

NINE_ROOT = REPO / "artifacts/matched_start_acquisition"
NINE_AGGREGATE = NINE_ROOT / "aggregate_qc/result.json"
NINE_RECEIPTS = NINE_ROOT / "transfer_verified"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_authorization(aggregate: Path = NINE_AGGREGATE) -> dict:
    manifest, specs = full40.load_plan()
    full40.validate_approved_predeclaration(manifest, specs)
    if not aggregate.is_file():
        raise FileNotFoundError(f"nine-case aggregate QC is absent: {aggregate}")
    report = json.loads(aggregate.read_text(encoding="utf-8"))
    expected_nine = sorted(
        spec.name
        for spec in specs
        if spec.disposition == "existing_nine_case_commissioning"
    )
    if (
        report.get("status") != "MATCHED_START_9_CASE_COMMISSIONING_QC_PASS"
        or report.get("complete_nine_case_panel") is not True
        or report.get("phase_manifest_sha256") != full40.PHASE_MANIFEST_SHA256
        or sorted(row.get("case") for row in report.get("cases", [])) != expected_nine
        or sorted(report.get("transfer", {})) != expected_nine
    ):
        raise ValueError("nine-case aggregate is not the exact complete commissioning pass")
    receipt_hashes = {}
    for name in expected_nine:
        receipt = NINE_RECEIPTS / f"{name}.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"missing nine-case raw receipt: {receipt}")
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if data.get("status") != "RAW_TRANSFER_VERIFIED" or data.get("case") != name:
            raise ValueError(f"invalid nine-case raw receipt: {name}")
        receipt_hashes[name] = sha256(receipt)
    remainder = sorted(
        spec.name for spec in specs if spec.disposition == "planned_new_remainder_case"
    )
    if len(remainder) != 31:
        raise ValueError("full40 remainder is not exactly 31 cases")
    return {
        "status": "MATCHED_START_FULL40_EXTENSION_AUTHORIZED",
        "scope": "31-case remainder only; the existing nine cases are excluded",
        "phase_manifest_sha256": full40.PHASE_MANIFEST_SHA256,
        "full40_predeclaration": str(full40.PREDECLARATION.relative_to(REPO)),
        "full40_predeclaration_sha256": full40.APPROVED_PREDECLARATION_SHA256,
        "nine_case_aggregate_qc": str(aggregate.relative_to(REPO)),
        "nine_case_aggregate_qc_sha256": sha256(aggregate),
        "nine_raw_transfer_receipt_sha256": receipt_hashes,
        "authorized_cases": remainder,
        "maximum_parallel_cases": 4,
        "resource_guards": {
            "worker_start_mem_available_gib_at_least": 64,
            "worker_running_mem_available_gib_at_least": 40,
            "worker_free_disk_gib_at_least": 100,
            "spark_free_disk_gib_at_least": 250,
        },
        "execution_guards": {
            "authorization_must_be_committed_before_sha_binding": True,
            "existing_nine_cases_must_not_be_generated_or_rerun": True,
            "maximum_parallel_cases": 4,
            "fail_stop_on_failed_or_orphaned_case": True,
        },
        "interpretation_guard": (
            "prescribed constant-rotation acquisition only; not closed-loop control "
            "or evidence of model sufficiency"
        ),
    }


def write_exclusive(path: Path, payload: dict) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_authorization()
    write_exclusive(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
