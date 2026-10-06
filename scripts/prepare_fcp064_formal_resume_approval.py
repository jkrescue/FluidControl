#!/usr/bin/env python3
"""Generate a non-authorizing P064 formal resume draft from completed R2 stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REUSED = (
    "precision.log", "evidence/precision_container.json", "evidence/precision_container_terminal.json",
    "validation10.log", "validation10/evaluation.json", "validation10/segments.json",
    "evidence/validation10_container.json", "evidence/validation10_container_terminal.json", "memory.jsonl",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    return json.loads(path.read_text())


def relative(repo, path):
    path = Path(path).resolve()
    require(path.is_relative_to(repo), "path escapes repo")
    return str(path.relative_to(repo))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "base-approval", "source-receipt", "freeze-receipt", "resume-runner",
                 "prior-output", "draft-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-unit", required=True)
    parser.add_argument("--prior-invocation", required=True)
    parser.add_argument("--planned-execution-unit", required=True)
    parser.add_argument("--output-relative-directory", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(not args.draft_output.exists(), "draft output exists")
    base = read(args.base_approval)
    require(base.get("status") == "FC_P064_APPROVED_ORIGINAL_FORMAL_EVALUATION"
            and base.get("formal_evaluation_authorized") is True, "base approval differs")
    freeze = read(args.freeze_receipt)
    source = read(args.source_receipt)
    require(freeze.get("status") == "FC_P064_FORMAL_SOURCE_FREEZE_COMPLETE"
            and source.get("status") == "FC_P026_FORMAL_SOURCE_CHAIN_FROZEN"
            and freeze.get("final_numerical_files_sha256") == source.get("final_files_sha256"),
            "v2 source closure differs")
    prior = args.prior_output.resolve()
    reused = {}
    for name in REUSED:
        path = prior / name
        require(path.is_file() and not path.is_symlink(), f"prior output absent: {name}")
        reused[name] = sha(path)
    for name in ("evidence/precision_container_terminal.json", "evidence/validation10_container_terminal.json"):
        state = read(prior / name)["State"]
        require(state["ExitCode"] == 0 and state["OOMKilled"] is False and state["Running"] is False,
                f"prior terminal stage differs: {name}")
    payload = dict(base)
    payload.update(
        status="FC_P064_FORMAL_RESUME_APPROVAL_PENDING_LEAD_REVIEW",
        intended_authorized_status="FC_P064_APPROVED_ORIGINAL_FORMAL_EVALUATION",
        formal_evaluation_authorized=False,
        reviewed_by_lead=False,
        output_relative_directory=args.output_relative_directory,
        source_chain_receipt={"path": relative(repo, args.source_receipt), "sha256": sha(args.source_receipt)},
        formal_source_freeze_receipt={"path": relative(repo, args.freeze_receipt), "sha256": sha(args.freeze_receipt)},
        reviewed_overlay_sha256=source["overlay_sha256"],
        source_sha256=source["final_files_sha256"],
        resume_runner_sha256=sha(args.resume_runner),
        planned_execution_unit=args.planned_execution_unit,
        resume_from={"relative_directory": relative(repo, prior), "unit": args.prior_unit,
                     "invocation": args.prior_invocation, "sha256": reused,
                     "completed_stages": ["precision", "validation10"]},
    )
    payload["generation_constraints"] = dict(payload["generation_constraints"], generator_authorizes_execution=False)
    args.draft_output.parent.mkdir(parents=True, exist_ok=True)
    args.draft_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "path": str(args.draft_output), "sha256": sha(args.draft_output)}))


if __name__ == "__main__":
    main()
