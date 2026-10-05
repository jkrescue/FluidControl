#!/usr/bin/env python3
"""Recover terminal evidence after systemd collected the successful P013 r2 unit.

This is a separate, bounded recovery; the original finalizer stays unchanged.
It audits persisted training only, and never authorizes scientific admission.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

UNIT = "fluid-control-fcp013-training-r2-20261005.service"
INVOCATION = "d0138175399a40f487493919a674e1a5"
CONTAINER_ID = "022a8fb1553b9355de39739e9f3e43647e2dd2e0ff6b876b8751ee2ac7522325"
CONTAINER_NAME = "fcp013-independent-force-training-r2-20261005"
IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
ROOT_NAME = "fcp013_independent_force_fno_training_r2_20261005"
OBSERVATION_SHA = "c6da08418d3e9c6287d50534fc669f560ed8d5d2a7f4237c90892e91655d1e6d"
CHAIN_NAME = "p013_posteval_chain_2e04cbe0cc18_immutable"
CHAIN_SHA = "f9810e70ca55d09d30f67d7112cc6300b5deaa16a1f6c8bb470e15a829775500"
AUDITOR_SHA = "cb554a59a0e25cbc41eb556e0ab09023f29f9268f006d6e9126958c5f721ad35"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collected_unit(fields):
    expected = {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead", "InvocationID": ""}
    if fields != expected:
        raise ValueError("unit is not the exact collected/inactive state")


def validate_docker_events(raw, *, since_seconds, until_seconds):
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    die, destroy = [], []
    for row in rows:
        if row.get("Type") != "container" or row.get("Actor", {}).get("ID") != CONTAINER_ID:
            raise ValueError("Docker event container identity differs")
        if row.get("Action") not in ("die", "destroy"):
            continue
        attributes = row["Actor"].get("Attributes", {})
        if attributes.get("image") != IMAGE or attributes.get("name") != CONTAINER_NAME:
            raise ValueError("Docker event image/name differs")
        stamp, nano = row.get("time"), row.get("timeNano")
        if (type(stamp) is not int or type(nano) is not int or nano // 10**9 != stamp
                or not since_seconds <= stamp <= until_seconds):
            raise ValueError("Docker event timestamp differs")
        if row["Action"] == "die":
            if attributes.get("exitCode") != "0":
                raise ValueError("Docker container did not exit successfully")
            die.append(row)
        else:
            destroy.append(row)
    if len(die) != 1 or len(destroy) != 1 or destroy[0]["timeNano"] <= die[0]["timeNano"]:
        raise ValueError("one successful die followed by destroy is required")
    return {"container_id": CONTAINER_ID, "image_id": IMAGE, "container_name": CONTAINER_NAME,
            "die_time": die[0]["time"], "die_time_nano": die[0]["timeNano"],
            "destroy_time_nano": destroy[0]["timeNano"], "exit_code": 0}


def validate_journal(raw, run_log, *, validate_guard, since_seconds, die_time):
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    messages, completions = [], []
    for row in rows:
        if row.get("_SYSTEMD_USER_UNIT") != UNIT or row.get("_SYSTEMD_INVOCATION_ID") != INVOCATION:
            raise ValueError("journal unit/invocation differs")
        message = row.get("MESSAGE")
        if not isinstance(message, str):
            raise ValueError("journal MESSAGE is not text")
        messages.append(message)
        try:
            value = json.loads(message)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("event") == "gpu_guard_complete":
            stamp = int(row["__REALTIME_TIMESTAMP"]) / 1_000_000
            if not since_seconds <= stamp <= die_time + 1:
                raise ValueError("journal guard timestamp is outside execution")
            completions.append(row)
    guard = validate_guard("\n".join(messages))
    if len(completions) != 1 or guard != validate_guard(run_log):
        raise ValueError("journal/run.log terminal guards differ")
    return {"unit": UNIT, "invocation_id": INVOCATION,
            "guard": guard, "journal_cursor": completions[0].get("__CURSOR"),
            "guard_realtime_timestamp": completions[0]["__REALTIME_TIMESTAMP"]}


def run(command):
    return subprocess.check_output(command, text=True, timeout=60)


def read_unit():
    command = ["systemctl", "--user", "show", UNIT]
    for key in ("LoadState", "ActiveState", "SubState", "InvocationID"):
        command.extend(("-p", key))
    return dict(line.split("=", 1) for line in run(command).splitlines() if "=" in line)


def require_container_absent():
    for condition in ("id=" + CONTAINER_ID, "name=^/" + CONTAINER_NAME + "$"):
        if run(["docker", "ps", "--all", "--no-trunc", "--filter", condition, "--format", "{{.ID}}"]).strip():
            raise ValueError("the training container ID/name still exists")


def persist_text(path, value):
    """Exclusive write, or verify identical bytes on an idempotent retry."""
    path = Path(path)
    if path.exists():
        if path.read_text() != value:
            raise ValueError("existing evidence differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_json(path, value, atomic_json):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("existing receipt differs: " + str(path))
    else:
        atomic_json(path, value)


def load_frozen_auditor(repo):
    chain = repo / "artifacts" / CHAIN_NAME
    if sha256(chain / "receipt.json") != CHAIN_SHA:
        raise ValueError("frozen audit chain receipt differs")
    receipt = json.loads((chain / "receipt.json").read_text())
    for relative, digest in receipt["sha256"].items():
        path = (chain / relative).resolve()
        if not path.is_relative_to(chain.resolve()) or sha256(path) != digest:
            raise ValueError("frozen audit dependency differs")
    path = chain / "scripts/audit_fcp013_dual_candidate.py"
    if sha256(path) != AUDITOR_SHA:
        raise ValueError("frozen candidate auditor differs")
    sys.path[:0] = [str(chain / "scripts"), str(chain / "numerical_source/src"),
                    str(chain / "numerical_source/scripts")]
    spec = importlib.util.spec_from_file_location("p013_collected_frozen_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    candidate = repo / "artifacts" / ROOT_NAME
    observation_path = candidate / "running_execution_evidence.json"
    if sha256(observation_path) != OBSERVATION_SHA:
        raise ValueError("immutable running observation differs")
    observation = json.loads(observation_path.read_text())
    if (observation["container_id"] != CONTAINER_ID or observation["image_id"] != IMAGE
            or observation["unit"]["InvocationID"] != INVOCATION):
        raise ValueError("observed runtime identity differs")
    since = observation["container_started_at"]
    since_seconds = int(datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp())
    current = read_unit()
    collected_unit(current)
    require_container_absent()
    evidence_dir = candidate / "collected_terminal_evidence"
    collection_path = evidence_dir / "collection.json"
    if collection_path.exists():
        collection = json.loads(collection_path.read_text())
        docker_raw = (evidence_dir / "docker_events.jsonl").read_text()
        journal_raw = (evidence_dir / "journal.jsonl").read_text()
        if collection["observation_sha256"] != OBSERVATION_SHA or collection["since"] != since:
            raise ValueError("saved collection identity differs")
    else:
        until = int(time.time())  # Never request a future event-stream deadline.
        docker_raw = run(["docker", "events", "--since", since, "--until", str(until),
                          "--filter", "container=" + CONTAINER_ID, "--format", "{{json .}}"])
        journal_raw = run(["journalctl", "--user", "-u", UNIT,
                           "_SYSTEMD_INVOCATION_ID=" + INVOCATION, "--no-pager", "-o", "json"])
        collection = {"since": since, "until": until, "observation_sha256": OBSERVATION_SHA}
    auditor = load_frozen_auditor(repo)
    docker = validate_docker_events(docker_raw, since_seconds=since_seconds, until_seconds=collection["until"])
    journal = validate_journal(journal_raw, (candidate / "run.log").read_text(),
                               validate_guard=auditor.validate_guard,
                               since_seconds=since_seconds, die_time=docker["die_time"])
    persist_text(evidence_dir / "docker_events.jsonl", docker_raw)
    persist_text(evidence_dir / "journal.jsonl", journal_raw)
    persist_json(collection_path, collection, auditor.atomic_json)
    audit = auditor.validate_candidate(repo, candidate)
    collected_unit(read_unit())
    require_container_absent()
    audit_path = candidate / "candidate_audit.json"
    persist_json(audit_path, audit, auditor.atomic_json)
    terminal = {
        "status": "COLLECTED_UNIT_DOCKER_AND_JOURNAL_TERMINAL_VERIFIED",
        "systemd_terminal_success_observed": False, "current_unit": current,
        "docker": docker, "journal": journal, "frozen_audit_chain_sha256": CHAIN_SHA,
        "sha256": {str(path.relative_to(candidate)): sha256(path) for path in
                   (observation_path, collection_path, evidence_dir / "docker_events.jsonl", evidence_dir / "journal.jsonl")},
    }
    receipt = {
        "status": "FC_P013_TRAINING_COMPLETE_NOT_ADMISSION", "execution_attempt": 2,
        "unit": UNIT, "terminal_unit": current, "terminal_evidence": terminal,
        "candidate_audit_sha256": sha256(audit_path), "finalizer_sha256": sha256(Path(__file__)),
        "optimizer_steps": audit["optimizer_steps"], "candidate_result_sha256": audit["candidate_result_sha256"],
        "dual_manifest_sha256": audit["dual_manifest_sha256"],
        "flow_model_sha256": audit["flow_model_sha256"], "flow_state_sha256": audit["flow_state_sha256"],
        "aerodynamic_model_sha256": audit["checkpoint_sha256"],
        "aerodynamic_state_sha256": audit["checkpoint_state_sha256"],
        "fixed_six_diagnostics_pending": True, "formal_evaluation_pending": True,
        "scientific_admission": False, "ppo_executed": False,
    }
    persist_json(candidate / "completion_receipt.json", receipt, auditor.atomic_json)
    print(json.dumps({"status": receipt["status"], "receipt_sha256": sha256(candidate / "completion_receipt.json")}))


if __name__ == "__main__":
    main()
