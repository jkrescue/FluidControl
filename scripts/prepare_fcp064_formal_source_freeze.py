#!/usr/bin/env python3
"""Source-only P064 formal closure over the reviewed P026 numerical archive."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


BASE_SHA = "ea18970b030f640616a7ff9301f44d79643417e2b764264fd32b511f07aea7fe"
BASE_PATH = Path(__file__).with_name("prepare_fcp028_formal_source_freeze.py")
if BASE_PATH.is_symlink() or hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError("reviewed formal source materializer changed")
spec = importlib.util.spec_from_file_location("_p064_source_freeze_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def cpu_closure(experiment):
    if experiment != "FC-P064":
        raise ValueError("P064-only source profile")
    result = dict(base.CPU_CLOSURE)
    result.pop("scripts/audit_fcp028_candidate.py")
    result.pop("scripts/verify_fcp028_dual_reload.py")
    result.update({
        "scripts/audit_fcp064_candidate.py": "candidate_auditor",
        "scripts/verify_fcp064_dual_reload.py": "cpu_reload",
    })
    return result


def orchestration_sources(args):
    return {
        "scripts/run_fcp026_posteval.py": args.numerical_runner,
        "scripts/run_fcp028_posteval.py": args.repo / "scripts/run_fcp028_posteval.py",
        "scripts/flow_repair_profiles.py": args.repo / "scripts/flow_repair_profiles.py",
        "scripts/run_fcp064_posteval.py": args.formal_runner,
        "scripts/run_fcp064_formal_resource_adapter.py": args.formal_runner.with_name(
            "run_fcp064_formal_resource_adapter.py"
        ),
    }


base.cpu_closure = cpu_closure
base.orchestration_sources = orchestration_sources

if __name__ == "__main__":
    base.main("FC-P064")
