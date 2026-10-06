#!/usr/bin/env python3
"""Dry-run inventory and explicitly requested FC-P028 source materialization.

Without ``--execute`` this only reads source bytes and the reviewed git base.
The explicit execute mode creates one new immutable source root; neither mode
imports project code or accesses candidates/data.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


BASE = "7216214b545fbbd50b2fb5ed866f231039b06b18"
NUMERICAL_RUNNER_SHA256 = "03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3"
TRAINING_CONFIG_SHA256 = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
OVERLAYS = {
    "src/fluid_control/dual_fno.py": None,
    "src/fluid_control/calibrated_checkpoint.py": None,
    "scripts/evaluate_tandem_fno.py": None,
    "scripts/diagnose_fno_force_window.py": None,
    "scripts/audit_dev30_validation_diagnostic.py": "dev30_overlay",
    "scripts/p026_history_inference.py": None,
    "scripts/p026_state_history.py": None,
}
CPU_CLOSURE = {
    "scripts/flow_repair_profiles.py": None,
    "src/fluid_control/__init__.py": None,
    "src/fluid_control/dual_fno.py": None,
    "src/fluid_control/calibrated_checkpoint.py": None,
    "src/fluid_control/tandem_datapipe.py": None,
    "scripts/train_tandem_fno.py": None,
    "scripts/p026_history_inference.py": None,
    "scripts/p026_state_history.py": None,
    "scripts/audit_fcp028_candidate.py": "candidate_auditor",
    "scripts/verify_fcp028_dual_reload.py": "cpu_reload",
}


def cpu_closure(experiment):
    result = dict(CPU_CLOSURE)
    if experiment == "FC-P029":
        result.update({"scripts/audit_fcp029_candidate.py": None,
                       "scripts/verify_fcp029_dual_reload.py": None})
    elif experiment != "FC-P028":
        raise ValueError("unsupported source profile")
    return result


def orchestration_sources(args):
    result = {"scripts/run_fcp026_posteval.py": args.numerical_runner,
              "scripts/flow_repair_profiles.py": args.repo / "scripts/flow_repair_profiles.py"}
    if args.experiment == "FC-P029":
        result.update({"scripts/run_fcp029_posteval.py": args.formal_runner,
                       "scripts/run_fcp028_posteval.py": args.repo / "scripts/run_fcp028_posteval.py",
                       "scripts/flow_repair_profiles.py": args.repo / "scripts/flow_repair_profiles.py"})
    else:
        result["scripts/run_fcp028_posteval.py"] = args.formal_runner
    return result


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {path}")
    return path.resolve()


def archive_inventory(repo):
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", "--format=tar", BASE, "src", "scripts", "conf", "cfd"],
        timeout=60,
    )
    result = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for entry in stream:
            if entry.isdir():
                continue
            if not entry.isfile() or entry.name in result or Path(entry.name).is_absolute() or ".." in Path(entry.name).parts:
                raise ValueError("invalid base archive member")
            result[entry.name] = hashlib.sha256(stream.extractfile(entry).read()).hexdigest()
    return archive, hashlib.sha256(archive).hexdigest(), result


def selected_map(repo, sources, aliases):
    result = {}
    for relative, alias in sources.items():
        path = aliases[alias] if alias else repo / relative
        result[relative] = sha(regular(path))
    return result


def write_file(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def materialize(args, value, archive):
    original = args.output.absolute()
    for path in (original, *original.parents):
        if path.is_symlink():
            raise ValueError(f"output path contains a symlink: {path}")
    output = original
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent must already exist")
    temporary = Path(tempfile.mkdtemp(prefix=".p028-formal-source-", dir=output.parent))
    try:
        numerical = temporary / "numerical_source"
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
            for entry in stream:
                if entry.isdir():
                    continue
                if not entry.isfile() or Path(entry.name).is_absolute() or ".." in Path(entry.name).parts:
                    raise ValueError("invalid base archive member during materialization")
                write_file(numerical, entry.name, stream.extractfile(entry).read())
        aliases = {
            "dev30_overlay": args.dev30_overlay,
            "candidate_auditor": args.candidate_auditor,
            "cpu_reload": args.cpu_reload,
        }
        for relative, alias in OVERLAYS.items():
            source = regular(aliases[alias] if alias else args.repo / relative)
            target = numerical / relative
            target.write_bytes(source.read_bytes())
        write_file(numerical, "training_config.yaml", regular(args.training_config).read_bytes())
        source_receipt = {
            "schema_version": 1,
            "status": "FC_P026_FORMAL_SOURCE_CHAIN_FROZEN",
            "numerical_base_commit": BASE,
            "base_archive_sha256": value["base_archive_sha256"],
            "base_files_sha256": value["base_files_sha256"],
            "overlay_sha256": value["reviewed_overlay_sha256"],
            "final_files_sha256": value["final_numerical_files_sha256"],
            "training_config_sha256": value["training_config_sha256"],
        }
        write_file(
            numerical,
            "source_chain_receipt.json",
            json.dumps(source_receipt, indent=2, sort_keys=True).encode() + b"\n",
        )
        orchestration = temporary / "orchestration"
        for relative, source in orchestration_sources(args).items():
            write_file(orchestration, relative, regular(source).read_bytes())
        cpu = temporary / "cpu_reload_source"
        for relative, alias in cpu_closure(args.experiment).items():
            source = regular(aliases[alias] if alias else args.repo / relative)
            write_file(cpu, relative, source.read_bytes())
        observed_numerical = {
            relative: sha(numerical / relative)
            for relative in value["final_numerical_files_sha256"]
        }
        if observed_numerical != value["final_numerical_files_sha256"]:
            raise ValueError("materialized numerical source differs")
        observed_cpu = {
            relative: sha(cpu / relative)
            for relative in value["cpu_reload_source_sha256"]
        }
        if observed_cpu != value["cpu_reload_source_sha256"]:
            raise ValueError("materialized CPU source differs")
        observed_external = {relative: sha(orchestration / relative)
                             for relative in value["external_orchestration"]}
        if observed_external != value["external_orchestration"]:
            raise ValueError("materialized orchestration source differs")
        if sha(numerical / "training_config.yaml") != value["training_config_sha256"]:
            raise ValueError("materialized training config differs")
        if not (cpu / "src/fluid_control/__init__.py").is_file():
            raise ValueError("fluid_control package marker missing")
        write_file(
            cpu,
            "source_manifest.json",
            json.dumps(value["cpu_reload_source_sha256"], indent=2, sort_keys=True).encode()
            + b"\n",
        )
        receipt = dict(value)
        receipt.update(
            status=args.experiment.replace("-", "_") + "_FORMAL_SOURCE_FREEZE_COMPLETE",
            materialized=True,
            relative_roots={
                "numerical": "numerical_source",
                "orchestration": "orchestration",
                "cpu_reload": "cpu_reload_source",
            },
            import_namespace_contract={
                "pythonpath": ["cpu_reload_source/src", "cpu_reload_source/scripts"],
                "fluid_control_package_marker": "cpu_reload_source/src/fluid_control/__init__.py",
                "model_or_data_import_performed": False,
            },
        )
        write_file(
            temporary,
            "receipt.json",
            json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n",
        )
        for path in temporary.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(path, 0o555)
        os.chmod(temporary, 0o555)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("renameat2 is required for exclusive publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(temporary),
            -100,
            os.fsencode(output),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(output)
            raise OSError(error, os.strerror(error), output)
    except BaseException:
        if temporary.exists():
            os.chmod(temporary, 0o755)
            for path in temporary.rglob("*"):
                if path.is_dir():
                    os.chmod(path, 0o755)
                elif path.exists():
                    os.chmod(path, 0o644)
            shutil.rmtree(temporary)
        raise


def main(experiment="FC-P028"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dev30-overlay", type=Path, required=True)
    parser.add_argument("--formal-runner", type=Path, required=True)
    parser.add_argument("--candidate-auditor", type=Path, required=True,
                        help="Shared implementation audit_fcp028_candidate.py, NOT the thin P029 entry")
    parser.add_argument("--cpu-reload", type=Path, required=True,
                        help="Shared implementation verify_fcp028_dual_reload.py, NOT the thin P029 entry")
    parser.add_argument("--numerical-runner", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.experiment = experiment
    if experiment == "FC-P029":
        if (args.candidate_auditor.name != "audit_fcp028_candidate.py"
                or args.cpu_reload.name != "verify_fcp028_dual_reload.py"
                or args.formal_runner.name != "run_fcp029_posteval.py"):
            raise ValueError("P029 requires shared P028 audit/reload implementations and explicit P029 formal entry")
    repo = args.repo.resolve()
    aliases = {
        "dev30_overlay": args.dev30_overlay,
        "candidate_auditor": args.candidate_auditor,
        "cpu_reload": args.cpu_reload,
    }
    archive, archive_sha, base_files = archive_inventory(repo)
    overlays = selected_map(repo, OVERLAYS, aliases)
    cpu = selected_map(repo, cpu_closure(experiment), aliases)
    numerical_runner = regular(args.numerical_runner)
    if sha(numerical_runner) != NUMERICAL_RUNNER_SHA256:
        raise ValueError("external numerical runner SHA differs")
    regular(args.formal_runner)
    training_config = regular(args.training_config)
    if sha(training_config) != TRAINING_CONFIG_SHA256:
        raise ValueError("training config SHA differs")
    value = {
        "status": experiment.replace("-", "_") + "_FORMAL_SOURCE_FREEZE_DRY_RUN_ONLY",
        "materialized": False,
        "models_or_data_accessed": False,
        "numerical_base_commit": BASE,
        "base_archive_sha256": archive_sha,
        "base_file_count": len(base_files),
        "base_files_sha256": base_files,
        "reviewed_overlay_sha256": overlays,
        "final_numerical_files_sha256": {**base_files, **overlays},
        "external_orchestration": {relative: sha(regular(source))
                                   for relative, source in orchestration_sources(args).items()},
        "cpu_reload_source_sha256": cpu,
        "training_config_sha256": sha(training_config),
        "future_materialization_contract": [
            "create a new exclusive root after canonical integration",
            "write the exact git archive for the pinned base",
            "overlay exactly the seven reviewed numerical files",
            "store the external orchestration pair outside the numerical file map",
            "store the exact CPU-reload closure and immutable SHA receipt",
            "verify every member before chmod 0444/0555",
        ],
    }
    if args.execute:
        if args.output is None:
            raise ValueError("--execute requires --output")
        materialize(args, value, archive)
        value["status"] = experiment.replace("-", "_") + "_FORMAL_SOURCE_FREEZE_COMPLETE"
        value["materialized"] = True
        value["output"] = str(args.output.resolve())
    elif args.output is not None:
        raise ValueError("--output is only valid with --execute")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
