#!/usr/bin/env python3
"""Project P026-only formal orchestration; completion is never admission.

The Lead supplies a byte-bound approval AFTER independent terminal review and
actual official dual reload. This runner does not replace that scientific audit.
No historical experiment profiles, automatic training, PPO, or frozen-test path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

IMAGE = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
BASE = "7216214b545fbbd50b2fb5ed866f231039b06b18"
OVERLAYS = {
    "src/fluid_control/dual_fno.py",
    "src/fluid_control/calibrated_checkpoint.py",
    "scripts/evaluate_tandem_fno.py",
    "scripts/diagnose_fno_force_window.py",
    "scripts/audit_dev30_validation_diagnostic.py",
    "scripts/p026_history_inference.py",
    "scripts/p026_state_history.py",
}
PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6",
    "unchanged_development_gate",
]
UNCHANGED_AUDITS = {
    "scripts/audit_full40_validation_gate.py": "54ea35b3692143caedffd4a954dc789e36a2a3b157545d19a88b6058870f2809",
    "scripts/audit_dynamic_fno_development_gates.py": "ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412",
    "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py": "4e78d8473d1d0f93b25031a3bf9dcc0582f43604f7b1f67c65e6754f032af100",
}
REQUIRED_SOURCES = {
    "src/fluid_control/dual_fno.py",
    "src/fluid_control/calibrated_checkpoint.py",
    "scripts/evaluate_tandem_fno.py",
    "scripts/diagnose_fno_force_window.py",
    "scripts/p026_state_history.py",
    "scripts/p026_history_inference.py",
    "scripts/audit_dev30_validation_diagnostic.py",
    "scripts/audit_full40_validation_gate.py",
    "scripts/audit_dynamic_fno_development_gates.py",
    "scripts/spark_gpu_guard.py",
    "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py",
    "conf/tandem_fno_full40_h20.yaml",
}
DATA = {
    "devdata": "data/curated/tandem_cylinders_matched_start_full40_dev30_v1",
    "full40": "data/curated/tandem_cylinders_matched_start_full40_v1",
    "dynamic": "data/curated/tandem_cylinders_full40_dynamic_validation_v1",
}
INPUTS = {
    "devdata/manifest.json": "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2",
    "full40/manifest.json": "1c9086b4d08da57e84fb4f6a22376f0935596727952a50c55acb1a130e77e90e",
    "dynamic/manifest.json": "bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae",
    "devdata/normalization.json": "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1",
    "full40/normalization.json": "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1",
}
PREDECL = "artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
PREDECL_SHA = "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
QC = "artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json"
QC_SHA = "9723203f922cbe6609f2c92b7b48d299ee9d948d694d3c6c2eab413472421d86"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound(root, relative, expected):
    path = root / relative
    require(
        not path.is_symlink() and root.resolve() in path.resolve().parents,
        "path escape",
    )
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "invalid SHA")
    require(path.is_file() and sha(path) == expected, f"SHA mismatch: {relative}")
    return path


def fields(value, expected):
    require(
        all(value.get(k) == v for k, v in expected.items()),
        f"contract differs: {expected}",
    )


def terminal(unit, invocation):
    # This runner is intentionally Main-only: both matched arms are approved
    # sequential user units with RemainAfterExit=true. No remote/dead fallback.
    fields(
        unit,
        {
            "InvocationID": invocation,
            "ActiveState": "active",
            "SubState": "exited",
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        },
    )


def candidate_contract(root, approval):
    k = approval["history_k"]
    require(type(k) is int and k in (1, 4), "unsupported history")
    required = {
        "result.json",
        "training_protocol.json",
        "dual_model_manifest.json",
        "flow/FNO.0.0.mdlus",
        "flow/checkpoint.0.0.pt",
        "aerodynamic/FNO.0.1.mdlus",
        "aerodynamic/checkpoint.0.1.pt",
    }
    require(set(approval["candidate_sha256"]) == required, "candidate proof incomplete")
    for name, digest in approval["candidate_sha256"].items():
        bound(root, name, digest)
    result = json.loads((root / "result.json").read_text())
    manifest = json.loads((root / "dual_model_manifest.json").read_text())
    protocol = json.loads((root / "training_protocol.json").read_text())
    digest = approval["candidate_sha256"]["training_protocol.json"]
    fields(
        result,
        {
            "status": f"FC_P026_K{k}_TRAINING_COMPLETE_NOT_ADMISSION",
            "history_k": k,
            "optimizer_steps": 171,
            "training_windows": 1368,
            "protocol_sha256": digest,
            "official_fresh_reload_verified": True,
            "scientific_admission": False,
        },
    )
    fields(
        protocol,
        {
            "training_experiment": "FC-P026",
            "training_windows": 1368,
            "optimizer_steps": 171,
            "accumulation_windows": 8,
            "learning_rate": 1.5625e-7,
            "validation_accessed": False,
            "frozen_test_accessed": False,
            "selection_performed": False,
        },
    )
    fields(
        manifest,
        {
            "status": f"FC_P026_K{k}_DUAL_FNO_MANIFEST_VERIFIED",
            "kind": f"FC_P026_K{k}_HISTORY_FORCE_FNO",
            "training_experiment": "FC-P026",
            "training_protocol_sha256": digest,
        },
    )
    fields(
        manifest["history_input"],
        {
            "profile": f"p026_k{k}",
            "history_length": k,
            "flow_input_channels": 6,
            "aerodynamic_input_channels": 6 if k == 1 else 18,
            "future_state_inputs": False,
            "future_force_inputs": False,
        },
    )
    require(
        manifest["history_input"] == protocol["history_input"],
        "history protocol differs",
    )
    records = result["records"]
    require(len(records) == 171, "incomplete updates")
    require(
        [(r["update"], r["consumed_windows"]) for r in records]
        == [(i, i * 8) for i in range(1, 172)],
        "update sequence differs",
    )
    require(
        result["sampler_order_sha256"] == protocol["sampler_order_sha256"],
        "order differs",
    )
    for role, epoch in (("flow", 0), ("aerodynamic", 1)):
        fields(
            manifest[role],
            {
                "model_sha256": approval["candidate_sha256"][
                    f"{role}/FNO.0.{epoch}.mdlus"
                ],
                "state_sha256": approval["candidate_sha256"][
                    f"{role}/checkpoint.0.{epoch}.pt"
                ],
            },
        )
    return manifest


def validate_candidate_loader(source, candidate, approval):
    """Use the exact frozen pure-CPU manifest validator, without model creation."""
    path = source / "src/fluid_control/dual_fno.py"
    require(
        sha(path) == approval["source_sha256"]["src/fluid_control/dual_fno.py"],
        "frozen loader bytes differ",
    )
    name = "_p026_formal_frozen_manifest_validator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        identity = module.validate_dual_fno_manifest(
            candidate / "dual_model_manifest.json",
            expected_sha256=approval["candidate_sha256"]["dual_model_manifest.json"],
        )
        fields(
            identity.payload,
            {"kind": f"FC_P026_K{approval['history_k']}_HISTORY_FORCE_FNO"},
        )
        require(
            identity.aerodynamic.directory == (candidate / "aerodynamic").resolve(),
            "validated aerodynamic role directory differs",
        )
        require(
            identity.flow.directory == (candidate / "flow").resolve(),
            "validated flow role directory differs",
        )
        for relative, field in (
            ("scripts/p026_state_history.py", "history_state_module_sha256"),
            ("scripts/p026_history_inference.py", "history_inference_module_sha256"),
            ("training_config.yaml", "config_sha256"),
        ):
            require(
                sha(source / relative) == identity.payload[field],
                f"actual frozen runtime source differs: {relative}",
            )
        return identity
    finally:
        sys.modules.pop(name, None)


def archive_files(archive):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for entry in stream:
            if entry.isdir():
                continue
            require(
                entry.isfile() and entry.name not in files,
                "nonregular/duplicate base member",
            )
            require(
                not Path(entry.name).is_absolute()
                and ".." not in Path(entry.name).parts,
                "base archive path escape",
            )
            files[entry.name] = hashlib.sha256(
                stream.extractfile(entry).read()
            ).hexdigest()
    return files


def verify_chain_maps(receipt, approval, base_files, archive_sha):
    fields(
        receipt,
        {
            "schema_version": 1,
            "status": "FC_P026_FORMAL_SOURCE_CHAIN_FROZEN",
            "numerical_base_commit": BASE,
            "base_archive_sha256": archive_sha,
            "base_files_sha256": base_files,
            "training_config_sha256": approval["training_config_sha256"],
        },
    )
    overlays = receipt["overlay_sha256"]
    require(set(overlays) == OVERLAYS, "reviewed overlay closure differs")
    require(
        overlays == approval["reviewed_overlay_sha256"], "approval overlay bytes differ"
    )
    expected = dict(base_files)
    expected.update(overlays)
    require(
        receipt["final_files_sha256"] == expected == approval["source_sha256"],
        "non-overlay base file changed or source omitted",
    )


def validate_source_chain(args, approval):
    proof = approval["source_chain_receipt"]
    path = bound(args.repo, proof["path"], proof["sha256"])
    receipt = json.loads(path.read_text())
    archive = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.repo),
            "archive",
            "--format=tar",
            BASE,
            "src",
            "scripts",
            "conf",
            "cfd",
        ],
        timeout=60,
    )
    verify_chain_maps(
        receipt, approval, archive_files(archive), hashlib.sha256(archive).hexdigest()
    )


def commands(k, model, manifest):
    """Exact P018 numerical CLI; only explicit history identity is added."""
    dual = [
        "--dual-fno-manifest",
        "/workspace/dual/dual_model_manifest.json",
        "--expected-dual-fno-manifest-sha256",
        manifest,
        "--dual-training-config",
        "/workspace/training_config.yaml",
        "--fno-history-profile",
        f"p026_k{k}",
    ]
    common = [
        "--normalization-data",
        "/workspace/devdata",
        "--config",
        "/workspace/conf/tandem_fno_full40_h20.yaml",
        "--checkpoint-dir",
        "/workspace/dual/aerodynamic",
    ]
    guard = [
        "python",
        "-u",
        "scripts/spark_gpu_guard.py",
        "--min-free-gib",
        "20",
        "--allocator-fraction",
        ".15",
        "--margin-gib",
        "4",
        "--",
    ]
    result = []
    for name, data, stride, batch in (
        ("validation10", "devdata", "25", "4"),
        ("dynamic6", "dynamic", "1", "8"),
    ):
        args = (
            [
                "python",
                "-u",
                "scripts/evaluate_tandem_fno.py",
                "--data",
                f"/workspace/{data}",
            ]
            + common
            + dual
            + [
                "--split",
                "validation",
                "--horizons",
                "1",
                "10",
                "50",
                "100",
                "--segment-stride",
                stride,
                "--evaluation-batch-size",
                batch,
                "--action-mode",
                "observed",
                "--visualizations-per-horizon",
                "0",
                "--output",
                f"/workspace/output/{name}/evaluation.json",
                "--segment-metrics-output",
                f"/workspace/output/{name}/segments.json",
            ]
        )
        result.append((name, True, guard + args))
        if name == "validation10":
            result.append(
                (
                    "validation_diagnostic",
                    False,
                    [
                        "python",
                        "scripts/audit_dev30_validation_diagnostic.py",
                        "--report",
                        "/workspace/output/validation10/evaluation.json",
                        "--segments",
                        "/workspace/output/validation10/segments.json",
                        "--data",
                        "/workspace/devdata",
                        "--checkpoint-dir",
                        "/workspace/dual/aerodynamic",
                        "--candidate-kind",
                        f"FC_P026_K{k}_HISTORY_FORCE_FNO",
                        "--output",
                        "/workspace/output/validation10/diagnostic.json",
                    ],
                )
            )
            result.append(
                (
                    "endpoint_gate",
                    False,
                    [
                        "python",
                        "scripts/audit_full40_validation_gate.py",
                        "--report",
                        "/workspace/output/validation10/evaluation.json",
                        "--segments",
                        "/workspace/output/validation10/segments.json",
                        "--predeclaration",
                        "/workspace/predecl.json",
                        "--checkpoint-dir",
                        "/workspace/dual/aerodynamic",
                        "--data",
                        "/workspace/devdata",
                        "--config",
                        "/workspace/conf/tandem_fno_full40_h20.yaml",
                        "--image-id",
                        IMAGE,
                        "--output",
                        "/workspace/output/validation10/endpoint_gate.json",
                    ],
                )
            )
        else:
            result.append(
                (
                    "dynamic_diagnostic",
                    False,
                    [
                        "python",
                        "cfd/tandem_cylinders/audit_full40_dynamic6_fno.py",
                        "--data",
                        "/workspace/dynamic",
                        "--checkpoint",
                        "/workspace/dual/aerodynamic",
                        "--checkpoint-epoch",
                        "1",
                        "--expected-model-sha",
                        model,
                        "--report",
                        "/workspace/output/dynamic6/evaluation.json",
                        "--segments",
                        "/workspace/output/dynamic6/segments.json",
                        "--physical-qc",
                        "/workspace/physical_qc.json",
                        "--output",
                        "/workspace/output/dynamic6/diagnostic.json",
                    ],
                )
            )
    result.extend(
        [
            (
                "force_window",
                True,
                guard
                + [
                    "python",
                    "-u",
                    "scripts/diagnose_fno_force_window.py",
                    "--data",
                    "/workspace/dynamic",
                ]
                + common
                + dual
                + [
                    "--expected-model-sha",
                    model,
                    "--output",
                    "/workspace/output/force_window/result.json",
                ],
            ),
            (
                "development_gate",
                False,
                [
                    "python",
                    "scripts/audit_dynamic_fno_development_gates.py",
                    "--force-window",
                    "/workspace/output/force_window/result.json",
                    "--checkpoint-sha256",
                    model,
                    "--output",
                    "/workspace/output/development_gate.json",
                ],
            ),
        ]
    )
    return result


def validate_terminal_proofs(repo, candidate, approval):
    """Validate actual P026 receipt contents, not only Lead-reviewed file bytes.

    Receipt maps use training-root-relative ``candidate/`` names; approval maps
    use candidate-relative names. Only that one literal prefix is normalized.
    No inference of an arm, terminal success, model reload or scientific PASS.
    """
    k = approval["history_k"]
    require(type(k) is int and k in (1, 4), "terminal proof arm differs")
    files = approval["candidate_sha256"]
    expected_files = {
        "result.json", "training_protocol.json", "dual_model_manifest.json",
        "flow/FNO.0.0.mdlus", "flow/checkpoint.0.0.pt",
        "aerodynamic/FNO.0.1.mdlus", "aerodynamic/checkpoint.0.1.pt",
    }
    require(isinstance(files, dict) and set(files) == expected_files,
            "terminal proof candidate file set differs")

    def exact_fields(value, expected):
        require(isinstance(value, dict), "terminal proof object missing")
        for key, wanted in expected.items():
            actual = value.get(key)
            require(actual == wanted and
                    (type(wanted) not in (bool, int) or type(actual) is type(wanted)),
                    f"terminal proof field differs: {key}")

    def digest(value, label):
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                f"terminal proof digest invalid: {label}")
        return value

    proofs = {}
    for name in ("independent_terminal_audit", "official_dual_reload"):
        proof = approval[name]
        path = bound(repo, proof["path"], proof["sha256"])
        require(proof["reviewed_by_lead"] is True, "unreviewed terminal proof")
        proofs[name] = json.loads(path.read_text())
    audit, reload = proofs["independent_terminal_audit"], proofs["official_dual_reload"]
    common = dict(history_k=k, training_protocol_sha256=files["training_protocol.json"],
                  dual_manifest_sha256=files["dual_model_manifest.json"],
                  candidate_result_sha256=files["result.json"],
                  scientific_admission=False, ppo_authorized=False)
    for proof in (audit, reload):
        exact_fields(proof, common)
        mapping = proof.get("candidate_sha256")
        require(isinstance(mapping, dict) and set(mapping) ==
                {"candidate/" + name for name in expected_files},
                "terminal proof seven-file map/prefix differs")
        normalized = {name[len("candidate/"):]: digest(value, name)
                      for name, value in mapping.items()}
        require(normalized == files, "terminal proof candidate bytes differ")
    exact_fields(audit, dict(
        status=f"FC_P026_K{k}_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION",
        actual_optimizer_steps=171, actual_training_windows=1368, accumulation_windows=8,
        training_unit=approval["training_unit"], training_invocation=approval["training_invocation"],
        dual_adapter_fresh_reload_verified=False))
    require(isinstance(audit.get("terminal_evidence"), dict), "terminal unit evidence missing")
    terminal(audit["terminal_evidence"], approval["training_invocation"])
    exact_fields(audit["terminal_evidence"], dict(LoadState="loaded"))
    exact_fields(reload, dict(
        status=f"FC_P026_K{k}_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
        candidate_audit_sha256=approval["independent_terminal_audit"]["sha256"],
        config_sha256=approval["training_config_sha256"], required_official_image_id=IMAGE,
        device="cpu", official_dual_reload_verified=True, forward_performed=False,
        optimizer_created=False, model_saved=False, gpu_used=False))
    # These are already byte-bound by candidate_contract; recheck the small JSON
    # here so this helper is independently testable without reading model bytes.
    result_path = bound(candidate, "result.json", files["result.json"])
    result = json.loads(result_path.read_text())
    tensors = {"flow": digest(result.get("flow_tensor_sha256"), "result flow"),
               "aerodynamic": digest(result.get("aerodynamic_terminal_tensor_sha256"), "result aerodynamic")}
    require(audit.get("tensor_sha256") == reload.get("tensor_sha256") == tensors,
            "terminal/reloaded/saved tensor identity differs")
    loader = digest(audit.get("role_loader_sha256"), "audited role loader")
    require(isinstance(reload.get("runtime_source_sha256"), dict) and
            reload["runtime_source_sha256"].get("src/fluid_control/dual_fno.py") == loader ==
            approval["source_sha256"]["src/fluid_control/dual_fno.py"],
            "audited/reloaded/formal role loader differs")
    return proofs


def preflight(args):
    require(sha(args.approval) == args.approval_sha256, "formal approval SHA differs")
    a = json.loads(args.approval.read_text())
    fields(
        a,
        {
            "status": "FC_P026_APPROVED_ORIGINAL_FORMAL_EVALUATION",
            "formal_evaluation_authorized": True,
            "frozen_test_accessed": False,
            "ppo_auto_launch": False,
            "protocol": PROTOCOL,
            "numerical_base_commit": BASE,
            "official_image_id": IMAGE,
            "runner_sha256": sha(__file__),
        },
    )
    require(
        args.candidate.resolve()
        == (args.repo / a["candidate_relative_directory"]).resolve(),
        "candidate path differs",
    )
    require(
        args.output.resolve() == (args.repo / a["output_relative_directory"]).resolve(),
        "output path differs",
    )
    require(not args.output.exists(), "output must be new; no implicit resume")
    require(
        args.repo.resolve() in args.output.resolve().parents
        and args.repo.resolve() in args.candidate.resolve().parents,
        "output/candidate escapes repo",
    )
    sources = a["source_sha256"]
    validate_source_chain(args, a)
    require(REQUIRED_SOURCES <= set(sources), "source closure incomplete")
    fields(sources, UNCHANGED_AUDITS)
    actual = {
        str(p.relative_to(args.source))
        for d in ("src", "scripts", "conf", "cfd")
        for p in (args.source / d).rglob("*")
        if p.is_file()
    }
    require(actual == set(sources), "source tree extra/missing files")
    for name, digest in sources.items():
        bound(args.source, name, digest)
    bound(args.source, "training_config.yaml", a["training_config_sha256"])
    for name, digest in INPUTS.items():
        role, filename = name.split("/", 1)
        bound(args.repo, DATA[role] + "/" + filename, digest)
    bound(args.repo, PREDECL, PREDECL_SHA)
    bound(args.repo, QC, QC_SHA)
    candidate_contract(args.candidate, a)
    validate_candidate_loader(args.source, args.candidate, a)
    validate_terminal_proofs(args.repo, args.candidate, a)
    require(
        re.fullmatch(
            r"fluid-control-fcp026-history-k[14]-[a-zA-Z0-9-]+\.service",
            a["training_unit"],
        )
        is not None,
        "training unit differs",
    )
    require(f"-k{a['history_k']}-" in a["training_unit"], "unit arm differs")
    require(
        re.fullmatch(r"[0-9a-f]{32}", a["training_invocation"]) is not None,
        "invalid invocation",
    )
    raw = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            a["training_unit"],
            "-p",
            "InvocationID",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
            "-p",
            "ExecMainCode",
            "-p",
            "ExecMainStatus",
            "-p",
            "MainPID",
        ],
        text=True,
        timeout=10,
    )
    terminal(
        dict(line.split("=", 1) for line in raw.splitlines() if "=" in line),
        a["training_invocation"],
    )
    return a


def memory():
    rows = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return {
        key: int(rows[key].split()[0]) * 1024 for key in ("MemFree", "MemAvailable")
    }


def run_container(args, a, name, gpu, command, deadline):
    """Create before start: exact ID exists before any GPU work; own-ID cleanup."""
    output = args.output
    mounts = [
        (args.source / part, f"/workspace/{part}")
        for part in ("src", "scripts", "conf", "cfd")
    ]
    mounts += [
        (args.source / "training_config.yaml", "/workspace/training_config.yaml"),
        (args.candidate, "/workspace/dual"),
        (args.repo / PREDECL, "/workspace/predecl.json"),
        (args.repo / QC, "/workspace/physical_qc.json"),
    ]
    mounts += [
        (args.repo / DATA[part], f"/workspace/{part}")
        for part in ("devdata", "dynamic")
    ]
    if name == "endpoint_gate":
        mounts = [(p, target) for p, target in mounts if target != "/workspace/devdata"]
        mounts += [
            (args.repo / DATA["full40"] / item, f"/workspace/devdata/{item}")
            for item in ("validation", "manifest.json", "normalization.json")
        ]
    cmd = [
        "docker",
        "create",
        "--network",
        "none",
        "--cpus",
        "8",
        "--memory",
        "64g",
        "--shm-size",
        "2g",
        "--pids-limit",
        "512",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-w",
        "/workspace",
        "-e",
        "XDG_CACHE_HOME=/tmp/cache",
        "-e",
        "LOCAL_CACHE=/tmp/physicsnemo-cache",
        "-e",
        "WARP_CACHE_PATH=/tmp/warp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPATH=/workspace/src:/workspace/scripts",
    ]
    if gpu:
        cmd += ["--gpus", "device=0"]
    else:
        cmd += [
            "--runtime",
            "runc",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
        ]
    for path, target in mounts:
        cmd += ["-v", f"{path.resolve()}:{target}:ro"]
    cmd += ["-v", f"{output.resolve()}:/workspace/output:rw", IMAGE] + command
    cid = subprocess.check_output(cmd, text=True, timeout=60).strip()
    require(
        re.fullmatch(r"[0-9a-f]{64}", cid) is not None, "invalid created container ID"
    )
    process = None
    try:
        inspect = json.loads(
            subprocess.check_output(["docker", "inspect", cid], text=True, timeout=10)
        )[0]
        require(
            inspect["Image"] == IMAGE
            and any(
                m["Source"] == str(output.resolve())
                and m["Destination"] == "/workspace/output"
                for m in inspect["Mounts"]
            ),
            "created container identity differs",
        )
        (output / "evidence" / f"{name}_container.json").write_text(
            json.dumps(inspect, indent=2)
        )
        with (output / f"{name}.log").open("x") as log, (output / "memory.jsonl").open(
            "a"
        ) as mem:
            process = subprocess.Popen(
                ["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT
            )
            while True:
                row = memory()
                row.update(time_unix=time.time(), step=name)
                mem.write(json.dumps(row) + "\n")
                mem.flush()
                require(
                    min(row["MemFree"], row["MemAvailable"]) >= 20 * 1024**3,
                    "dual 20GiB guard",
                )
                require(time.monotonic() < deadline, "formal 3-hour deadline")
                if process.poll() is not None:
                    break
                time.sleep(2)
            require(process.returncode == 0, f"{name} failed")
        final = json.loads(
            subprocess.check_output(["docker", "inspect", cid], text=True, timeout=10)
        )[0]
        (output / "evidence" / f"{name}_container_terminal.json").write_text(
            json.dumps(final, indent=2)
        )
        require(
            final["State"]["ExitCode"] == 0
            and not final["State"]["OOMKilled"]
            and not final["State"]["Running"],
            "container terminal failure",
        )
    finally:
        # Removal targets only the exact ID returned by our own create. No other
        # running container is selected, and removal prevents a late start race.
        subprocess.run(
            ["docker", "rm", "-f", cid],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
        )
        if process is not None:
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "source", "candidate", "output", "approval"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    a = preflight(args)
    plan = commands(
        a["history_k"],
        a["candidate_sha256"]["aerodynamic/FNO.0.1.mdlus"],
        a["candidate_sha256"]["dual_model_manifest.json"],
    )
    if not args.execute:
        print(
            json.dumps(
                {"status": "FC_P026_FORMAL_PREFLIGHT_ONLY", "commands": plan}, indent=2
            )
        )
        return
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    require(not gpu.strip(), "GPU compute already active")
    require(min(memory().values()) >= 20 * 1024**3, "startup dual memory guard")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("validation10", "dynamic6", "force_window", "evidence"):
        (args.output / name).mkdir()
    (args.output / "evidence/formal_approval.json").write_bytes(
        args.approval.read_bytes()
    )

    def interrupt(signum, frame):
        raise RuntimeError(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    deadline = time.monotonic() + 10800
    precision = "import json,os,torch; x={'NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE'),'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),'float32_matmul_precision':torch.get_float32_matmul_precision()}; assert x=={'NVIDIA_TF32_OVERRIDE':None,'cuda_matmul_allow_tf32':True,'cudnn_allow_tf32':True,'float32_matmul_precision':'high'},x; print(json.dumps(x,sort_keys=True))"
    run_container(args, a, "precision", True, ["python", "-c", precision], deadline)
    for name, gpu, command in plan:
        run_container(args, a, name, gpu, command, deadline)
    for name in (
        "validation10/evaluation.json",
        "validation10/segments.json",
        "validation10/diagnostic.json",
        "validation10/endpoint_gate.json",
        "dynamic6/evaluation.json",
        "dynamic6/segments.json",
        "dynamic6/diagnostic.json",
        "force_window/result.json",
        "development_gate.json",
    ):
        require((args.output / name).is_file(), f"missing formal output: {name}")
    candidate_contract(args.candidate, a)
    outputs = {
        str(p.relative_to(args.output)): sha(p)
        for p in args.output.rglob("*")
        if p.is_file()
    }
    receipt = {
        "status": "FC_P026_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION",
        "history_k": a["history_k"],
        "formal_approval_sha256": args.approval_sha256,
        "candidate_sha256": a["candidate_sha256"],
        "source_sha256": a["source_sha256"],
        "protocol": PROTOCOL,
        "sha256": outputs,
        "scientific_admission": False,
        "ppo_auto_launched": False,
        "frozen_test_accessed": False,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
