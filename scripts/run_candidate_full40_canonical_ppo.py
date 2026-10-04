#!/usr/bin/env python3
"""Candidate-aware wrapper for the existing canonical surrogate PPO trainer.

The default is a CPU-only dry-run.  Execution requires an immutable accepted
dry-run receipt and is still subject to every check in the legacy trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


RUNTIME_IMAGE_ID = "sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_readiness_module(repo: Path):
    path = repo / "scripts/audit_candidate_ppo_readiness.py"
    spec = importlib.util.spec_from_file_location("candidate_ppo_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command_contract(args, readiness: dict) -> dict:
    return {
        "candidate_kind": readiness.get("candidate_identity", {}).get("candidate_kind"),
        "checkpoint_sha256": readiness.get("candidate_identity", {}).get(
            "checkpoint_sha256"
        ),
        "checkpoint_state_sha256": readiness.get("candidate_identity", {}).get(
            "checkpoint_state_sha256"
        ),
        "resolved_config_sha256": readiness.get("candidate_identity", {}).get(
            "resolved_config_sha256"
        ),
        "candidate_readiness_implementation_sha256": sha256(
            args.repo / "scripts/audit_candidate_ppo_readiness.py"
        ),
        "canonical_trainer_sha256": sha256(
            args.repo / "scripts/train_full40_hydrogym_ppo_canonical.py"
        ),
        "container_launcher_sha256": sha256(
            args.repo / "scripts/run_candidate_full40_canonical_ppo_spark.sh"
        ),
        "full40_manifest_sha256": sha256(args.data / "manifest.json"),
        "full40_normalization_sha256": sha256(args.data / "normalization.json"),
        "canonical_baselines_sha256": sha256(args.baselines),
        "promotion_receipt_sha256": sha256(args.promotion_receipt),
        "validation_report_sha256": sha256(args.validation_report),
        "validation_segments_sha256": sha256(args.validation_segments),
        "predeclaration_sha256": sha256(args.predeclaration),
        "episode_steps": 100,
        "observation_dimensions": 69,
        "max_abs_omega": 0.75,
        "max_delta_omega": 0.1,
        "timesteps": args.timesteps,
        "checkpoint_interval": args.checkpoint_interval,
        "seed": args.seed,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "vecnormalize": {
            "norm_obs": False,
            "norm_reward": False,
            "reason": "preserve existing canonical PPO numerical contract",
        },
        "reward_contract": "canonical_joint_v1 unchanged",
        "new_policy_required": True,
    }


def build_trainer_command(args, output: Path, mode: str) -> list[str]:
    candidate = args.candidate_root.resolve()
    command = [
        sys.executable,
        "-u",
        str(args.repo / "scripts/train_full40_hydrogym_ppo_canonical.py"),
        "--data", str(args.data),
        "--config", str(candidate / "resolved_config.yaml"),
        "--checkpoint-dir", str(candidate / "best"),
        "--dev30-data", str(args.dev30_data),
        "--promotion-receipt", str(args.promotion_receipt),
        "--baselines", str(args.baselines),
        "--validation-gate", str(args.endpoint_gate),
        "--validation-report", str(args.validation_report),
        "--validation-segments", str(args.validation_segments),
        "--predeclaration", str(args.predeclaration),
        "--window-gate", str(args.window_gate),
        "--dynamic-gate", str(args.dynamic_gate),
        "--image-id", args.official_image_id,
        "--runtime-image-id", args.runtime_image_id,
        "--output", str(output),
        "--episode-steps", "100",
        "--timesteps", str(args.timesteps),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--gpu-memory-fraction", str(args.gpu_memory_fraction),
        "--seed", str(args.seed),
    ]
    if mode == "execute":
        command.extend(("--vecnormalize-output", str(output / "vecnormalize.pkl")))
    command.append("--execute" if mode == "execute" else "--dry-run")
    return command


def recompute_readiness(args) -> dict:
    module = load_readiness_module(args.repo)
    data_artifacts = module.parse_data_artifacts(args.data_artifact)
    data_artifacts = {
        key: path.resolve() if path.is_absolute() else (args.repo / path).resolve()
        for key, path in data_artifacts.items()
    }
    return module.audit(
        repo_root=args.repo,
        candidate_root=args.candidate_root,
        lineage_path=args.lineage,
        posteval_receipt_path=args.posteval_receipt,
        endpoint_gate_path=args.endpoint_gate,
        window_gate_path=args.window_gate,
        dynamic_gate_path=args.dynamic_gate,
        development_gate_path=args.development_gate,
        validation_manifest_path=args.data / "manifest.json",
        normalization_path=args.dev30_data / "normalization.json",
        data_artifacts=data_artifacts,
        official_image_id=args.official_image_id,
    )


def validate_arguments(args) -> None:
    if args.runtime_image_id != RUNTIME_IMAGE_ID:
        raise ValueError("HydroGym runtime image ID differs")
    if args.episode_steps != 100:
        raise ValueError("candidate surrogate PPO episode must be exactly H100")
    if args.timesteps < 2048 or args.timesteps % 512:
        raise ValueError("timesteps must be >=2048 and divisible by 512")
    if args.checkpoint_interval < 512 or args.timesteps % args.checkpoint_interval:
        raise ValueError("checkpoint interval must divide timesteps and be >=512")
    if not 0.0 < args.gpu_memory_fraction <= 0.20:
        raise ValueError("GPU memory fraction must be in (0, 0.20]")
    root = args.repo.resolve()
    for name in (
        "candidate_root", "lineage", "posteval_receipt", "endpoint_gate",
        "window_gate", "dynamic_gate", "development_gate", "data",
        "dev30_data", "promotion_receipt", "validation_report",
        "validation_segments", "predeclaration", "baselines", "output",
    ):
        path = getattr(args, name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} escapes repository") from error
    artifact_root = (root / "artifacts/hydrogym").resolve()
    try:
        args.output.resolve().relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("output must remain under artifacts/hydrogym") from error
    if args.approved_preflight is not None:
        try:
            args.approved_preflight.resolve().relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(
                "approved preflight must remain under artifacts/hydrogym"
            ) from error
    if args.execute and (
        not Path("/.dockerenv").is_file()
        or os.environ.get("CANDIDATE_PPO_RUNTIME_IMAGE_ID") != RUNTIME_IMAGE_ID
        or os.environ.get("CANDIDATE_PPO_GPU_GUARD_ACTIVE") != "1"
    ):
        raise ValueError(
            "direct-host execution forbidden; use the pinned guarded Spark launcher"
        )


def dry_run(args, readiness: dict) -> dict:
    receipt = {
        "status": "CANDIDATE_CANONICAL_PPO_DRY_RUN_BLOCKED",
        "candidate_readiness": readiness,
        "command_contract": None,
        "canonical_preflight": None,
        "training_executed": False,
        "policy_created": False,
        "vecnormalize_created": False,
        "frozen_test_accessed": False,
    }
    if readiness["status"] != "CANDIDATE_PPO_CPU_DRY_RUN_READY":
        return receipt
    contract = command_contract(args, readiness)
    with tempfile.TemporaryDirectory() as directory:
        canonical_output = Path(directory) / "canonical_preflight.json"
        command = build_trainer_command(args, canonical_output, "dry-run")
        subprocess.run(command, cwd=args.repo, check=True)
        canonical = load(canonical_output)
    if canonical.get("status") != "FULL40_CANONICAL_PPO_EXECUTION_READY":
        raise RuntimeError("legacy canonical PPO preflight is blocked")
    if canonical.get("checkpoint_sha256") != contract["checkpoint_sha256"]:
        raise ValueError("legacy preflight checkpoint differs from candidate")
    receipt.update(
        status="CANDIDATE_CANONICAL_PPO_DRY_RUN_READY",
        command_contract=contract,
        canonical_preflight=canonical,
    )
    return receipt


def execute(args, readiness: dict) -> dict:
    if args.approved_preflight is None:
        raise ValueError("--execute requires --approved-preflight")
    approved = load(args.approved_preflight)
    contract = command_contract(args, readiness)
    if (
        approved.get("status") != "CANDIDATE_CANONICAL_PPO_DRY_RUN_READY"
        or approved.get("candidate_readiness") != readiness
        or approved.get("command_contract") != contract
        or approved.get("canonical_preflight", {}).get("status")
        != "FULL40_CANONICAL_PPO_EXECUTION_READY"
        or approved.get("canonical_preflight", {}).get("checkpoint_sha256")
        != contract["checkpoint_sha256"]
        or approved.get("training_executed") is not False
    ):
        raise ValueError("approved candidate dry-run differs from current inputs")
    if readiness["status"] != "CANDIDATE_PPO_CPU_DRY_RUN_READY":
        raise RuntimeError("candidate readiness is blocked")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    command = build_trainer_command(args, args.output, "execute")
    subprocess.run(command, cwd=args.repo, check=True)
    audit_path = args.output / "audit.json"
    vec_path = args.output / "vecnormalize.pkl"
    audit = load(audit_path)
    iterations = audit.get("iterations")
    if (
        audit.get("status") != "FULL40_CANONICAL_PPO_SURROGATE_RUN_COMPLETE"
        or audit.get("physicsnemo_checkpoint_sha256") != contract["checkpoint_sha256"]
        or audit.get("vecnormalize_contract")
        != "identity: norm_obs=false, norm_reward=false; preserves legacy PPO numerics"
        or not isinstance(iterations, list)
        or not iterations
    ):
        raise ValueError("candidate PPO audit contract differs")
    final_policy = Path(iterations[-1]["checkpoint"]).resolve()
    try:
        final_policy.relative_to(args.output.resolve())
    except ValueError as error:
        raise ValueError("final policy escapes candidate output") from error
    if (
        not final_policy.is_file()
        or sha256(final_policy) != iterations[-1].get("checkpoint_sha256")
        or not vec_path.is_file()
        or sha256(vec_path) != audit.get("vecnormalize_sha256")
    ):
        raise ValueError("candidate policy/VecNormalize binding differs")
    receipt = {
        "status": "CANDIDATE_CANONICAL_PPO_SURROGATE_RUN_BOUND",
        "candidate_readiness_sha256": sha256(args.approved_preflight),
        "command_contract": contract,
        "canonical_audit_sha256": sha256(audit_path),
        "final_policy": str(final_policy.relative_to(args.output.resolve())),
        "final_policy_sha256": sha256(final_policy),
        "vecnormalize": "vecnormalize.pkl",
        "vecnormalize_sha256": sha256(vec_path),
        "training_executed": True,
        "evaluated_on_surrogate_only": True,
        "real_cfd_validation_complete": False,
        "frozen_test_accessed": False,
    }
    write_exclusive(args.output / "candidate_binding_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--posteval-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-gate", type=Path, required=True)
    parser.add_argument("--window-gate", type=Path, required=True)
    parser.add_argument("--dynamic-gate", type=Path, required=True)
    parser.add_argument("--development-gate", type=Path, required=True)
    parser.add_argument("--data-artifact", action="append", default=[])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dev30-data", type=Path, required=True)
    parser.add_argument("--promotion-receipt", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--validation-segments", type=Path, required=True)
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-preflight", type=Path)
    parser.add_argument("--official-image-id", default=None)
    parser.add_argument("--runtime-image-id", default=RUNTIME_IMAGE_ID)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--checkpoint-interval", type=int, default=2048)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--execute", action="store_true")
    return parser


def normalize_paths(args) -> None:
    args.repo = args.repo.resolve()
    for name in (
        "candidate_root", "lineage", "posteval_receipt", "endpoint_gate",
        "window_gate", "dynamic_gate", "development_gate", "data",
        "dev30_data", "promotion_receipt", "validation_report",
        "validation_segments", "predeclaration", "baselines", "output",
        "approved_preflight",
    ):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            setattr(args, name, (args.repo / path).resolve())


def main() -> None:
    args = build_parser().parse_args()
    normalize_paths(args)
    module = load_readiness_module(args.repo)
    if args.official_image_id is None:
        args.official_image_id = module.IMAGE_ID
    validate_arguments(args)
    readiness = recompute_readiness(args)
    result = execute(args, readiness) if args.execute else dry_run(args, readiness)
    if not args.execute:
        write_exclusive(args.output, result)
    print(result["status"])


if __name__ == "__main__":
    main()
