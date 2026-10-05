from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_candidate_full40_canonical_ppo.py"
)
SPEC = importlib.util.spec_from_file_location("candidate_ppo_launcher", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

LEGACY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/train_full40_hydrogym_ppo_canonical.py"
)
LEGACY_SPEC = importlib.util.spec_from_file_location("legacy_canonical_ppo", LEGACY_SCRIPT)
LEGACY = importlib.util.module_from_spec(LEGACY_SPEC)
assert LEGACY_SPEC.loader is not None
LEGACY_SPEC.loader.exec_module(LEGACY)


def make_args(tmp_path: Path) -> SimpleNamespace:
    repo = tmp_path / "repo"
    candidate = repo / "artifacts/candidate"
    (candidate / "best").mkdir(parents=True)
    (candidate / "resolved_config.yaml").write_text("model: {}\n")
    data = repo / "full40"
    dynamic_data = repo / "dynamic6"
    dev30 = repo / "dev30"
    data.mkdir()
    dynamic_data.mkdir()
    dev30.mkdir()
    for path, content in (
        (data / "manifest.json", "full-manifest"),
        (data / "normalization.json", "normalization"),
        (dynamic_data / "manifest.json", "dynamic-manifest"),
        (dev30 / "normalization.json", "normalization"),
    ):
        path.write_text(content)
    evidence = {}
    for name in (
        "promotion_receipt", "baselines", "validation_report",
        "validation_segments", "predeclaration", "lineage", "posteval_receipt",
        "endpoint_gate", "window_gate", "dynamic_gate", "development_gate",
        "endpoint_evaluation_config",
    ):
        path = repo / f"{name}.json"
        path.write_text("{}\n")
        evidence[name] = path
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "audit_candidate_ppo_readiness.py").write_text("# readiness\n")
    (scripts / "train_full40_hydrogym_ppo_canonical.py").write_text("# trainer\n")
    (scripts / "run_candidate_full40_canonical_ppo_spark.sh").write_text("# launcher\n")
    return SimpleNamespace(
        repo=repo, candidate_root=candidate, data=data, dynamic_data=dynamic_data,
        dev30_data=dev30,
        output=repo / "artifacts/hydrogym/preflight.json", data_artifact=[],
        runtime_image_id=MODULE.RUNTIME_IMAGE_ID,
        official_image_id="sha256:official", episode_steps=100, timesteps=8192,
        checkpoint_interval=2048, gpu_memory_fraction=0.20, seed=20261003,
        approved_preflight=None, execute=False, **evidence,
    )


def readiness() -> dict:
    return {
        "status": "CANDIDATE_PPO_CPU_DRY_RUN_READY",
        "candidate_identity": {
            "candidate_kind": "fixture_h100",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_state_sha256": "b" * 64,
            "resolved_config_sha256": "c" * 64,
        },
    }


def p008_readiness() -> dict:
    value = readiness()
    value["candidate_identity"].update({
        "candidate_kind": "full_train_force_row_recalibration",
        "checkpoint_epoch": 0,
        "checkpoint_relative_directory": "candidate_build/candidate",
        "checkpoint_model_file": "FNO.0.0.mdlus",
        "checkpoint_state_file": "checkpoint.0.0.pt",
        "resolved_config_path": (
            "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/"
            "resolved_config.yaml"
        ),
        "precision_protocol": {
            "NVIDIA_TF32_OVERRIDE": None,
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
            "float32_matmul_precision": "high",
        },
    })
    return value


def test_command_uses_candidate_h100_and_preserves_legacy_contract(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    dry = MODULE.build_trainer_command(args, args.output, "dry-run")
    execute = MODULE.build_trainer_command(args, args.repo / "run", "execute")
    assert dry[dry.index("--config") + 1].endswith("candidate/resolved_config.yaml")
    assert dry[dry.index("--checkpoint-dir") + 1].endswith("candidate/best")
    assert dry[dry.index("--episode-steps") + 1] == "100"
    assert "--vecnormalize-output" not in dry
    assert execute[execute.index("--vecnormalize-output") + 1].endswith(
        "run/vecnormalize.pkl"
    )
    contract = MODULE.command_contract(args, readiness())
    assert contract["observation_dimensions"] == 69
    assert contract["reward_contract"] == "canonical_joint_v1 unchanged"
    assert contract["vecnormalize"] == {
        "norm_obs": False, "norm_reward": False,
        "reason": "preserve existing canonical PPO numerical contract",
    }


def test_p008_command_preserves_epoch_zero_identity_without_aliasing(
    tmp_path: Path,
) -> None:
    args = make_args(tmp_path)
    identity = p008_readiness()
    command = MODULE.build_trainer_command(args, args.output, "dry-run", identity)
    assert command[command.index("--checkpoint-dir") + 1].endswith(
        "candidate/candidate_build/candidate"
    )
    assert command[command.index("--config") + 1].endswith(
        "tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml"
    )
    assert "--allow-calibrated-epoch-zero" in command
    assert command[command.index("--expected-calibrated-model-sha256") + 1] == "a" * 64
    assert command[command.index("--expected-calibrated-state-sha256") + 1] == "b" * 64
    contract = MODULE.command_contract(args, identity)
    assert contract["checkpoint_epoch"] == 0
    assert contract["checkpoint_model_file"] == "FNO.0.0.mdlus"
    assert contract["checkpoint_state_file"] == "checkpoint.0.0.pt"
    assert contract["precision_protocol"]["float32_matmul_precision"] == "high"


def test_legacy_entry_keeps_no_vecnormalize_default() -> None:
    args = LEGACY.build_parser().parse_args([
        "--data", "data", "--config", "config", "--checkpoint-dir", "checkpoint",
        "--baselines", "baselines", "--image-id", "image",
        "--runtime-image-id", "runtime", "--output", "output", "--dry-run",
    ])
    assert args.vecnormalize_output is None
    assert args.allow_calibrated_epoch_zero is False


def test_calibrated_precision_requires_default_tf32_high(monkeypatch) -> None:
    torch_stub = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        get_float32_matmul_precision=lambda: "high",
    )
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    value = LEGACY.validate_calibrated_precision_protocol(torch_stub)
    assert value["float32_matmul_precision"] == "high"
    torch_stub.backends.cuda.matmul.allow_tf32 = False
    with pytest.raises(ValueError, match="default-TF32/high"):
        LEGACY.validate_calibrated_precision_protocol(torch_stub)


@pytest.mark.parametrize(
    ("field", "value"),
    (("episode_steps", 99), ("timesteps", 1024),
     ("checkpoint_interval", 1000), ("gpu_memory_fraction", 0.21),
     ("runtime_image_id", "sha256:wrong")),
)
def test_invalid_runtime_contract_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    args = make_args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        MODULE.validate_arguments(args)


def test_direct_host_execute_is_rejected(tmp_path: Path, monkeypatch) -> None:
    args = make_args(tmp_path)
    args.execute = True
    monkeypatch.delenv("CANDIDATE_PPO_RUNTIME_IMAGE_ID", raising=False)
    monkeypatch.delenv("CANDIDATE_PPO_GPU_GUARD_ACTIVE", raising=False)
    with pytest.raises(ValueError, match="direct-host execution forbidden"):
        MODULE.validate_arguments(args)


def test_dry_run_shell_rejects_execute_smuggling() -> None:
    shell = SCRIPT.with_name("run_candidate_full40_canonical_ppo_spark.sh")
    result = subprocess.run(
        [str(shell), "--dry-run", "--execute"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "cannot be smuggled" in result.stderr


def test_relative_paths_are_resolved_from_repo_not_caller_cwd(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    args.candidate_root = Path("artifacts/candidate")
    args.output = Path("artifacts/hydrogym/preflight.json")
    MODULE.normalize_paths(args)
    assert args.candidate_root == args.repo / "artifacts/candidate"
    assert args.output == args.repo / "artifacts/hydrogym/preflight.json"


def test_blocked_candidate_never_invokes_legacy_trainer(tmp_path: Path, monkeypatch) -> None:
    args = make_args(tmp_path)
    monkeypatch.setattr(
        MODULE.subprocess, "run",
        lambda *unused, **unused_kw: (_ for _ in ()).throw(AssertionError("invoked")),
    )
    blocked = {
        **readiness(), "status": "CANDIDATE_PPO_CPU_DRY_RUN_BLOCKED",
        "blockers": [{"kind": "MISSING"}],
    }
    result = MODULE.dry_run(args, blocked)
    assert result["status"].endswith("BLOCKED")
    assert result["training_executed"] is False
    assert result["policy_created"] is False


def test_ready_dry_run_requires_legacy_preflight_to_pass(
    tmp_path: Path, monkeypatch
) -> None:
    args = make_args(tmp_path)

    def fake_run(command, **unused):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({
            "status": "FULL40_CANONICAL_PPO_EXECUTION_READY",
            "checkpoint_sha256": "a" * 64,
        }))

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    result = MODULE.dry_run(args, readiness())
    assert result["status"].endswith("READY")
    assert result["training_executed"] is False
    assert result["command_contract"]["new_policy_required"] is True


@pytest.mark.parametrize("precision_valid", [True, False])
def test_execute_binds_new_policy_vecnormalize_and_p008_precision(
    tmp_path: Path, monkeypatch, precision_valid: bool
) -> None:
    args = make_args(tmp_path)
    candidate_readiness = p008_readiness()
    contract = MODULE.command_contract(args, candidate_readiness)
    approved = args.repo / "approved.json"
    approved.write_text(json.dumps({
        "status": "CANDIDATE_CANONICAL_PPO_DRY_RUN_READY",
        "candidate_readiness": candidate_readiness,
        "command_contract": contract,
        "canonical_preflight": {
            "status": "FULL40_CANONICAL_PPO_EXECUTION_READY",
            "checkpoint_sha256": "a" * 64,
        },
        "training_executed": False,
    }))
    args.approved_preflight = approved
    args.output = args.repo / "run"

    def fake_run(command, **unused):
        output = Path(command[command.index("--output") + 1])
        checkpoints = output / "checkpoints"
        checkpoints.mkdir(parents=True)
        policy = checkpoints / "ppo_00008192.zip"
        policy.write_bytes(b"policy")
        vec = output / "vecnormalize.pkl"
        vec.write_bytes(b"identity-vec")
        (output / "audit.json").write_text(json.dumps({
            "status": "FULL40_CANONICAL_PPO_SURROGATE_RUN_COMPLETE",
            "physicsnemo_checkpoint_sha256": "a" * 64,
            "precision_protocol": (
                candidate_readiness["candidate_identity"]["precision_protocol"]
                if precision_valid
                else {"float32_matmul_precision": "highest"}
            ),
            "vecnormalize_contract": (
                "identity: norm_obs=false, norm_reward=false; preserves legacy PPO numerics"
            ),
            "vecnormalize_sha256": MODULE.sha256(vec),
            "iterations": [{
                "checkpoint": str(policy),
                "checkpoint_sha256": MODULE.sha256(policy),
            }],
        }))

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    if not precision_valid:
        with pytest.raises(ValueError, match="audit contract differs"):
            MODULE.execute(args, candidate_readiness)
        return
    result = MODULE.execute(args, candidate_readiness)
    assert result["status"] == "CANDIDATE_CANONICAL_PPO_SURROGATE_RUN_BOUND"
    assert result["final_policy_sha256"] == MODULE.sha256(
        args.output / "checkpoints/ppo_00008192.zip"
    )
    assert result["real_cfd_validation_complete"] is False


def test_execute_rejects_stale_approved_preflight_before_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    args = make_args(tmp_path)
    approved = args.repo / "approved.json"
    approved.write_text(json.dumps({"status": "CANDIDATE_CANONICAL_PPO_DRY_RUN_READY"}))
    args.approved_preflight = approved
    args.output = args.repo / "run"
    monkeypatch.setattr(
        MODULE.subprocess, "run",
        lambda *unused, **unused_kw: (_ for _ in ()).throw(AssertionError("invoked")),
    )
    with pytest.raises(ValueError):
        MODULE.execute(args, readiness())
