#!/usr/bin/env python3
"""Strictly gated full40 PhysicsNeMo-FNO + HydroGym canonical PPO.

The command never reads or enumerates the frozen split. ``--execute`` is
fail-closed until endpoint, causal-window, and dynamic-action validation
artifacts all bind to the exact full40 model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from functools import partial
from pathlib import Path

from fluid_control.canonical_joint_v1 import (
    ACTION_LIMIT,
    MIN_EPISODE_STEPS,
    evaluate_canonical_episode,
    validate_baseline,
    validate_full40_action_contract,
)

PROFILE = "matched_start_full40_v1"
VALIDATION_GATE_STATUS = "FULL40_VALIDATION_SURROGATE_READINESS_PASS"
WINDOW_GATE_STATUS = "FULL40_VALIDATION_CANONICAL_WINDOW_FIDELITY_PASS"
DYNAMIC_GATE_STATUS = "FULL40_VALIDATION_DYNAMIC_ACTION_PASS"
EXPECTED_COUNTS = {"train": 20, "validation": 10, "frozen_test": 10}
TRAIN_PHASES = ("00", "02", "04", "06")
VALIDATION_PHASES = ("01", "05")
HYDROGYM_RUNTIME_IMAGE_ID = (
    "sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c"
)
TRAIN_CASES = tuple(
    f"matched_start_acquisition_train_b{phase}_zero" for phase in TRAIN_PHASES
)
VALIDATION_CASES = tuple(
    f"matched_start_acquisition_validation_b{phase}_zero"
    for phase in VALIDATION_PHASES
)


class TrainingDiagnosticsAccumulator:
    """Collect train-only SB3/Monitor diagnostics between immutable checkpoints."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.applied_actions: list[float] = []
        self.reward_component_sums: dict[str, float] = {}
        self.reward_component_counts: dict[str, int] = {}
        self.rate_limited_steps = 0
        self.environment_steps = 0

    def record(self, infos) -> None:
        for info in infos:
            if not isinstance(info, dict):
                raise TypeError("SB3 training info must be a mapping")
            action = float(info["applied_omega"])
            if not math.isfinite(action):
                raise FloatingPointError("non-finite applied action in training")
            self.applied_actions.append(action)
            self.environment_steps += 1
            self.rate_limited_steps += int(bool(info.get("rate_limited", False)))
            for key, value in info.items():
                if key.startswith("reward_"):
                    number = float(value)
                    if not math.isfinite(number):
                        raise FloatingPointError(f"non-finite training {key}")
                    self.reward_component_sums[key] = (
                        self.reward_component_sums.get(key, 0.0) + number
                    )
                    self.reward_component_counts[key] = (
                        self.reward_component_counts.get(key, 0) + 1
                    )
            episode = info.get("episode")
            if episode is not None:
                episode_return = float(episode["r"])
                episode_length = int(episode["l"])
                if not math.isfinite(episode_return) or episode_length < 1:
                    raise ValueError("invalid SB3 Monitor episode diagnostics")
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "minimum": None, "maximum": None}
        return {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    def snapshot(self, *, reset: bool = False) -> dict:
        actions = self.applied_actions
        result = {
            "environment_steps": self.environment_steps,
            "episodes_completed": len(self.episode_returns),
            "episode_return": self._summary(self.episode_returns),
            "episode_length": self._summary(
                [float(value) for value in self.episode_lengths]
            ),
            "applied_omega": {
                **self._summary(actions),
                "rms": (
                    math.sqrt(sum(value * value for value in actions) / len(actions))
                    if actions
                    else None
                ),
            },
            "rate_limited_fraction": (
                self.rate_limited_steps / self.environment_steps
                if self.environment_steps
                else None
            ),
            "canonical_reward_component_mean_per_environment_step": {
                key: self.reward_component_sums[key]
                / self.reward_component_counts[key]
                for key in sorted(self.reward_component_sums)
            },
            "diagnostic_scope": (
                "train-only surrogate learning diagnostics; not validation, model "
                "selection, real-CFD evidence, or physical improvement"
            ),
        }
        if reset:
            self.reset()
        return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
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


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_gate_module():
    path = Path(__file__).with_name("audit_full40_validation_gate.py")
    spec = importlib.util.spec_from_file_location("full40_validation_gate_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load full40 validation gate implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_promotion_module():
    path = Path(__file__).with_name("verify_dev30_full40_promotion.py")
    spec = importlib.util.spec_from_file_location("dev30_promotion_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dev30/full40 promotion verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_dev30_promotion_receipt(
    path: Path,
    *,
    dev30_data: Path,
    full40_data: Path,
    predeclaration: Path,
) -> dict:
    """Recompute promotion identity and bind both releases byte-for-byte."""
    module = _load_promotion_module()
    recomputed = module.verify(
        dev30_data.resolve(), full40_data.resolve(), predeclaration.resolve()
    )
    stored = read_json(path)
    if stored != recomputed:
        raise ValueError("stored dev30 promotion receipt differs from recomputation")
    required = {
        "status": "DEV30_FULL40_PROMOTION_PASS",
        "formal_gate_input_authorized": True,
        "ppo_identity_prerequisite_passed": True,
        "frozen_hdf_opened_or_enumerated": False,
    }
    if any(recomputed.get(key) != value for key, value in required.items()):
        raise ValueError("dev30/full40 promotion identity did not pass")
    dev_manifest = dev30_data / "manifest.json"
    full_manifest = full40_data / "manifest.json"
    dev_normalization = dev30_data / "normalization.json"
    full_normalization = full40_data / "normalization.json"
    normalization_sha = recomputed.get("details", {}).get("normalization", {}).get(
        "sha256"
    )
    if (
        sha256(dev_normalization) != normalization_sha
        or sha256(full_normalization) != normalization_sha
    ):
        raise ValueError("promotion normalization SHA binding differs")
    return {
        "promotion_receipt_sha256": sha256(path),
        "promotion_verifier_sha256": sha256(
            Path(module.__file__).resolve()
        ),
        "dev30_manifest_sha256": sha256(dev_manifest),
        "full40_manifest_sha256": sha256(full_manifest),
        "dev30_normalization_sha256": sha256(dev_normalization),
        "full40_normalization_sha256": sha256(full_normalization),
    }


def validate_evidence_gate(
    path: Path,
    *,
    expected_status: str,
    checkpoint_sha256: str,
) -> dict:
    gate = read_json(path)
    required = {
        "status": expected_status,
        "profile": PROFILE,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_test_accessed": False,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise ValueError(f"{path.name} {key} differs")
    producer = Path(str(gate.get("producer_script", "")))
    evidence = Path(str(gate.get("evidence_path", "")))
    if not producer.is_file() or sha256(producer) != gate.get("producer_script_sha256"):
        raise ValueError(f"{path.name} producer is not cryptographically bound")
    if not evidence.is_file() or sha256(evidence) != gate.get("evidence_sha256"):
        raise ValueError(f"{path.name} evidence is not cryptographically bound")
    if gate.get("validation_phases") != ["b01", "b05"]:
        raise ValueError(f"{path.name} validation phases differ")
    if expected_status == WINDOW_GATE_STATUS:
        window_required = {
            "causal_window_seconds": 6.15,
            "total_drag_window_fidelity_pass": True,
            "rear_cl_fluctuation_window_fidelity_pass": True,
            "rear_cl_mean_bias_window_fidelity_pass": True,
        }
        if any(gate.get(key) != value for key, value in window_required.items()):
            raise ValueError(f"{path.name} canonical window metrics did not pass")
    elif expected_status == DYNAMIC_GATE_STATUS:
        dynamic_required = {
            "max_abs_omega": 0.75,
            "max_delta_omega": 0.1,
            "minimum_horizon_steps": 100,
            "dynamic_action_validation_pass": True,
        }
        if any(gate.get(key) != value for key, value in dynamic_required.items()):
            raise ValueError(f"{path.name} dynamic action contract did not pass")
    return gate


def validate_baselines(
    path: Path, *, required_phases: tuple[str, ...] | None = None
) -> dict[str, dict]:
    document = read_json(path)
    if document.get("status") != "FULL40_CANONICAL_ZERO_BASELINES":
        raise ValueError("canonical baseline artifact status differs")
    if (
        document.get("profile") != PROFILE
        or document.get("frozen_test_accessed") is not False
    ):
        raise ValueError("canonical baseline scope differs")
    phases = document.get("phases")
    expected_phases = set(
        required_phases
        or (
            *(f"b{phase}" for phase in TRAIN_PHASES),
            *(f"b{phase}" for phase in VALIDATION_PHASES),
        )
    )
    if not isinstance(phases, dict) or set(phases) != expected_phases:
        raise ValueError("canonical baseline phases differ")
    result = {}
    for phase, row in phases.items():
        expected_split = "train" if phase[1:] in TRAIN_PHASES else "validation"
        expected_case = f"matched_start_acquisition_{expected_split}_{phase}_zero"
        if row.get("split") != expected_split or row.get("case") != expected_case:
            raise ValueError(f"baseline identity differs for {phase}")
        source = Path(str(row.get("source_evidence", "")))
        if not source.is_file() or sha256(source) != row.get("source_evidence_sha256"):
            raise ValueError(f"baseline evidence is not bound for {phase}")
        result[phase] = validate_baseline(
            {
                "total_drag": row.get("total_drag"),
                "rear_cl_fluctuation_rms": row.get("rear_cl_fluctuation_rms"),
                "source": f"{source}:{expected_case}:predeclared_final_60D/U",
            }
        )
    return result


def validate_train20_baselines(path: Path) -> dict[str, dict]:
    """Read only the committed train20 zero branches for software commissioning."""
    document = read_json(path)
    scope = document.get("scope", {})
    if document.get("status") != "FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY":
        raise ValueError("train-only baseline summary status differs")
    if (
        scope.get("split") != "train"
        or scope.get("validation_or_frozen_results_read") is not False
        or scope.get("phase_bins") != [0, 2, 4, 6]
    ):
        raise ValueError("train-only baseline summary scope differs")
    phases = document.get("phases")
    expected = tuple(f"b{phase}" for phase in TRAIN_PHASES)
    if not isinstance(phases, dict) or set(phases) != set(expected):
        raise ValueError("train-only baseline phases differ")
    result = {}
    for phase in expected:
        row = phases[phase].get("same_phase_zero", {})
        expected_case = f"matched_start_acquisition_train_{phase}_zero"
        if row.get("case") != expected_case:
            raise ValueError(f"train-only zero identity differs for {phase}")
        metrics = row.get("metrics", {})
        result[phase] = validate_baseline(
            {
                "total_drag": metrics.get("mean_cd_total"),
                "rear_cl_fluctuation_rms": metrics.get(
                    "rms_cl_rear_fluctuation"
                ),
                "source": f"{path}:{phase}.same_phase_zero:predeclared_final_60D/U",
            }
        )
    return result


def smoke_preflight(
    *,
    data: Path,
    config: Path,
    checkpoint_dir: Path,
    baselines: Path,
    image_id: str,
    episode_steps: int,
) -> dict:
    """Permit train-only software commissioning without a validation claim."""
    blockers: list[str] = []
    action_contract = None
    checkpoint_sha = None
    try:
        manifest = read_json(data / "manifest.json")
        action_contract = validate_full40_action_contract(manifest)
        if manifest.get("trajectory_counts") != EXPECTED_COUNTS:
            raise ValueError("full40 split counts differ")
        if not (data / "normalization.json").is_file():
            raise ValueError("normalization is missing")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"full40_data_invalid:{error}")
    if not config.is_file():
        blockers.append("physicsnemo_config_missing")
    if image_id != HYDROGYM_RUNTIME_IMAGE_ID:
        blockers.append("hydrogym_runtime_image_id_differs")
    models = sorted(checkpoint_dir.glob("FNO.*.mdlus"))
    if len(models) != 1:
        blockers.append("checkpoint_generation_count_not_one")
    else:
        checkpoint_sha = sha256(models[0])
    if episode_steps < MIN_EPISODE_STEPS:
        blockers.append(f"episode_steps_below_{MIN_EPISODE_STEPS}")
    for case in TRAIN_CASES:
        if not (data / "train" / f"{case}.h5").is_file():
            blockers.append(f"missing_train_zero_case:{case}")
    try:
        baseline_rows = validate_train20_baselines(baselines)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"canonical_train_baselines_invalid:{error}")
        baseline_rows = None
    return {
        "status": (
            "FULL40_CANONICAL_PPO_TRAIN_ONLY_SMOKE_READY"
            if not blockers
            else "FULL40_CANONICAL_PPO_TRAIN_ONLY_SMOKE_BLOCKED"
        ),
        "profile": PROFILE,
        "training_executed": False,
        "action_contract": action_contract,
        "episode_steps": episode_steps,
        "train_cases": list(TRAIN_CASES),
        "validation_cases_opened": [],
        "frozen_test_directory_enumerated_or_opened": False,
        "checkpoint_sha256": checkpoint_sha,
        "baseline_phases_verified": sorted(baseline_rows) if baseline_rows else [],
        "baseline_artifact_sha256": sha256(baselines) if baseline_rows else None,
        "blockers": blockers,
        "scientific_scope": (
            "train-only software commissioning; no validation, real-CFD, control, "
            "or physical-benefit claim"
        ),
    }


def preflight(
    *,
    data: Path,
    config: Path,
    checkpoint_dir: Path,
    dev30_data: Path,
    promotion_receipt: Path,
    validation_gate: Path,
    validation_report: Path,
    validation_segments: Path,
    predeclaration: Path,
    window_gate: Path,
    dynamic_gate: Path,
    baselines: Path,
    image_id: str,
    runtime_image_id: str,
    episode_steps: int,
) -> dict:
    blockers: list[str] = []
    action_contract = None
    checkpoint_sha = None
    baseline_rows = None
    evidence_gate_sha256 = {}
    promotion_lineage = None
    try:
        manifest = read_json(data / "manifest.json")
        action_contract = validate_full40_action_contract(manifest)
        if manifest.get("trajectory_counts") != EXPECTED_COUNTS:
            raise ValueError("full40 split counts differ")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"full40_manifest_invalid:{error}")
    if not config.is_file():
        blockers.append("physicsnemo_config_missing")
    if runtime_image_id != HYDROGYM_RUNTIME_IMAGE_ID:
        blockers.append("hydrogym_runtime_image_id_differs")
    if episode_steps < MIN_EPISODE_STEPS:
        blockers.append(f"episode_steps_below_{MIN_EPISODE_STEPS}")
    for split, cases in (("train", TRAIN_CASES), ("validation", VALIDATION_CASES)):
        for case in cases:
            if not (data / split / f"{case}.h5").is_file():
                blockers.append(f"missing_{split}_zero_case:{case}")
    try:
        promotion_lineage = validate_dev30_promotion_receipt(
            promotion_receipt,
            dev30_data=dev30_data,
            full40_data=data,
            predeclaration=predeclaration,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"dev30_promotion_receipt_invalid:{error}")
    try:
        gate_module = _load_gate_module()
        recomputed = gate_module.audit(
            validation_report,
            validation_segments,
            predeclaration,
            checkpoint_dir,
            data,
            config,
            image_id,
        )
        stored = read_json(validation_gate)
        if stored != recomputed:
            raise ValueError("stored validation gate differs from recomputation")
        if recomputed.get("status") != VALIDATION_GATE_STATUS:
            raise ValueError("full40 validation endpoint gate did not pass")
        if recomputed.get("joint_terminal_readiness") is not True:
            raise ValueError("joint terminal readiness is false")
        if recomputed["h100_force_gate"].get("beats_persistence") is not True:
            raise ValueError("H100 total-drag model does not beat persistence")
        checkpoint_sha = recomputed["checkpoint_sha256"]
        if promotion_lineage is None:
            raise ValueError("dev30/full40 promotion lineage is unavailable")
        if (
            recomputed["data_manifest_sha256"]
            != promotion_lineage["full40_manifest_sha256"]
            or recomputed["normalization_sha256"]
            != promotion_lineage["full40_normalization_sha256"]
        ):
            raise ValueError("validation gate differs from promoted full40 lineage")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"validation_gate_invalid:{error}")
    if checkpoint_sha is not None:
        for path, status, label in (
            (window_gate, WINDOW_GATE_STATUS, "window_gate"),
            (dynamic_gate, DYNAMIC_GATE_STATUS, "dynamic_gate"),
        ):
            try:
                validate_evidence_gate(
                    path, expected_status=status, checkpoint_sha256=checkpoint_sha
                )
                evidence_gate_sha256[label] = sha256(path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                blockers.append(f"{label}_invalid:{error}")
    else:
        blockers.extend(
            ("window_gate_blocked_by_checkpoint", "dynamic_gate_blocked_by_checkpoint")
        )
    try:
        baseline_rows = validate_baselines(baselines)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"canonical_baselines_invalid:{error}")
    return {
        "status": (
            "FULL40_CANONICAL_PPO_EXECUTION_READY"
            if not blockers
            else "FULL40_CANONICAL_PPO_EXECUTION_BLOCKED"
        ),
        "profile": PROFILE,
        "training_executed": False,
        "action_contract": action_contract,
        "observation_contract": "64 velocity probes + 4 forces + omega = 69D",
        "episode_steps": episode_steps,
        "train_cases": list(TRAIN_CASES),
        "validation_cases": list(VALIDATION_CASES),
        "frozen_test_directory_enumerated_or_opened": False,
        "checkpoint_sha256": checkpoint_sha,
        "dev30_full40_promotion_lineage": promotion_lineage,
        "validation_gate_sha256": (
            sha256(validation_gate) if checkpoint_sha is not None else None
        ),
        "baseline_phases_verified": sorted(baseline_rows) if baseline_rows else [],
        "baseline_artifact_sha256": sha256(baselines) if baseline_rows else None,
        "evidence_gate_sha256": evidence_gate_sha256,
        "blockers": blockers,
        "scientific_scope": (
            "surrogate PPO readiness only; real-CFD replay remains mandatory and "
            "no surrogate score proves physical closed-loop benefit"
        ),
    }


class ConstantPolicy:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, observation, deterministic: bool = True):
        import numpy as np

        del deterministic
        shape = (
            (len(observation), 1)
            if getattr(observation, "ndim", 1) > 1
            else (1,)
        )
        return np.full(shape, self.value, dtype=np.float32), None


def evaluate_policy(env, policy, steps: int) -> dict:
    observation, _ = env.reset()
    rows = []
    reward_sums: dict[str, float] = {}
    for step in range(steps):
        action = policy.predict(observation, deterministic=True)[0]
        observation, _, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "time": float((step + 1) * 0.1),
                "predicted_front_cd": info["predicted_front_cd"],
                "predicted_front_cl": info["predicted_front_cl"],
                "predicted_rear_cd": info["predicted_rear_cd"],
                "predicted_rear_cl": info["predicted_rear_cl"],
                "applied_omega": info["applied_omega"],
                "applied_delta_omega": info["applied_delta_omega"],
            }
        )
        for key, value in info.items():
            if key.startswith("reward_"):
                reward_sums[key] = reward_sums.get(key, 0.0) + float(value)
        if terminated or (truncated and step + 1 < steps):
            raise RuntimeError("canonical surrogate evaluation ended early")
    flow = env.unwrapped.flow
    audit = evaluate_canonical_episode(
        rows,
        flow.canonical_baseline,
        initial_omega=float(flow.initial_state["omega"]),
    )
    return {
        **audit,
        "reward_sum": float(sum(reward_sums.values())),
        "reward_components": reward_sums,
        "evaluated_on_surrogate_only": True,
    }


def execute(args, readiness: dict, *, train_only_smoke: bool = False) -> dict:
    import torch
    from evaluate_tandem_fno import load_composed_config
    from physicsnemo.utils import load_checkpoint
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train_tandem_fno import build_model

    from fluid_control.full40_canonical_hydrogym import make_full40_canonical_env

    allowed = (
        "FULL40_CANONICAL_PPO_TRAIN_ONLY_SMOKE_READY"
        if train_only_smoke
        else "FULL40_CANONICAL_PPO_EXECUTION_READY"
    )
    if readiness["status"] != allowed:
        raise RuntimeError("refusing PPO execution because readiness is blocked")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or device.index not in (None, 0)
        or not torch.cuda.is_available()
    ):
        raise ValueError("canonical PPO execution requires available GPU0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=0)
    baselines = (
        validate_train20_baselines(args.baselines)
        if train_only_smoke
        else validate_baselines(args.baselines)
    )
    if sha256(args.baselines) != readiness["baseline_artifact_sha256"]:
        raise ValueError("baseline artifact changed after preflight")
    if not train_only_smoke:
        lineage = readiness["dev30_full40_promotion_lineage"]
        current_lineage = {
            "promotion_receipt_sha256": sha256(args.promotion_receipt),
            "dev30_manifest_sha256": sha256(args.dev30_data / "manifest.json"),
            "full40_manifest_sha256": sha256(args.data / "manifest.json"),
            "dev30_normalization_sha256": sha256(
                args.dev30_data / "normalization.json"
            ),
            "full40_normalization_sha256": sha256(
                args.data / "normalization.json"
            ),
        }
        if any(lineage.get(key) != value for key, value in current_lineage.items()):
            raise ValueError("dev30/full40 promotion lineage changed after preflight")
        if sha256(args.validation_gate) != readiness["validation_gate_sha256"]:
            raise ValueError("validation gate changed after preflight")
        for label, path in (
            ("window_gate", args.window_gate),
            ("dynamic_gate", args.dynamic_gate),
        ):
            if sha256(path) != readiness["evidence_gate_sha256"][label]:
                raise ValueError(f"{label} changed after preflight")
    cfg = load_composed_config(args.config)
    model_files = sorted(args.checkpoint_dir.glob("FNO.*.mdlus"))
    if len(model_files) != 1 or sha256(model_files[0]) != readiness["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after readiness verification")
    network = build_model(cfg).to(device).eval().requires_grad_(False)
    epoch = load_checkpoint(args.checkpoint_dir, models=network, device=device)
    if epoch < 1:
        raise FileNotFoundError("full40 PhysicsNeMo checkpoint did not load")

    def make_case(split: str, case: str, phase: str):
        return Monitor(
            make_full40_canonical_env(
                data=args.data,
                split=split,
                case=case,
                frame=0,
                network=network,
                checkpoint_epoch=epoch,
                baseline=baselines[f"b{phase}"],
                episode_steps=args.episode_steps,
                device=device,
            )
        )

    train_env = DummyVecEnv(
        [
            partial(make_case, "train", case, phase)
            for case, phase in zip(TRAIN_CASES, TRAIN_PHASES, strict=True)
        ]
    )
    args.output.mkdir(parents=True)
    checkpoint_root = args.output / "checkpoints"
    checkpoint_root.mkdir()
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        device=device,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
    )
    diagnostics = TrainingDiagnosticsAccumulator()

    class CanonicalTrainingDiagnosticsCallback(BaseCallback):
        def _on_step(self) -> bool:
            diagnostics.record(self.locals.get("infos", ()))
            return True

    callback = CanonicalTrainingDiagnosticsCallback(verbose=0)
    iterations = []
    completed = 0
    try:
        model.save(checkpoint_root / "ppo_000000.zip")
        while completed < args.timesteps:
            model.learn(
                total_timesteps=args.checkpoint_interval,
                reset_num_timesteps=(completed == 0),
                callback=callback,
            )
            completed += args.checkpoint_interval
            target = checkpoint_root / f"ppo_{completed:08d}.zip"
            model.save(target)
            iterations.append(
                {
                    "timesteps": completed,
                    "checkpoint": str(target),
                    "checkpoint_sha256": sha256(target),
                    "train_diagnostics": diagnostics.snapshot(reset=True),
                }
            )
            write_atomic(args.output / "progress.json", {"iterations": iterations})
    finally:
        train_env.close()

    policies = {
        "zero": ConstantPolicy(0.0),
        "constant_m075": ConstantPolicy(-ACTION_LIMIT),
        "constant_p075": ConstantPolicy(ACTION_LIMIT),
        "ppo_final": model,
    }
    validation = []
    if not train_only_smoke:
        for case, phase in zip(VALIDATION_CASES, VALIDATION_PHASES, strict=True):
            for label, policy in policies.items():
                env = make_case("validation", case, phase)
                try:
                    validation.append(
                        {
                            "case": case,
                            "phase": f"b{phase}",
                            "policy": label,
                            **evaluate_policy(env, policy, args.episode_steps),
                        }
                    )
                finally:
                    env.close()
    return {
        "status": (
            "FULL40_CANONICAL_PPO_TRAIN_ONLY_SMOKE_COMPLETE"
            if train_only_smoke
            else "FULL40_CANONICAL_PPO_SURROGATE_RUN_COMPLETE"
        ),
        "scientific_status": (
            "train_only_software_commissioning_no_control_claim"
            if train_only_smoke
            else "surrogate_only_requires_predeclared_real_cfd_replay"
        ),
        "training_executed": True,
        "physicsnemo_checkpoint_epoch": epoch,
        "physicsnemo_checkpoint_sha256": readiness["checkpoint_sha256"],
        "iterations": iterations,
        "validation_policy": (
            "not opened in train-only smoke"
            if train_only_smoke
            else "single final validation evaluation; no checkpoint selection"
        ),
        "validation": validation,
        "frozen_test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dev30-data", type=Path)
    parser.add_argument("--promotion-receipt", type=Path)
    parser.add_argument("--validation-gate", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--validation-segments", type=Path)
    parser.add_argument("--predeclaration", type=Path)
    parser.add_argument("--window-gate", type=Path)
    parser.add_argument("--dynamic-gate", type=Path)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--checkpoint-interval", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261003)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--train-only-smoke", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    minimum_timesteps = 512 if args.train_only_smoke else 2048
    if args.timesteps < minimum_timesteps or args.timesteps % 512:
        parser.error(
            f"timesteps must be >={minimum_timesteps} and divisible by 512"
        )
    if args.checkpoint_interval < 512 or args.checkpoint_interval % 512:
        parser.error("checkpoint interval must be divisible by 512")
    if args.timesteps % args.checkpoint_interval:
        parser.error("checkpoint interval must divide timesteps")
    if not 0.0 < args.gpu_memory_fraction <= 0.20:
        parser.error("gpu memory fraction must be in (0, 0.20]")
    if args.train_only_smoke:
        readiness = smoke_preflight(
            data=args.data,
            config=args.config,
            checkpoint_dir=args.checkpoint_dir,
            baselines=args.baselines,
            image_id=args.runtime_image_id,
            episode_steps=args.episode_steps,
        )
    else:
        required = (
            args.validation_gate,
            args.validation_report,
            args.validation_segments,
            args.predeclaration,
            args.window_gate,
            args.dynamic_gate,
            args.dev30_data,
            args.promotion_receipt,
        )
        if any(path is None for path in required):
            parser.error("formal dry-run/execute requires all validation gate paths")
        readiness = preflight(
            data=args.data,
            config=args.config,
            checkpoint_dir=args.checkpoint_dir,
            dev30_data=args.dev30_data,
            promotion_receipt=args.promotion_receipt,
            validation_gate=args.validation_gate,
            validation_report=args.validation_report,
            validation_segments=args.validation_segments,
            predeclaration=args.predeclaration,
            window_gate=args.window_gate,
            dynamic_gate=args.dynamic_gate,
            baselines=args.baselines,
            image_id=args.image_id,
            runtime_image_id=args.runtime_image_id,
            episode_steps=args.episode_steps,
        )
    if args.dry_run:
        write_exclusive(args.output, readiness)
        print(json.dumps(readiness, indent=2))
        return
    expected_ready = (
        "FULL40_CANONICAL_PPO_TRAIN_ONLY_SMOKE_READY"
        if args.train_only_smoke
        else "FULL40_CANONICAL_PPO_EXECUTION_READY"
    )
    if readiness["status"] != expected_ready:
        blocked = args.output.with_suffix(".blocked.json")
        write_exclusive(blocked, readiness)
        raise RuntimeError(f"execution blocked; audit written to {blocked}")
    result = execute(args, readiness, train_only_smoke=args.train_only_smoke)
    write_exclusive(args.output / "audit.json", result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "validation"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
