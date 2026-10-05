#!/usr/bin/env python3
"""CPU-only, fail-closed readiness audit for a new FNO PPO candidate.

This adapter does not train or load a model, enumerate frozen data, or authorize
PPO by itself.  It binds one candidate generation to the three historical
canonical surrogate gates and the additional development-admission gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import yaml

from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero


PROFILE = "matched_start_full40_v1"
IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
ENDPOINT_STATUS = "FULL40_VALIDATION_SURROGATE_READINESS_PASS"
WINDOW_STATUS = "FULL40_VALIDATION_CANONICAL_WINDOW_FIDELITY_PASS"
DYNAMIC_STATUS = "FULL40_VALIDATION_DYNAMIC_ACTION_PASS"
DEVELOPMENT_STATUS = "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS"
FC_P003C_KIND = "true_state_paired_step_lambda10"
FC_P008_KIND = "full_train_force_row_recalibration"
FC_P013_KIND = "fcp013_independent_force_dual_fno"
FC_P013_CONFIG = Path("artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml")
HOST_REPO = Path("/workspace/fluid_control")
FC_P008_CONFIG = Path(
    "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/"
    "resolved_config.yaml"
)
FC_P008_PRECISION = {
    "NVIDIA_TF32_OVERRIDE": None,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "high",
}
FC_P008_PRECISION_STATUS = "FC_P008_FORMAL_EVALUATION_DEFAULT_TF32_HIGH"
FC_P003C_SOURCES = {
    "/workspace/train8": {
        "lineage_key": "train8",
        "windows": 408,
        "stride": 2,
        "release_status": "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED",
        "trajectory_count": 8,
    },
    "/workspace/train16": {
        "lineage_key": "train16",
        "windows": 240,
        "stride": 2,
        "release_status": "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED",
        "trajectory_count": 16,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def archive_payload(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("model archive has duplicate members")
        if set(names) != {"model.pt", "args.json", "metadata.json"}:
            raise ValueError("model archive member set differs")
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(names)
        }


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}:{path}")


def _receipt_binds(receipt: dict, receipt_root: Path, path: Path) -> None:
    resolved = path.resolve()
    root = receipt_root.resolve()
    try:
        relative = str(resolved.relative_to(root))
    except ValueError as error:
        raise ValueError(f"posteval artifact escapes receipt root: {path}") from error
    recorded = receipt.get("sha256", {}).get(relative)
    if recorded != sha256(resolved):
        raise ValueError(f"posteval receipt does not bind {relative}")


def _validate_receipt_table(receipt: dict, receipt_root: Path) -> None:
    table = receipt.get("sha256")
    if not isinstance(table, dict) or not table:
        raise ValueError("posteval receipt SHA table is missing")
    root = receipt_root.resolve()
    for raw, expected in table.items():
        if not isinstance(raw, str) or not isinstance(expected, str):
            raise ValueError("posteval receipt SHA entry type differs")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"posteval receipt path is not confined: {raw}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"posteval receipt path escapes root: {raw}") from error
        _require_file(path, f"posteval receipt artifact {raw}")
        if sha256(path) != expected:
            raise ValueError(f"posteval receipt artifact differs: {raw}")


def _validate_fcp003c_training_sources(
    *, candidate_root: Path, completion: dict, data_lineage: dict, normalization: str
) -> None:
    """Bind FC-P003C's fixed train-only sources without changing legacy semantics."""
    table = completion.get("sha256")
    if not isinstance(table, dict):
        raise ValueError("candidate completion SHA table is missing")
    relative = "training_data_sources.json"
    path = candidate_root / relative
    _require_file(path, "FC-P003C training data sources")
    if table.get(relative) != sha256(path):
        raise ValueError("FC-P003C completion does not bind training data sources")
    value = load(path)
    if (
        value.get("base_root") != "/workspace/base"
        or value.get("base_windows") != 720
        or value.get("total_windows") != 1368
    ):
        raise ValueError("FC-P003C base training source contract differs")
    sources = value.get("additional_sources")
    if not isinstance(sources, list) or len(sources) != len(FC_P003C_SOURCES):
        raise ValueError("FC-P003C additional training source count differs")
    by_root = {row.get("root"): row for row in sources if isinstance(row, dict)}
    if set(by_root) != set(FC_P003C_SOURCES):
        raise ValueError("FC-P003C additional training source roots differ")
    for root, expected in FC_P003C_SOURCES.items():
        row = by_root[root]
        lineage_key = expected["lineage_key"]
        if row.get("manifest_sha256") != data_lineage.get(lineage_key):
            raise ValueError(f"FC-P003C {lineage_key} manifest SHA differs")
        if row.get("normalization_sha256") != normalization:
            raise ValueError(f"FC-P003C {lineage_key} normalization SHA differs")
        for key in ("windows", "stride", "release_status", "trajectory_count"):
            if row.get(key) != expected[key]:
                raise ValueError(f"FC-P003C {lineage_key} {key} differs")


def _common_gate(
    gate: dict,
    *,
    status: str,
    checkpoint: str,
    normalization: str,
    validation_manifest: str,
) -> None:
    required = {
        "status": status,
        "profile": PROFILE,
        "checkpoint_sha256": checkpoint,
        "normalization_sha256": normalization,
        "data_manifest_sha256": validation_manifest,
        "frozen_test_accessed": False,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            if key == "status":
                raise RuntimeError(f"scientific gate did not pass: {status}")
            raise ValueError(f"{status} {key} differs")


def _bound_evidence(gate: dict, repo_root: Path) -> None:
    for prefix in ("producer_script", "evidence"):
        raw = gate.get(f"{prefix}_path" if prefix == "evidence" else prefix)
        path = Path(str(raw or ""))
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as error:
            raise ValueError(f"{prefix} escapes repository") from error
        _require_file(path, prefix)
        if sha256(path) != gate.get(f"{prefix}_sha256"):
            raise ValueError(f"{prefix} SHA differs")


def p013_candidate_identity(*, repo_root, candidate_root, lineage, receipt_path,
                            normalization_path, data_artifacts):
    """Bind actual P013 evidence without adapting it into a legacy training schema."""
    from fluid_control.dual_control_contract import verify_dual_control_binding
    from fluid_control.dual_fno import ARCHITECTURE, validate_dual_fno_manifest, validate_dual_runtime_files

    required = {
        "status": "FC_P013_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION",
        "candidate_kind": FC_P013_KIND, "checkpoint_epoch": 1,
        "checkpoint_relative_directory": "candidate/aerodynamic",
        "checkpoint_model_file": "FNO.0.1.mdlus",
        "checkpoint_state_file": "checkpoint.0.1.pt",
        "training_performed": True, "optimizer_training_performed": True,
        "calibration_fit_performed": False, "optimizer_steps": 1368,
        "validation_or_frozen_accessed": False, "ppo_auto_launch": False,
        "official_image_id": IMAGE_ID,
    }
    if any(lineage.get(k) != v for k, v in required.items()):
        raise ValueError("P013 lineage identity/scope differs")
    relative = candidate_root.relative_to(repo_root)
    if lineage.get("candidate_root") not in (str(candidate_root), str(relative), str(HOST_REPO / relative)):
        raise ValueError("P013 candidate root identity differs")
    config = repo_root / FC_P013_CONFIG
    manifest = candidate_root / "candidate/dual_model_manifest.json"
    identity = validate_dual_fno_manifest(manifest, expected_sha256=lineage["dual_manifest_sha256"])
    validate_dual_runtime_files(identity, config_path=config, normalization_path=normalization_path)
    for model in (identity.flow.model, identity.aerodynamic.model):
        archive_payload(model)
        with zipfile.ZipFile(model) as archive:
            arguments = json.loads(archive.read("args.json")).get("__args__", {})
        if arguments.get("dimension") != 2 or any(
                arguments.get(k) != v for k, v in ARCHITECTURE.items() if k != "force_channels"):
            raise ValueError("P013 official model archive architecture differs")
    expected = {
        "checkpoint_sha256": identity.aerodynamic.model_sha256,
        "checkpoint_state_sha256": identity.aerodynamic.state_sha256,
        "flow_model_sha256": identity.flow.model_sha256,
        "flow_state_sha256": identity.flow.state_sha256,
        "resolved_config_sha256": sha256(config),
    }
    if any(lineage.get(k) != v for k, v in expected.items()):
        raise ValueError("P013 persisted dual identity differs")
    completion = load(candidate_root / "completion_receipt.json")
    audit_path = candidate_root / "candidate_audit.json"
    candidate_audit = load(audit_path)
    if (completion.get("status") != "FC_P013_TRAINING_COMPLETE_NOT_ADMISSION"
            or completion.get("candidate_audit_sha256") != sha256(audit_path)
            or completion.get("optimizer_steps") != 1368
            or completion.get("scientific_admission") is not False
            or completion.get("ppo_executed") is not False
            or any(lineage.get(k) != v for k, v in candidate_audit.items())):
        raise ValueError("P013 terminal audit/completion binding differs")
    _validate_receipt_table(candidate_audit, candidate_root)
    completion_expected = {
        "dual_manifest_sha256": identity.manifest_sha256,
        "flow_model_sha256": identity.flow.model_sha256,
        "flow_state_sha256": identity.flow.state_sha256,
        "aerodynamic_model_sha256": identity.aerodynamic.model_sha256,
        "aerodynamic_state_sha256": identity.aerodynamic.state_sha256,
        "candidate_result_sha256": lineage["candidate_result_sha256"],
    }
    if any(completion.get(k) != v for k, v in completion_expected.items()):
        raise ValueError("P013 completion model/result identity differs")
    from audit_fcp011_candidate import INPUT_SHA
    # The terminal audit reports the release hashes through candidate/result.json.
    result = load(candidate_root / "candidate/result.json")
    if result.get("input_sha256") != INPUT_SHA or set(data_artifacts) != set(INPUT_SHA):
        raise ValueError("P013 data artifact keys differ")
    if any(sha256(data_artifacts[k]) != v for k, v in INPUT_SHA.items()):
        raise ValueError("P013 data release bytes differ")
    receipt_path.resolve().relative_to(candidate_root)
    binding = verify_dual_control_binding(
        manifest_path=manifest, expected_manifest_sha256=identity.manifest_sha256,
        training_config=config, normalization_path=normalization_path,
        checkpoint_dir=identity.aerodynamic.directory,
        expected_checkpoint_sha256=identity.aerodynamic.model_sha256,
        posteval_receipt=receipt_path, expected_posteval_receipt_sha256=sha256(receipt_path),
        development_auditor=repo_root / "scripts/audit_dynamic_fno_development_gates.py",
    )
    precision = load(receipt_path.parent / "precision.json")
    if precision != {"status": "FC_P013_FORMAL_EVALUATION_DEFAULT_TF32_HIGH",
                     "official_image_id": IMAGE_ID, **FC_P008_PRECISION}:
        raise ValueError("P013 formal precision evidence differs")
    approval = load(receipt_path.parent / "evidence/formal_evaluation_approval.json")
    if (approval.get("candidate_kind") != FC_P013_KIND
            or approval.get("candidate_completion_receipt_sha256") != sha256(candidate_root / "completion_receipt.json")
            or approval.get("candidate_result_sha256") != lineage["candidate_result_sha256"]
            or approval.get("dual_manifest_sha256") != identity.manifest_sha256):
        raise ValueError("P013 formal approval belongs to another candidate")
    return {
        "config": config, "epoch": 1,
        "checkpoint_relative_directory": Path("candidate/aerodynamic"),
        "model_file": identity.aerodynamic.model.name,
        "state_file": identity.aerodynamic.state.name,
        "dual_control_binding": binding,
        "dual_manifest_path": str(manifest.relative_to(repo_root)),
        "dual_posteval_receipt_path": str(receipt_path.resolve().relative_to(repo_root)),
    }


def audit(
    *,
    repo_root: Path,
    candidate_root: Path,
    lineage_path: Path,
    posteval_receipt_path: Path,
    endpoint_gate_path: Path,
    endpoint_evaluation_config_path: Path,
    window_gate_path: Path,
    dynamic_gate_path: Path,
    development_gate_path: Path,
    validation_manifest_path: Path,
    dynamic_validation_manifest_path: Path,
    normalization_path: Path,
    data_artifacts: dict[str, Path],
    official_image_id: str = IMAGE_ID,
) -> dict:
    """Return READY or BLOCKED without throwing for ordinary bad evidence."""
    repo_root = repo_root.resolve()
    candidate_root = (
        candidate_root.resolve()
        if candidate_root.is_absolute()
        else (repo_root / candidate_root).resolve()
    )
    paths = {
        "candidate_root": candidate_root,
        "lineage": lineage_path,
        "posteval_receipt": posteval_receipt_path,
        "endpoint_gate": endpoint_gate_path,
        "endpoint_evaluation_config": endpoint_evaluation_config_path,
        "canonical_window_gate": window_gate_path,
        "canonical_dynamic_gate": dynamic_gate_path,
        "development_gate": development_gate_path,
        "validation_manifest": validation_manifest_path,
        "dynamic_validation_manifest": dynamic_validation_manifest_path,
        "normalization": normalization_path,
    }
    blockers: list[dict[str, str]] = []
    if official_image_id != IMAGE_ID:
        blockers.append(
            {
                "kind": "SCHEMA_ERROR",
                "item": "official_image_id",
                "detail": "official PhysicsNeMo image ID differs from the pinned runtime",
            }
        )
    for label, path in paths.items():
        if label == "candidate_root":
            if not path.is_dir():
                blockers.append({"kind": "MISSING", "item": label, "detail": str(path)})
        elif not path.is_file():
            blockers.append({"kind": "MISSING", "item": label, "detail": str(path)})
    for key, path in sorted(data_artifacts.items()):
        if not path.is_file():
            blockers.append({"kind": "MISSING", "item": f"data:{key}", "detail": str(path)})
    if blockers:
        return _result(blockers=blockers, official_image_id=official_image_id)

    try:
        lineage = load(lineage_path)
        receipt = load(posteval_receipt_path)
        endpoint = load(endpoint_gate_path)
        window = load(window_gate_path)
        dynamic = load(dynamic_gate_path)
        development = load(development_gate_path)
        validation_manifest = load(validation_manifest_path)
        dynamic_validation_manifest = load(dynamic_validation_manifest_path)
        normalization = load(normalization_path)
        validation_manifest_sha = sha256(validation_manifest_path)
        dynamic_validation_manifest_sha = sha256(dynamic_validation_manifest_path)
        normalization_sha = sha256(normalization_path)
        if (
            validation_manifest.get("profile") != PROFILE
            or validation_manifest.get("max_abs_omega") != 0.75
            or validation_manifest.get("trajectory_counts")
            != {"train": 20, "validation": 10, "frozen_test": 10}
        ):
            raise ValueError("validation manifest action/split contract differs")
        if (
            dynamic_validation_manifest.get("profile")
            != "full40_dynamic_validation_v1"
            or dynamic_validation_manifest.get("max_abs_omega") != 0.75
            or dynamic_validation_manifest.get("trajectory_counts")
            != {"train": 0, "validation": 6, "frozen_test": 0}
            or dynamic_validation_manifest.get("training_access") != "FORBIDDEN"
            or dynamic_validation_manifest.get("frozen_test_accessed") is not False
            or dynamic_validation_manifest.get("normalization_sha256")
            != normalization_sha
        ):
            raise ValueError("dynamic validation manifest action/split contract differs")

        candidate_kind = lineage.get("candidate_kind")
        p013 = None
        if candidate_kind == FC_P013_KIND:
            p013 = p013_candidate_identity(
                repo_root=repo_root, candidate_root=candidate_root, lineage=lineage,
                receipt_path=posteval_receipt_path, normalization_path=normalization_path,
                data_artifacts=data_artifacts,
            )
            config, epoch = p013["config"], p013["epoch"]
            config_sha = sha256(config)
            checkpoint_relative_directory = p013["checkpoint_relative_directory"]
            model_file, state_file = p013["model_file"], p013["state_file"]
        else:
            if not str(lineage.get("status", "")).endswith("CANDIDATE_LINEAGE_PASS"):
                raise ValueError("candidate lineage status differs")
            candidate_kind = lineage.get("candidate_kind")
            if candidate_kind == FC_P008_KIND:
                required_scope = {
                    "training_performed": False,
                    "optimizer_training_performed": False,
                    "calibration_fit_performed": True,
                    "validation_or_frozen_accessed": False,
                    "ppo_auto_launch": False,
                    "parent_checkpoint_epoch": 2,
                    "calibration_generation": 1,
                }
            else:
                required_scope = {
                    "training_performed": candidate_kind == FC_P003C_KIND,
                    "frozen_test_opened_or_enumerated": False,
                    "ppo_auto_launch": False,
                }
            if any(lineage.get(key) != value for key, value in required_scope.items()):
                raise ValueError("candidate lineage scope differs")
            try:
                candidate_relative = str(candidate_root.relative_to(repo_root))
            except ValueError as error:
                raise ValueError("candidate root escapes repository") from error
            if lineage.get("candidate_root") not in (str(candidate_root), candidate_relative):
                raise ValueError("candidate root identity differs")

            epoch = int(lineage["checkpoint_epoch"])
            if candidate_kind == FC_P008_KIND:
                if (
                    epoch != 0
                    or lineage.get("checkpoint_relative_directory")
                    != "candidate_build/candidate"
                    or lineage.get("checkpoint_model_file") != "FNO.0.0.mdlus"
                    or lineage.get("checkpoint_state_file") != "checkpoint.0.0.pt"
                ):
                    raise ValueError("FC-P008 checkpoint identity differs")
                checkpoint_relative_directory = Path("candidate_build/candidate")
                model_file, state_file = "FNO.0.0.mdlus", "checkpoint.0.0.pt"
                config = (repo_root / FC_P008_CONFIG).resolve()
            else:
                checkpoint_relative_directory = Path("best")
                model_file = f"FNO.0.{epoch}.mdlus"
                state_file = f"checkpoint.0.{epoch}.pt"
                config = candidate_root / "resolved_config.yaml"
            checkpoint_dir = candidate_root / checkpoint_relative_directory
            model = checkpoint_dir / model_file
            state = checkpoint_dir / state_file
            launch = candidate_root / "launch_receipt.json"
            completion = candidate_root / "completion_receipt.json"
            for label, path in (
                ("model", model), ("state", state), ("resolved config", config),
                ("launch receipt", launch), ("completion receipt", completion),
            ):
                _require_file(path, label)
            if sha256(model) != lineage.get("checkpoint_sha256"):
                raise ValueError("candidate model SHA differs")
            if sha256(state) != lineage.get("checkpoint_state_sha256"):
                raise ValueError("candidate state SHA differs")
            config_sha = sha256(config)
            if candidate_kind == FC_P008_KIND:
                if config_sha != lineage.get("data_lineage", {}).get("resolved_config"):
                    raise ValueError("FC-P008 resolved config SHA differs")
            elif config_sha != lineage.get("resolved_config_sha256"):
                raise ValueError("candidate resolved config SHA differs")
            if sha256(launch) != lineage.get("launch_receipt_sha256"):
                raise ValueError("candidate launch receipt SHA differs")
            if sha256(completion) != lineage.get("completion_receipt_sha256"):
                raise ValueError("candidate completion receipt SHA differs")
            completion_value = load(completion)
            if archive_payload(model) != lineage.get("checkpoint_generation_payload_sha256"):
                raise ValueError("candidate model archive payload differs")
            if candidate_kind == FC_P008_KIND:
                validate_calibrated_epoch_zero(
                    checkpoint_dir,
                    epoch,
                    allow=True,
                    expected_model_sha256=lineage.get("checkpoint_sha256"),
                    expected_state_sha256=lineage.get("checkpoint_state_sha256"),
                )
                if lineage.get("precision_protocol") != FC_P008_PRECISION:
                    raise ValueError("FC-P008 lineage precision protocol differs")
                precision_path = posteval_receipt_path.parent / "precision.json"
                _require_file(precision_path, "FC-P008 formal precision evidence")
                precision = load(precision_path)
                expected_precision = {
                    "status": FC_P008_PRECISION_STATUS,
                    "official_image_id": official_image_id,
                    **FC_P008_PRECISION,
                }
                if precision != expected_precision:
                    raise ValueError("FC-P008 formal precision evidence differs")
                if receipt.get("precision_sha256") != sha256(precision_path):
                    raise ValueError("FC-P008 receipt precision SHA differs")
                _receipt_binds(receipt, posteval_receipt_path.parent, precision_path)

            config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
            if not isinstance(config_value, dict):
                raise ValueError("resolved config is not a mapping")
            training, model_config = config_value.get("training", {}), config_value.get("model", {})
            data_config = config_value.get("data", {})
            expected_model = {
                "in_channels": 6,
                "out_channels": 7,
                "latent_channels": 48,
                "num_fno_layers": 5,
                "num_fno_modes": [32, 32],
                "decoder_layers": 2,
                "decoder_layer_size": 128,
                "padding": 8,
                "coord_features": True,
            }
            if any(model_config.get(key) != value for key, value in expected_model.items()):
                raise ValueError("candidate FNO architecture differs")
            if (
                training.get("rollout_steps") != 100
                or training.get("validation_rollout_steps") != 100
                or data_config.get("force_indices") != [0, 1, 2, 3]
            ):
                raise ValueError("candidate is not the canonical H100 six-input/seven-output FNO")
            with zipfile.ZipFile(model) as archive:
                args = json.loads(archive.read("args.json"))
            archive_args = args.get("__args__", {})
            if any(archive_args.get(key) != value for key, value in expected_model.items()):
                raise ValueError("candidate model archive architecture differs")
            if archive_args.get("dimension") != 2:
                raise ValueError("candidate model archive dimension differs")
            if (
                normalization.get("state_channels") != ["u", "v", "gauge_pressure"]
                or normalization.get("all_force_channels")
                != ["front_cd", "front_cl", "rear_cd", "rear_cl"]
                or len(normalization.get("state_mean", [])) != 3
                or len(normalization.get("state_std", [])) != 3
                or len(normalization.get("all_force_mean", [])) != 4
                or len(normalization.get("all_force_std", [])) != 4
            ):
                raise ValueError("candidate normalization channel contract differs")

            data_lineage = lineage.get("data_lineage")
            if not isinstance(data_lineage, dict):
                raise ValueError("candidate data lineage is missing")
            if candidate_kind == FC_P003C_KIND:
                _validate_fcp003c_training_sources(
                    candidate_root=candidate_root,
                    completion=completion_value,
                    data_lineage=data_lineage,
                    normalization=normalization_sha,
                )
                expected_data = data_lineage
            else:
                if data_lineage.get("normalization") != normalization_sha:
                    raise ValueError("candidate normalization SHA differs")
                expected_data = {
                    key: value
                    for key, value in data_lineage.items()
                    if key != "normalization"
                }
            if set(data_artifacts) != set(expected_data):
                raise ValueError("candidate data artifact key set differs")
            for key, path in data_artifacts.items():
                if sha256(path) != expected_data[key]:
                    raise ValueError(f"candidate data artifact differs: {key}")

        if not str(receipt.get("status", "")).endswith("POSTEVAL_COMPLETE"):
            raise ValueError("posteval completion status differs")
        if (
            receipt.get("checkpoint_sha256") != lineage["checkpoint_sha256"]
            or receipt.get("frozen_test_accessed") is not False
            or receipt.get("ppo_auto_launched") is not False
        ):
            raise ValueError("posteval receipt scope differs")
        receipt_root = posteval_receipt_path.parent
        if receipt_root.resolve() != posteval_receipt_path.resolve().parent:
            raise ValueError("posteval receipt root differs")
        try:
            receipt_root.resolve().relative_to(candidate_root)
        except ValueError as error:
            raise ValueError("posteval receipt is outside candidate root") from error
        _validate_receipt_table(receipt, receipt_root)
        for path in (lineage_path, endpoint_gate_path, development_gate_path):
            _receipt_binds(receipt, receipt_root, path)

        checkpoint = lineage["checkpoint_sha256"]
        _common_gate(
            endpoint, status=ENDPOINT_STATUS, checkpoint=checkpoint,
            normalization=normalization_sha, validation_manifest=validation_manifest_sha,
        )
        if (
            endpoint.get("split") != "validation"
            or endpoint.get("joint_terminal_readiness") is not True
            or endpoint.get("h100_force_gate", {}).get("beats_persistence") is not True
            or endpoint.get("physicsnemo_image_id") != official_image_id
            or endpoint.get("model_config_sha256")
            != sha256(endpoint_evaluation_config_path)
        ):
            raise RuntimeError("scientific endpoint readiness did not pass")

        for gate, path, status in (
            (window, window_gate_path, WINDOW_STATUS),
            (dynamic, dynamic_gate_path, DYNAMIC_STATUS),
        ):
            _common_gate(
                gate, status=status, checkpoint=checkpoint,
                normalization=normalization_sha,
                validation_manifest=dynamic_validation_manifest_sha,
            )
            if gate.get("validation_phases") != ["b01", "b05"]:
                raise ValueError(f"{status} validation phases differ")
            _bound_evidence(gate, repo_root)
        if any(window.get(key) is not True for key in (
            "total_drag_window_fidelity_pass",
            "rear_cl_fluctuation_window_fidelity_pass",
            "rear_cl_mean_bias_window_fidelity_pass",
        )) or window.get("causal_window_seconds") != 6.15:
            raise RuntimeError("scientific canonical window gate did not pass")
        if (
            dynamic.get("max_abs_omega") != 0.75
            or dynamic.get("max_delta_omega") != 0.1
            or dynamic.get("minimum_horizon_steps") != 100
            or dynamic.get("dynamic_action_validation_pass") is not True
        ):
            raise RuntimeError("scientific canonical dynamic gate did not pass")

        if (
            development.get("status") != DEVELOPMENT_STATUS
            or development.get("checkpoint_sha256") != checkpoint
            or development.get("frozen_test_accessed") is not False
            or development.get("ppo_authorized") is not False
            or development.get("window_gate", {}).get("status") != "PASS"
            or development.get("dynamic_action_gate", {}).get("status") != "PASS"
        ):
            raise RuntimeError("scientific development admission did not pass")
        force_result = receipt_root / "force_window/result.json"
        _require_file(force_result, "development force-window evidence")
        _receipt_binds(receipt, receipt_root, force_result)
        if development.get("source_force_window_sha256") != sha256(force_result):
            raise ValueError("development gate force-window evidence differs")
    except RuntimeError as error:
        blockers.append({"kind": "SCIENTIFIC_FAIL", "item": "gate", "detail": str(error)})
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile, yaml.YAMLError) as error:
        blockers.append({"kind": "SCHEMA_ERROR", "item": "evidence", "detail": str(error)})

    identities = None
    if not blockers:
        identities = {
            "candidate_kind": lineage["candidate_kind"],
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": lineage["checkpoint_sha256"],
            "checkpoint_state_sha256": lineage["checkpoint_state_sha256"],
            "resolved_config_sha256": config_sha,
            "resolved_config_path": str(config.relative_to(repo_root)),
            "checkpoint_relative_directory": str(checkpoint_relative_directory),
            "checkpoint_model_file": model_file,
            "checkpoint_state_file": state_file,
            "precision_protocol": (
                FC_P008_PRECISION if candidate_kind in (FC_P008_KIND, FC_P013_KIND) else None
            ),
            "normalization_sha256": normalization_sha,
            "validation_manifest_sha256": validation_manifest_sha,
            "dynamic_validation_manifest_sha256": dynamic_validation_manifest_sha,
            "lineage_sha256": sha256(lineage_path),
            "posteval_receipt_sha256": sha256(posteval_receipt_path),
            "endpoint_gate_sha256": sha256(endpoint_gate_path),
            "endpoint_evaluation_config_sha256": sha256(
                endpoint_evaluation_config_path
            ),
            "canonical_window_gate_sha256": sha256(window_gate_path),
            "canonical_dynamic_gate_sha256": sha256(dynamic_gate_path),
            "development_gate_sha256": sha256(development_gate_path),
            "data_artifact_sha256": {
                key: sha256(path) for key, path in sorted(data_artifacts.items())
            },
        }
        if p013 is not None:
            identities.update({key: p013[key] for key in (
                "dual_control_binding", "dual_manifest_path", "dual_posteval_receipt_path"
            )})
    return _result(
        blockers=blockers, official_image_id=official_image_id, identities=identities
    )


def _result(
    *,
    blockers: list[dict[str, str]],
    official_image_id: str,
    identities: dict | None = None,
) -> dict:
    return {
        "status": (
            "CANDIDATE_PPO_CPU_DRY_RUN_READY"
            if not blockers
            else "CANDIDATE_PPO_CPU_DRY_RUN_BLOCKED"
        ),
        "candidate_identity": identities,
        "physicsnemo_image_id": official_image_id,
        "surrogate_episode_steps": 100,
        "observation_contract": "64 velocity probes + 4 forces + omega = 69D",
        "action_contract": {"max_abs_omega": 0.75, "max_delta_omega": 0.1},
        "blockers": blockers,
        "cpu_only": True,
        "training_executed": False,
        "policy_created": False,
        "ppo_execution_authorized": False,
        "frozen_test_directory_enumerated_or_opened": False,
        "scope": (
            "Candidate-specific input readiness for a separately reviewed PPO launch; "
            "not PPO training, real-CFD validation, or physical success."
        ),
    }


def parse_data_artifacts(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"invalid or duplicate --data-artifact: {value}")
        result[key] = Path(raw_path)
    return result


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
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
    parser.add_argument("--posteval-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-gate", type=Path, required=True)
    parser.add_argument("--endpoint-evaluation-config", type=Path, required=True)
    parser.add_argument("--window-gate", type=Path, required=True)
    parser.add_argument("--dynamic-gate", type=Path, required=True)
    parser.add_argument("--development-gate", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--dynamic-validation-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--data-artifact", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--official-image-id", default=IMAGE_ID)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit(
        repo_root=args.repo,
        candidate_root=args.candidate_root,
        lineage_path=args.lineage,
        posteval_receipt_path=args.posteval_receipt,
        endpoint_gate_path=args.endpoint_gate,
        endpoint_evaluation_config_path=args.endpoint_evaluation_config,
        window_gate_path=args.window_gate,
        dynamic_gate_path=args.dynamic_gate,
        development_gate_path=args.development_gate,
        validation_manifest_path=args.validation_manifest,
        dynamic_validation_manifest_path=args.dynamic_validation_manifest,
        normalization_path=args.normalization,
        data_artifacts=parse_data_artifacts(args.data_artifact),
        official_image_id=args.official_image_id,
    )
    write_exclusive(args.output, result)
    print(result["status"])


if __name__ == "__main__":
    main()
