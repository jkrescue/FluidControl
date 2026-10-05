"""Project-owned adapter for an independently checkpointed force FNO.

Both wrapped modules remain ordinary PhysicsNeMo FNO instances.  This adapter
only combines their raw output channels: state residuals come from the frozen
flow model and learned force-output channels come from the aerodynamic model.  It does
not apply the residual update, mask, spatial pooling, or physical scaling.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
MANIFEST_STATUS = "FC_P013_DUAL_FNO_MANIFEST_VERIFIED"
SYSTEM_KIND = "FC_P013_INDEPENDENT_FORCE_FNO"
FLOW_KIND = "FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"
AERO_KIND = "FC_P013_INDEPENDENT_FORCE_FNO_AERODYNAMIC_CHECKPOINT"
P015_MANIFEST_STATUS = "FC_P015_DUAL_FNO_MANIFEST_VERIFIED"
P015_SYSTEM_KIND = "FC_P015_WINDOW_ACCUMULATION_FORCE_FNO"
P015_AERO_KIND = "FC_P015_WINDOW_ACCUMULATION_FORCE_FNO_AERODYNAMIC_CHECKPOINT"
FLOW_MODEL_SHA256 = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
FLOW_STATE_SHA256 = "4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e"
CONFIG_SHA256 = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
NORMALIZATION_SHA256 = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
FORCE_CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
ARCHITECTURE = {
    "in_channels": 6,
    "out_channels": 7,
    "latent_channels": 48,
    "num_fno_layers": 5,
    "num_fno_modes": [32, 32],
    "decoder_layers": 2,
    "decoder_layer_size": 128,
    "padding": 8,
    "coord_features": True,
    "force_channels": list(FORCE_CHANNELS),
}
PRECISION_PROTOCOL = {
    "float32_matmul_precision": "high",
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
}
_SHA = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _confined_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("dual checkpoint path must be a nonempty relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("dual checkpoint file is missing or escapes the candidate root")
    return path


@dataclass(frozen=True)
class CheckpointIdentity:
    role: str
    directory: Path
    model: Path
    state: Path
    epoch: int
    model_sha256: str
    state_sha256: str
    metadata_kind: str


@dataclass(frozen=True)
class DualFNOIdentity:
    manifest_path: Path
    manifest_sha256: str
    flow: CheckpointIdentity
    aerodynamic: CheckpointIdentity
    payload: dict[str, Any]


def _experiment_contract(kind: str) -> dict[str, Any]:
    """Explicit project experiment identities; never infer steps from filenames."""
    if kind == SYSTEM_KIND:
        return {"status": MANIFEST_STATUS, "aero_kind": AERO_KIND,
                "optimizer_steps": 1368, "extra": {}}
    if kind == P015_SYSTEM_KIND:
        return {"status": P015_MANIFEST_STATUS, "aero_kind": P015_AERO_KIND,
                "optimizer_steps": 171,
                "extra": {"training_experiment": "FC-P015", "accumulation_windows": 8,
                          "training_windows": 1368, "optimizer_steps": 171}}
    raise ValueError("dual FNO experiment kind is not supported")


def _checkpoint_identity(root: Path, payload: object, *, role: str,
                         aerodynamic_kind: str = AERO_KIND) -> CheckpointIdentity:
    if not isinstance(payload, dict):
        raise TypeError(f"dual manifest {role} identity must be an object")
    required = {
        "role",
        "checkpoint_relative_directory",
        "model_file",
        "state_file",
        "checkpoint_epoch",
        "model_sha256",
        "state_sha256",
        "metadata_kind",
        "frozen",
    }
    if not required.issubset(payload):
        raise ValueError(f"dual manifest {role} identity is incomplete")
    if payload["role"] != role or payload["frozen"] is not (role == "flow"):
        raise ValueError(f"dual manifest {role} role/frozen contract differs")
    epoch = payload["checkpoint_epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError(f"dual manifest {role} epoch is invalid")
    expected_epoch = 0 if role == "flow" else 1
    expected_kind = FLOW_KIND if role == "flow" else aerodynamic_kind
    if epoch != expected_epoch or payload["metadata_kind"] != expected_kind:
        raise ValueError(f"dual manifest {role} epoch/kind differs")
    directory_value = payload["checkpoint_relative_directory"]
    if not isinstance(directory_value, str) or Path(directory_value).is_absolute():
        raise ValueError(f"dual manifest {role} directory must be relative")
    directory = (root / directory_value).resolve()
    if not directory.is_relative_to(root.resolve()) or not directory.is_dir():
        raise ValueError(f"dual manifest {role} directory is missing or escapes root")
    model = _confined_file(root, str(Path(directory_value) / payload["model_file"]))
    state = _confined_file(root, str(Path(directory_value) / payload["state_file"]))
    expected_model_name = f"FNO.0.{epoch}.mdlus"
    expected_state_name = f"checkpoint.0.{epoch}.pt"
    if model.name != expected_model_name or state.name != expected_state_name:
        raise ValueError(f"dual manifest {role} official checkpoint filenames differ")
    if model.parent != directory or state.parent != directory:
        raise ValueError(f"dual manifest {role} files do not belong to its checkpoint directory")
    official_files = sorted(
        path.name
        for pattern in ("FNO.0.*.mdlus", "checkpoint.0.*.pt")
        for path in directory.glob(pattern)
    )
    if official_files != sorted((expected_model_name, expected_state_name)):
        raise ValueError(f"dual manifest {role} checkpoint directory is ambiguous")
    model_sha = payload["model_sha256"]
    state_sha = payload["state_sha256"]
    if not isinstance(model_sha, str) or not _SHA.fullmatch(model_sha):
        raise ValueError(f"dual manifest {role} model SHA is invalid")
    if not isinstance(state_sha, str) or not _SHA.fullmatch(state_sha):
        raise ValueError(f"dual manifest {role} state SHA is invalid")
    if sha256(model) != model_sha or sha256(state) != state_sha:
        raise ValueError(f"dual manifest {role} checkpoint SHA differs")
    return CheckpointIdentity(
        role=role,
        directory=directory,
        model=model,
        state=state,
        epoch=epoch,
        model_sha256=model_sha,
        state_sha256=state_sha,
        metadata_kind=expected_kind,
    )


def validate_dual_fno_manifest(
    manifest_path: Path, *, expected_sha256: str | None = None
) -> DualFNOIdentity:
    """Validate an explicitly supported checkpoint pair before constructing models."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = sha256(manifest_path)
    if expected_sha256 is not None and manifest_sha != expected_sha256:
        raise ValueError("dual FNO manifest SHA differs")
    payload = _read_object(manifest_path)
    required = {
        "schema_version",
        "status",
        "kind",
        "config_sha256",
        "normalization_sha256",
        "precision_protocol",
        "architecture",
        "flow_parent_model_sha256",
        "flow_parent_state_sha256",
        "aerodynamic_initial_model_sha256",
        "aerodynamic_initial_state_sha256",
        "flow",
        "aerodynamic",
    }
    if not required.issubset(payload):
        raise ValueError("dual FNO manifest is incomplete")
    contract = _experiment_contract(payload["kind"])
    exact = {
        "schema_version": SCHEMA_VERSION,
        "status": contract["status"],
        "config_sha256": CONFIG_SHA256,
        "normalization_sha256": NORMALIZATION_SHA256,
        "flow_parent_model_sha256": FLOW_MODEL_SHA256,
        "flow_parent_state_sha256": FLOW_STATE_SHA256,
        "aerodynamic_initial_model_sha256": FLOW_MODEL_SHA256,
        "aerodynamic_initial_state_sha256": FLOW_STATE_SHA256,
        **contract["extra"],
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ValueError("dual FNO manifest fixed identity differs")
    architecture = payload["architecture"]
    if architecture != ARCHITECTURE:
        raise ValueError("dual FNO architecture differs")
    precision = payload["precision_protocol"]
    if precision != PRECISION_PROTOCOL:
        raise ValueError("dual FNO precision protocol differs")
    flow = _checkpoint_identity(manifest_path.parent, payload["flow"], role="flow")
    aerodynamic = _checkpoint_identity(
        manifest_path.parent, payload["aerodynamic"], role="aerodynamic",
        aerodynamic_kind=contract["aero_kind"],
    )
    if flow.model_sha256 != FLOW_MODEL_SHA256 or flow.state_sha256 != FLOW_STATE_SHA256:
        raise ValueError("dual FNO flow checkpoint is not the frozen P009 parent")
    if flow.directory == aerodynamic.directory or flow.model == aerodynamic.model or flow.state == aerodynamic.state:
        raise ValueError("dual FNO checkpoint roles are not independently persisted")
    return DualFNOIdentity(manifest_path, manifest_sha, flow, aerodynamic, payload)


def validate_dual_runtime_files(
    identity: DualFNOIdentity, *, config_path: Path, normalization_path: Path
) -> None:
    """Bind evaluator runtime inputs to the manifest's immutable identities."""
    if sha256(config_path.resolve()) != identity.payload["config_sha256"]:
        raise ValueError("dual FNO runtime configuration SHA differs")
    if sha256(normalization_path.resolve()) != identity.payload["normalization_sha256"]:
        raise ValueError("dual FNO runtime normalization SHA differs")


def validate_runtime_precision() -> dict[str, Any]:
    """Require the unchanged default-TF32/high protocol used by the parent."""
    import torch

    observed = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if observed != PRECISION_PROTOCOL:
        raise ValueError("dual FNO runtime precision protocol differs")
    return observed


def dual_fno_requested(
    manifest_path: Path | None,
    expected_manifest_sha256: str | None,
    *,
    single_model_calibrated_arguments: tuple[object, ...] = (),
) -> bool:
    """Validate dual/single CLI selection without changing the legacy default."""
    if manifest_path is None:
        if expected_manifest_sha256 is not None:
            raise ValueError("expected dual manifest SHA was provided without a manifest")
        return False
    if not expected_manifest_sha256:
        raise ValueError("dual FNO selection requires an expected manifest SHA")
    if any(value not in (None, False) for value in single_model_calibrated_arguments):
        raise ValueError("single-model calibrated arguments cannot accompany dual FNO")
    return True


def require_dual_training_config(use_dual: bool, path: Path | None) -> None:
    """Keep training-source identity distinct from unchanged evaluation config."""
    if use_dual and path is None:
        raise ValueError("dual FNO evaluation requires the bound training config")
    if not use_dual and path is not None:
        raise ValueError("dual training config cannot accompany a single FNO")


def combine_dual_raw(flow_raw, aerodynamic_raw):
    """Return raw 7-channel output without residual updates or force pooling."""
    if flow_raw.ndim != 4 or aerodynamic_raw.ndim != 4:
        raise ValueError("dual FNO outputs must be four-dimensional")
    if flow_raw.shape != aerodynamic_raw.shape or flow_raw.shape[1] != 7:
        raise ValueError("dual FNO raw output shapes differ")
    if flow_raw.device != aerodynamic_raw.device or flow_raw.dtype != aerodynamic_raw.dtype:
        raise ValueError("dual FNO raw output device/dtype differs")
    import torch

    return torch.cat((flow_raw[:, :3], aerodynamic_raw[:, 3:7]), dim=1)


def make_dual_fno_adapter(flow_model, aerodynamic_model):
    """Wrap two already-loaded official FNOs with the raw-channel adapter."""
    import torch

    class DualFNOAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flow_model = flow_model
            self.aerodynamic_model = aerodynamic_model

        def forward(self, inputs):
            return combine_dual_raw(
                self.flow_model(inputs), self.aerodynamic_model(inputs)
            )

    flow_model.eval().requires_grad_(False)
    aerodynamic_model.eval().requires_grad_(False)
    return DualFNOAdapter().eval()


def load_dual_fno(
    manifest_path: Path,
    cfg,
    device,
    *,
    build_model: Callable,
    load_checkpoint: Callable | None = None,
    expected_manifest_sha256: str | None = None,
):
    """Build and officially load both models, returning a raw-output adapter."""
    identity = validate_dual_fno_manifest(
        manifest_path, expected_sha256=expected_manifest_sha256
    )
    validate_runtime_precision()
    cfg_architecture = {
        key: (list(value) if key == "num_fno_modes" else value)
        for key, value in {
            "in_channels": int(cfg.model.in_channels),
            "out_channels": int(cfg.model.out_channels),
            "latent_channels": int(cfg.model.latent_channels),
            "num_fno_layers": int(cfg.model.num_fno_layers),
            "num_fno_modes": cfg.model.num_fno_modes,
            "decoder_layers": int(cfg.model.decoder_layers),
            "decoder_layer_size": int(cfg.model.decoder_layer_size),
            "padding": int(cfg.model.padding),
            "coord_features": bool(cfg.model.coord_features),
        }.items()
    }
    cfg_architecture["force_channels"] = list(FORCE_CHANNELS)
    if cfg_architecture != ARCHITECTURE:
        raise ValueError("runtime FNO configuration differs from dual manifest")
    if load_checkpoint is None:
        from physicsnemo.utils import load_checkpoint as official_load_checkpoint

        load_checkpoint = official_load_checkpoint
    flow_model = build_model(cfg).to(device)
    aerodynamic_model = build_model(cfg).to(device)
    flow_metadata: dict[str, Any] = {}
    aerodynamic_metadata: dict[str, Any] = {}
    flow_epoch = load_checkpoint(
        identity.flow.directory,
        models=flow_model,
        metadata_dict=flow_metadata,
        device=device,
    )
    aerodynamic_epoch = load_checkpoint(
        identity.aerodynamic.directory,
        models=aerodynamic_model,
        metadata_dict=aerodynamic_metadata,
        device=device,
    )
    if flow_epoch != identity.flow.epoch or aerodynamic_epoch != identity.aerodynamic.epoch:
        raise ValueError("dual FNO loaded checkpoint epoch differs")
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero

    validate_calibrated_epoch_zero(
        identity.flow.directory,
        flow_epoch,
        allow=True,
        expected_model_sha256=identity.flow.model_sha256,
        expected_state_sha256=identity.flow.state_sha256,
        expected_kind=FLOW_KIND,
    )
    contract = _experiment_contract(identity.payload["kind"])
    required_aero_metadata = {
        "status": contract["aero_kind"],
        "checkpoint_epoch": 1,
        "flow_parent_model_sha256": FLOW_MODEL_SHA256,
        "flow_parent_state_sha256": FLOW_STATE_SHA256,
        "aerodynamic_initial_model_sha256": FLOW_MODEL_SHA256,
        "aerodynamic_initial_state_sha256": FLOW_STATE_SHA256,
        "optimizer_steps": contract["optimizer_steps"],
        **contract["extra"],
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    if any(aerodynamic_metadata.get(key) != value for key, value in required_aero_metadata.items()):
        raise ValueError("dual FNO aerodynamic checkpoint metadata differs")
    return make_dual_fno_adapter(flow_model, aerodynamic_model), identity
