"""Project-owned adapter for an independently checkpointed force FNO.

Both wrapped modules remain ordinary PhysicsNeMo FNO instances.  This adapter
only combines their raw output channels: state residuals come from the frozen
flow model and learned force-output channels come from the aerodynamic model.  It does
not apply the residual update, mask, spatial pooling, or physical scaling.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
P018_MANIFEST_STATUS = "FC_P018_DUAL_FNO_MANIFEST_VERIFIED"
P018_SYSTEM_KIND = "FC_P018_REDUCED_RATE_FORCE_FNO"
P018_AERO_KIND = "FC_P018_REDUCED_RATE_FORCE_FNO_AERODYNAMIC_CHECKPOINT"
P018_PROTOCOL_SHA256 = "310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d"
P018_LEARNING_RATE = 1.5625e-7
P026_K1_MANIFEST_STATUS = "FC_P026_K1_DUAL_FNO_MANIFEST_VERIFIED"
P026_K4_MANIFEST_STATUS = "FC_P026_K4_DUAL_FNO_MANIFEST_VERIFIED"
P026_K1_SYSTEM_KIND = "FC_P026_K1_HISTORY_FORCE_FNO"
P026_K4_SYSTEM_KIND = "FC_P026_K4_HISTORY_FORCE_FNO"
P026_K1_AERO_KIND = "FC_P026_K1_HISTORY_AERODYNAMIC_CHECKPOINT"
P026_K4_AERO_KIND = "FC_P026_K4_HISTORY_AERODYNAMIC_CHECKPOINT"
P026_AERO_INITIAL_MODEL_SHA256 = "8a89f4774923afa698328e8e65ae337e7e0efefdae0bc9452d6b7a78758fb70d"
P026_AERO_INITIAL_STATE_SHA256 = "d78d43d43738dd63b9556819e22f6b57c16993affaaff94dc3a29b6250994d6c"
P026_HISTORY_STATE_SHA256 = "2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c"
P026_HISTORY_INFERENCE_SHA256 = "fd568f6457b980046a0419be96562291e9f96736270d45f20dd2adb2ffc5878c"
P026_HISTORY_OBJECTIVE_SHA256 = "4d27fb53e05df73ba94d84bf42ba8205d78ebe6f91de832a68659870ea7d77c0"
P026_ORDER_SHA256 = "177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f"
P026_LEARNING_RATE = 1.5625e-7
P028_MANIFEST_STATUS = "FC_P028_DUAL_FNO_MANIFEST_VERIFIED"
P028_SYSTEM_KIND = "FC_P028_FLOW_ROLLOUT_REPAIR"
P028_FLOW_KIND = "FC_P028_FLOW_ROLLOUT_CHECKPOINT"
P028_PARENT_MANIFEST_SHA256 = "7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7"
P028_AERO_PARENT_MODEL_SHA256 = "e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5"
P028_AERO_PARENT_STATE_SHA256 = "ab2fe103bca0a8c84156e2c9fd7ded5336f2d4236436b593fc414e564d7e92d3"
P028_AERO_PROTOCOL_SHA256 = "daf22b2464744509260f1eb9e0b20d3b80da484c8985f3a22887293bbb40cb30"
P028_LEARNING_RATE = 1e-5
P029_MANIFEST_STATUS = "FC_P029_DUAL_FNO_MANIFEST_VERIFIED"
P029_SYSTEM_KIND = "FC_P029_CONTROL_AWARE_FLOW_REPAIR"
P029_FLOW_KIND = "FC_P029_CONTROL_AWARE_FLOW_CHECKPOINT"
P064_H25_KIND = "FC_P064_B_H25_BOUNDED_CONTROL_AWARE_FLOW_REPAIR"
P064_H25_STATUS = "FC_P064_B_H25_BOUNDED_DUAL_FNO_MANIFEST_VERIFIED"
P064_H25_FLOW_KIND = "FC_P064_B_H25_BOUNDED_FLOW_CHECKPOINT"
P064_B_MANIFEST_SHA256 = "92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891"
P064_B_MODEL_SHA256 = "57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e"
P064_B_STATE_SHA256 = "abc8eb89523d50a2c019d31b7e6a23dda384e9afc7d465d512abe32a80d1e216"
P064_B_PROTOCOL_SHA256 = "7bb41ca973bfabfc01e73796fb744d628490291d268bdf584d2e6ef523b7621e"
P064_MANIFEST_STATUS = {
    "A": "FC_P064_ARM_A_DUAL_FNO_MANIFEST_VERIFIED",
    "B": "FC_P064_ARM_B_DUAL_FNO_MANIFEST_VERIFIED",
    "C": "FC_P064_ARM_C_DUAL_FNO_MANIFEST_VERIFIED",
    "D": "FC_P064_ARM_D_DUAL_FNO_MANIFEST_VERIFIED",
}
P064_SYSTEM_KIND = {
    "A": "FC_P064_ARM_A_CONTROLLED_AERO_FORCE_FNO",
    "B": "FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO",
    "C": "FC_P064_ARM_C_CONTROLLED_AERO_FORCE_FNO",
    "D": "FC_P064_ARM_D_CONTROLLED_AERO_FORCE_FNO",
}
P064_AERO_KIND = {
    "A": "FC_P064_ARM_A_CONTROLLED_AERO_CHECKPOINT",
    "B": "FC_P064_ARM_B_CONTROLLED_AERO_CHECKPOINT",
    "C": "FC_P064_ARM_C_CONTROLLED_AERO_CHECKPOINT",
    "D": "FC_P064_ARM_D_CONTROLLED_AERO_CHECKPOINT",
}
P064_SCHEDULE_SHA256 = {
    "A": "ec1db78eff3807dc3c3d451ba0bb4542ba531fcb4b7a1c15a39e2b1ac15e1b0c",
    "B": "2c7a129724fdaaf6d7dda16eb992d548e92c56ac392d95814dbb57e77320eb55",
    "C": "93537e23ce606732dfd48e78a3b92def987918c0cb71b1b8ca7e8126564081d1",
    "D": "f3f32e70190b8864ca432dc88bacc7efd5c4317ab4a74a67ed6a22613edabfef",
}
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
    if kind == 'FC_P064_TEMPORAL_INCREMENT_AUX_K1_FRESH_FORCE_FNO':
        contract = _experiment_contract(P064_SYSTEM_KIND['B'])
        contract.update(status='FC_P064_TEMPORAL_INCREMENT_AUX_DUAL_FNO_MANIFEST_VERIFIED',
                        aero_kind='FC_P064_TEMPORAL_INCREMENT_AUX_AERODYNAMIC_CHECKPOINT', temporal_increment_aux=True)
        contract['extra']['training_experiment'] = 'FC-P064-TEMPORAL-INCREMENT'
        return contract
    if kind == 'FC_P064_RESPONSE_AUX_K1_FRESH_FORCE_FNO':
        contract = _experiment_contract(P064_SYSTEM_KIND['B'])
        contract.update(status='FC_P064_RESPONSE_AUX_DUAL_FNO_MANIFEST_VERIFIED',
                        aero_kind='FC_P064_RESPONSE_AUX_AERODYNAMIC_CHECKPOINT',
                        response_aux=True)
        contract['extra']['training_experiment'] = 'FC-P064-RESPONSE-AUX'
        return contract
    """Explicit project experiment identities; never infer steps from filenames."""
    if kind == SYSTEM_KIND:
        return {"status": MANIFEST_STATUS, "aero_kind": AERO_KIND,
                "flow_kind": FLOW_KIND, "flow_epoch": 0, "flow_frozen": True,
                "aero_epoch": 1, "aero_frozen": False,
                "optimizer_steps": 1368, "extra": {}}
    if kind == P015_SYSTEM_KIND:
        return {"status": P015_MANIFEST_STATUS, "aero_kind": P015_AERO_KIND,
                "flow_kind": FLOW_KIND, "flow_epoch": 0, "flow_frozen": True,
                "aero_epoch": 1, "aero_frozen": False,
                "optimizer_steps": 171,
                "extra": {"training_experiment": "FC-P015", "accumulation_windows": 8,
                          "training_windows": 1368, "optimizer_steps": 171}}
    if kind == P018_SYSTEM_KIND:
        return {"status": P018_MANIFEST_STATUS, "aero_kind": P018_AERO_KIND,
                "flow_kind": FLOW_KIND, "flow_epoch": 0, "flow_frozen": True,
                "aero_epoch": 1, "aero_frozen": False,
                "optimizer_steps": 171,
                "extra": {"training_experiment": "FC-P018", "accumulation_windows": 8,
                          "training_windows": 1368, "optimizer_steps": 171,
                          "actual_learning_rate": P018_LEARNING_RATE,
                          "training_protocol_sha256": P018_PROTOCOL_SHA256,
                          "training_protocol_file": "training_protocol.json"}}
    p026 = {
        P026_K1_SYSTEM_KIND: (1, P026_K1_MANIFEST_STATUS, P026_K1_AERO_KIND),
        P026_K4_SYSTEM_KIND: (4, P026_K4_MANIFEST_STATUS, P026_K4_AERO_KIND),
    }
    if kind in p026:
        history_k, status, aero_kind = p026[kind]
        return {
            "status": status,
            "aero_kind": aero_kind,
            "flow_kind": FLOW_KIND,
            "flow_epoch": 0,
            "flow_frozen": True,
            "aero_epoch": 1,
            "aero_frozen": False,
            "optimizer_steps": 171,
            "p026_history_k": history_k,
            "extra": {
                "training_experiment": "FC-P026",
                "accumulation_windows": 8,
                "training_windows": 1368,
                "optimizer_steps": 171,
                "actual_learning_rate": P026_LEARNING_RATE,
                "training_protocol_file": "training_protocol.json",
            },
        }
    for arm in ("A", "B", "C", "D"):
        if kind == P064_SYSTEM_KIND[arm]:
            return {
                "status": P064_MANIFEST_STATUS[arm],
                "aero_kind": P064_AERO_KIND[arm],
                "flow_kind": FLOW_KIND,
                "flow_epoch": 0,
                "flow_frozen": True,
                "aero_epoch": 1,
                "aero_frozen": False,
                "optimizer_steps": 32,
                "p026_history_k": 1,
                "p064_arm": arm,
                "extra": {
                    "training_experiment": "FC-P064",
                    "arm": arm,
                    "accumulation_windows": 8,
                    "training_windows": 256,
                    "optimizer_steps": 32,
                    "actual_learning_rate": P026_LEARNING_RATE,
                    "training_protocol_file": "training_protocol.json",
                },
            }
    if kind == P064_H25_KIND:
        return {
            "status": P064_H25_STATUS, "aero_kind": P064_AERO_KIND["B"],
            "flow_kind": P064_H25_FLOW_KIND, "flow_epoch": 1, "flow_frozen": False,
            "aero_epoch": 1, "aero_frozen": True, "optimizer_steps": 32,
            "p026_history_k": 1, "flow_repair": True, "p064_b_h25": True,
            "extra": {"training_experiment": "FC-P064-B-H25-BOUNDED",
                      "accumulation_windows": 8, "training_windows": 256,
                      "optimizer_steps": 32, "actual_learning_rate": P028_LEARNING_RATE,
                      "training_protocol_file": "training_protocol.json",
                      "parent_manifest_sha256": P064_B_MANIFEST_SHA256,
                      "scientific_admission": False},
        }
    if kind in (P028_SYSTEM_KIND, P029_SYSTEM_KIND):
        p029 = kind == P029_SYSTEM_KIND
        return {
            "status": P029_MANIFEST_STATUS if p029 else P028_MANIFEST_STATUS,
            "aero_kind": P026_K1_AERO_KIND,
            "flow_kind": P029_FLOW_KIND if p029 else P028_FLOW_KIND,
            "flow_epoch": 1,
            "flow_frozen": False,
            "aero_epoch": 1,
            "aero_frozen": True,
            "optimizer_steps": 171,
            "p026_history_k": 1,
            "flow_repair": True,
            "p029": p029,
            "extra": {
                "training_experiment": "FC-P029" if p029 else "FC-P028",
                "accumulation_windows": 8,
                "training_windows": 1368,
                "optimizer_steps": 171,
                "actual_learning_rate": P028_LEARNING_RATE,
                "training_protocol_file": "training_protocol.json",
                "parent_manifest_sha256": P028_PARENT_MANIFEST_SHA256,
            },
        }
    raise ValueError("dual FNO experiment kind is not supported")


def _p026_history_input(history_k: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": f"p026_k{history_k}",
        "history_length": history_k,
        "flow_input_channels": 6,
        "aerodynamic_input_channels": 6 if history_k == 1 else 18,
        "left_padding": "trajectory_frame0",
        "autoregressive_state_source": "frozen_flow_prediction",
        "future_state_inputs": False,
        "future_force_inputs": False,
    }


def _p026_inventory() -> dict[str, Any]:
    return {
        "windows": 1368,
        "warm": 1300,
        "padded": 68,
        "family_windows": [720, 408, 240],
        "family_padded": [20, 16, 32],
    }


def _validate_p026_protocol(
    root: Path, payload: dict[str, Any], contract: dict[str, Any]
) -> None:
    history_k = contract.get("p026_history_k")
    if history_k is None:
        return
    history_input = _p026_history_input(history_k)
    aero_architecture = dict(ARCHITECTURE)
    aero_architecture["in_channels"] = 6 if history_k == 1 else 18
    if contract.get("p064_arm"):
        arm = contract["p064_arm"]
        exact = {
            "history_input": history_input,
            "parent_history_inventory": _p026_inventory(),
            "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
            "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
            "flow_architecture": ARCHITECTURE,
            "aerodynamic_architecture": aero_architecture,
        }
        if any(payload.get(key) != value for key, value in exact.items()):
            raise ValueError("P064 history, parent inventory, or role architecture differs")
        protocol_path = _confined_file(root, payload.get("training_protocol_file"))
        protocol = _read_object(protocol_path)
        protocol_sha = sha256(protocol_path)
        expected = {
            "training_experiment": "FC-P064",
            "arm": arm,
            "parent_experiment": "FC-P026-K1",
            "history_input": history_input,
            "training_windows": 256,
            "accumulation_windows": 8,
            "optimizer_steps": 32,
            "learning_rate": P026_LEARNING_RATE,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "seed": 20261003,
            "chunk_size": 10,
            "rollout_steps": 100,
            "parent_sampler_order_sha256": P026_ORDER_SHA256,
            "schedule_sha256": P064_SCHEDULE_SHA256[arm],
            "diagnostic_counts": [0, 256],
            "objective": "equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE",
            "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
            "history_objective_sha256": P026_HISTORY_OBJECTIVE_SHA256,
            "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
            "action_semantics": "stored_prescribed_action_samples_not_exact_nominal_time_commands",
            "controlled_b00_action_semantics": (
                "actual_closed_loop_applied_endpoint_omega_samples"
                if arm in ("B", "C")
                else "not_applicable_no_b00_windows"
            ),
            "b00_windows": {"A": 0, "B": 64, "C": 128, "D": 32}[arm],
            "b00_weight": {"A": 0.0, "B": 0.25, "C": 0.5, "D": 0.125}[arm],
            "replacement_within_each_update": {
                "A": [], "B": [0, 4], "C": [0, 2, 4, 6], "D": [0, 4]
            }[arm],
            "allocator_fraction": 0.06,
            "wall_seconds": 3600,
            "validation_accessed": False,
            "frozen_test_accessed": False,
            "selection_performed": False,
        }
        if arm == "D":
            expected.update(
                controlled_b00_action_semantics="actual_k1_projected_closed_loop_applied_endpoint_omega_samples",
                b02_windows=32,
                b02_weight=0.125,
                controlled_b02_action_semantics="actual_symmetry_canonical_closed_loop_applied_endpoint_omega_samples",
                controlled_source_profile="32_b00_k1_projected_plus_32_b02_canonical_policy",
            )
        if contract.get('response_aux'):
            auxiliary = protocol.get('auxiliary_response')
            digest = hashlib.sha256(json.dumps(auxiliary, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if digest != '778e442b5c971db169788e9ac91ea27aaa5335282f1403a7808f620bdce56c52':
                raise ValueError('fixed response auxiliary protocol differs')
            expected.update(training_experiment='FC-P064-RESPONSE-AUX',
                            auxiliary_response=auxiliary, candidate_profile='FC_P064_RESPONSE_AUX_K1_FRESH')
        if contract.get('temporal_increment_aux'):
            expected.update(training_experiment='FC-P064-TEMPORAL-INCREMENT',
                objective='original_mixed_absolute_plus_H1_temporal_increment',
                diagnostic_objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',
                mixed_forward_preserved=True, candidate_profile='FC_P064_TEMPORAL_INCREMENT_AUX_K1_FRESH',
                temporal_increment=dict(weight=1.0, dt=0.1, divide_by_dt=False,
                    edges=99, within_chunk_edges=90, recomputed_boundary_edges=9,
                    extra_aero_calls_per_window=9, extra_aero_samples_per_window=18,
                    both_endpoints_require_gradient=True, normalization='unchanged_train_force_std'),
                history_objective_sha256='64aa1c04cb064b4482baed2a816c7a1af5c45708d30079f785d60375a78e79cd')
        if (
            protocol != expected
            or payload.get("training_protocol_sha256") != protocol_sha
            or payload.get("training_semantics") != protocol
        ):
            raise ValueError("P064 training protocol bytes or values differ")
        return
    exact = {
        "history_input": history_input,
        "history_inventory": _p026_inventory(),
        "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
        "flow_architecture": ARCHITECTURE,
        "aerodynamic_architecture": aero_architecture,
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ValueError("P026 history or role architecture contract differs")
    protocol_path = _confined_file(root, payload.get("training_protocol_file"))
    protocol_sha = sha256(protocol_path)
    protocol = _read_object(protocol_path)
    if (
        payload.get("training_protocol_sha256") != protocol_sha
        or payload.get("training_semantics") != protocol
    ):
        raise ValueError("P026 training protocol bytes or semantics differ")
    if contract.get("flow_repair"):
        protocol_exact = {
            "experiment": "FC-P028",
            "optimized_role": "flow",
            "fixed_role": "aerodynamic",
            "horizon": 10,
            "original_window_horizon": 100,
            "training_windows": 1368,
            "optimizer_steps": 171,
            "accumulation_windows": 8,
            "seed": 20261003,
            "learning_rate": P028_LEARNING_RATE,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "objective": "ten_equal_masked_normalized_state_MSE",
            "sampler_order_sha256": P026_ORDER_SHA256,
            "force_loss": False,
            "terminal_selection": False,
            "future_truth_inputs": False,
        }
        if contract.get("p064_b_h25"):
            protocol_exact.update(
                experiment="FC-P064-B-H25-BOUNDED", horizon=25,
                training_windows=256, optimizer_steps=32,
                selected_training_order_sha256="06c922e8f1476af52705fcc88521bc039d3fa6bd7a9c00b130179af712e36691",
                objective="half_parent_scaled_field_MSE_plus_half_parent_scaled_four_force_MSE",
                field_weight=0.5, force_weight=0.5, scale_windows=1368,
                scale_horizon=10, scale_source="same_P064_B_parent_original_H10_1368_windows",
                force_loss=True,
                force_timing="aero_current_state_and_current_next_action_predicts_next_force",
            )
        if contract.get("p029") or contract.get("p064_b_h25"):
            if contract.get("p029"):
                protocol_exact.update(
                    experiment="FC-P029",
                    objective="half_parent_scaled_field_MSE_plus_half_parent_scaled_four_force_MSE",
                    field_weight=0.5, force_weight=0.5, scale_windows=1368,
                    force_loss=True,
                    force_timing="aero_current_state_and_current_next_action_predicts_next_force",
                )
            scales = payload.get("fixed_scales")
            if (not isinstance(scales, dict) or set(scales) != {"field", "force"}
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                           for v in scales.values())):
                raise ValueError("P029 fixed scales must be finite positive numbers")
            scale_sha = payload.get("scales_receipt_sha256")
            if not isinstance(scale_sha, str) or re.fullmatch(r"[0-9a-f]{64}", scale_sha) is None:
                raise ValueError("P029 scales receipt SHA differs")
            protocol_exact.update(fixed_scales=scales, scales_receipt_sha256=scale_sha)
            for key, expected in protocol_exact.items():
                if type(expected) in (bool, int) and type(protocol.get(key)) is not type(expected):
                    raise ValueError("P029 fixed protocol scalar type differs")
        if protocol != protocol_exact:
            raise ValueError("P028 training protocol values differ")
        return
    protocol_exact = {
        "training_experiment": "FC-P026",
        "history_input": history_input,
        "training_windows": 1368,
        "accumulation_windows": 8,
        "optimizer_steps": 171,
        "learning_rate": P026_LEARNING_RATE,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "seed": 20261003,
        "chunk_size": 10,
        "rollout_steps": 100,
        "sampler_order_sha256": P026_ORDER_SHA256,
        "diagnostic_counts": [0, 456, 912, 1368],
        "objective": "equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE",
        "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
        "action_semantics": "stored_prescribed_action_samples_not_exact_nominal_time_commands",
        "inventory": _p026_inventory(),
        "allocator_fraction": 0.06,
        "wall_seconds": 14400,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "selection_performed": False,
    }
    if set(protocol) != set(protocol_exact) | {"history_objective_sha256"}:
        raise ValueError("P026 training protocol keys differ")
    if any(protocol.get(key) != value for key, value in protocol_exact.items()):
        raise ValueError("P026 training protocol values differ")
    if protocol["history_objective_sha256"] != P026_HISTORY_OBJECTIVE_SHA256:
        raise ValueError("P026 history objective SHA differs")


def _validate_p018_protocol(root: Path, payload: dict[str, Any]) -> None:
    if payload.get("kind") != P018_SYSTEM_KIND:
        return
    protocol_path = _confined_file(root, payload.get("training_protocol_file"))
    if sha256(protocol_path) != P018_PROTOCOL_SHA256:
        raise ValueError("P018 actual training protocol SHA differs")
    protocol = _read_object(protocol_path)
    if (protocol.get("training_experiment") != "FC-P018"
            or protocol.get("base_config_sha256") != CONFIG_SHA256
            or protocol.get("sole_optimizer_override") != {"learning_rate": P018_LEARNING_RATE}
            or payload.get("actual_learning_rate") != P018_LEARNING_RATE
            or payload.get("training_protocol_sha256") != P018_PROTOCOL_SHA256
            or payload.get("training_semantics", {}).get("learning_rate") != P018_LEARNING_RATE):
        raise ValueError("P018 effective learning rate or protocol differs")


def _checkpoint_identity(
    root: Path, payload: object, *, role: str, expected_kind: str,
    expected_epoch: int, expected_frozen: bool,
) -> CheckpointIdentity:
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
    if payload["role"] != role or payload["frozen"] is not expected_frozen:
        raise ValueError(f"dual manifest {role} role/frozen contract differs")
    epoch = payload["checkpoint_epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError(f"dual manifest {role} epoch is invalid")
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
    manifest_path: Path,
    *,
    expected_sha256: str | None = None,
    allow_engineering_fixture: bool = False,
) -> DualFNOIdentity:
    """Validate an explicitly supported checkpoint pair before constructing models."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = sha256(manifest_path)
    if expected_sha256 is not None and manifest_sha != expected_sha256:
        raise ValueError("dual FNO manifest SHA differs")
    payload = _read_object(manifest_path)
    if payload.get("engineering_fixture_not_candidate") is True and not allow_engineering_fixture:
        raise ValueError("engineering fixture is not a scientific candidate")
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
        "aerodynamic_initial_model_sha256": (
            P028_AERO_PARENT_MODEL_SHA256
            if contract.get("p064_arm") or contract.get("p064_b_h25")
            else P026_AERO_INITIAL_MODEL_SHA256
            if contract.get("p026_history_k") is not None
            else FLOW_MODEL_SHA256
        ),
        "aerodynamic_initial_state_sha256": (
            P028_AERO_PARENT_STATE_SHA256
            if contract.get("p064_arm") or contract.get("p064_b_h25")
            else P026_AERO_INITIAL_STATE_SHA256
            if contract.get("p026_history_k") is not None
            else FLOW_STATE_SHA256
        ),
        **contract["extra"],
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise ValueError("dual FNO manifest fixed identity differs")
    if contract.get("flow_repair") and (
        payload.get("aerodynamic_parent_model_sha256") != (P064_B_MODEL_SHA256 if contract.get("p064_b_h25") else P028_AERO_PARENT_MODEL_SHA256)
        or payload.get("aerodynamic_parent_state_sha256") != (P064_B_STATE_SHA256 if contract.get("p064_b_h25") else P028_AERO_PARENT_STATE_SHA256)
    ):
        raise ValueError("P028 aerodynamic parent identity differs")
    _validate_p018_protocol(manifest_path.parent, payload)
    _validate_p026_protocol(manifest_path.parent, payload, contract)
    architecture = payload["architecture"]
    if architecture != ARCHITECTURE:
        raise ValueError("dual FNO architecture differs")
    precision = payload["precision_protocol"]
    if precision != PRECISION_PROTOCOL:
        raise ValueError("dual FNO precision protocol differs")
    flow = _checkpoint_identity(
        manifest_path.parent, payload["flow"], role="flow",
        expected_kind=contract["flow_kind"], expected_epoch=contract["flow_epoch"],
        expected_frozen=contract["flow_frozen"],
    )
    aerodynamic = _checkpoint_identity(
        manifest_path.parent, payload["aerodynamic"], role="aerodynamic",
        expected_kind=contract["aero_kind"], expected_epoch=contract["aero_epoch"],
        expected_frozen=contract["aero_frozen"],
    )
    if not contract.get("flow_repair") and (
        flow.model_sha256 != FLOW_MODEL_SHA256 or flow.state_sha256 != FLOW_STATE_SHA256
    ):
        raise ValueError("dual FNO flow checkpoint is not the frozen P009 parent")
    if contract.get("flow_repair") and (
        aerodynamic.model_sha256 != (P064_B_MODEL_SHA256 if contract.get("p064_b_h25") else P028_AERO_PARENT_MODEL_SHA256)
        or aerodynamic.state_sha256 != (P064_B_STATE_SHA256 if contract.get("p064_b_h25") else P028_AERO_PARENT_STATE_SHA256)
    ):
        raise ValueError("P028 aerodynamic checkpoint is not the frozen P026 K1 parent")
    if flow.directory == aerodynamic.directory or flow.model == aerodynamic.model or flow.state == aerodynamic.state:
        raise ValueError("dual FNO checkpoint roles are not independently persisted")
    return DualFNOIdentity(manifest_path, manifest_sha, flow, aerodynamic, payload)


def validate_dual_runtime_files(
    identity: DualFNOIdentity, *, config_path: Path, normalization_path: Path
) -> None:
    """Bind evaluator runtime inputs to the manifest's immutable identities."""
    _validate_p018_protocol(identity.manifest_path.parent, identity.payload)
    _validate_p026_protocol(
        identity.manifest_path.parent,
        identity.payload,
        _experiment_contract(identity.payload["kind"]),
    )
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


def _runtime_architecture(cfg) -> dict[str, Any]:
    architecture = {
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
    architecture["force_channels"] = list(FORCE_CHANNELS)
    return architecture


def _p026_history_adapter_factory() -> Callable:
    """Bind the imported project glue bytes before any P026 model is exposed."""
    import p026_history_inference
    import p026_state_history

    inference_path = Path(p026_history_inference.__file__).resolve()
    state_path = Path(p026_state_history.__file__).resolve()
    if (
        not inference_path.is_file()
        or sha256(inference_path) != P026_HISTORY_INFERENCE_SHA256
        or not state_path.is_file()
        or sha256(state_path) != P026_HISTORY_STATE_SHA256
    ):
        raise ValueError("P026 imported history module source SHA differs")
    return p026_history_inference.make_history_dual_fno_adapter


def load_dual_fno(
    manifest_path: Path,
    cfg,
    device,
    *,
    build_model: Callable,
    load_checkpoint: Callable | None = None,
    expected_manifest_sha256: str | None = None,
    allow_engineering_fixture: bool = False,
):
    """Build and officially load both models, returning a raw-output adapter."""
    identity = validate_dual_fno_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        allow_engineering_fixture=allow_engineering_fixture,
    )
    validate_runtime_precision()
    cfg_architecture = _runtime_architecture(cfg)
    if cfg_architecture != ARCHITECTURE:
        raise ValueError("runtime FNO configuration differs from dual manifest")
    if load_checkpoint is None:
        from physicsnemo.utils import load_checkpoint as official_load_checkpoint

        load_checkpoint = official_load_checkpoint
    contract = _experiment_contract(identity.payload["kind"])
    history_k = contract.get("p026_history_k")
    aerodynamic_cfg = cfg
    if history_k is not None:
        if identity.payload["flow_architecture"] != cfg_architecture:
            raise ValueError("P026 runtime flow architecture differs")
        aerodynamic_cfg = copy.deepcopy(cfg)
        aerodynamic_cfg.model.in_channels = 6 if history_k == 1 else 18
        if _runtime_architecture(aerodynamic_cfg) != identity.payload[
            "aerodynamic_architecture"
        ]:
            raise ValueError("P026 runtime aerodynamic architecture differs")
    flow_model = build_model(cfg).to(device)
    aerodynamic_model = build_model(aerodynamic_cfg).to(device)
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
    if not contract.get("flow_repair"):
        from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero

        validate_calibrated_epoch_zero(
            identity.flow.directory,
            flow_epoch,
            allow=True,
            expected_model_sha256=identity.flow.model_sha256,
            expected_state_sha256=identity.flow.state_sha256,
            expected_kind=FLOW_KIND,
        )
    else:
        required_flow_metadata = {
            "status": contract["flow_kind"],
            "training_experiment": contract["extra"]["training_experiment"],
            "checkpoint_epoch": 1,
            "training_protocol_sha256": identity.payload["training_protocol_sha256"],
            "training_protocol_file": "training_protocol.json",
            "accumulation_windows": 8,
            "training_windows": 1368,
            "optimizer_steps": 171,
            "actual_learning_rate": P028_LEARNING_RATE,
            "parent_manifest_sha256": P028_PARENT_MANIFEST_SHA256,
            "flow_parent_model_sha256": FLOW_MODEL_SHA256,
            "flow_parent_state_sha256": FLOW_STATE_SHA256,
            "aerodynamic_parent_model_sha256": P028_AERO_PARENT_MODEL_SHA256,
            "aerodynamic_parent_state_sha256": P028_AERO_PARENT_STATE_SHA256,
        }
        if contract.get("p064_b_h25"):
            required_flow_metadata.update(
                training_windows=256, optimizer_steps=32,
                parent_manifest_sha256=P064_B_MANIFEST_SHA256,
                aerodynamic_parent_model_sha256=P064_B_MODEL_SHA256,
                aerodynamic_parent_state_sha256=P064_B_STATE_SHA256,
            )
        if contract.get("p029") or contract.get("p064_b_h25"):
            required_flow_metadata.update(
                fixed_scales=identity.payload["fixed_scales"],
                scales_receipt_sha256=identity.payload["scales_receipt_sha256"],
            )
        if any(flow_metadata.get(key) != value for key, value in required_flow_metadata.items()):
            raise ValueError("P028 flow checkpoint metadata differs")
    required_aero_metadata = {
        "status": contract["aero_kind"],
        "checkpoint_epoch": 1,
        "flow_parent_model_sha256": FLOW_MODEL_SHA256,
        "flow_parent_state_sha256": FLOW_STATE_SHA256,
        "aerodynamic_initial_model_sha256": (
            P026_AERO_INITIAL_MODEL_SHA256
            if history_k is not None
            else FLOW_MODEL_SHA256
        ),
        "aerodynamic_initial_state_sha256": (
            P026_AERO_INITIAL_STATE_SHA256
            if history_k is not None
            else FLOW_STATE_SHA256
        ),
        "optimizer_steps": contract["optimizer_steps"],
        **contract["extra"],
        "selection_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    if contract.get("p064_arm") or contract.get("p064_b_h25"):
        arm = contract.get("p064_arm", "B")
        required_aero_metadata = {
            "status": P064_AERO_KIND[arm],
            "checkpoint_epoch": 1,
            "training_experiment": "FC-P064",
            "arm": arm,
            "history_profile": "p026_k1",
            "history_k": 1,
            "model_in_channels": 6,
            "training_protocol_sha256": (P064_B_PROTOCOL_SHA256 if contract.get("p064_b_h25") else identity.payload["training_protocol_sha256"]),
            "training_protocol_file": "training_protocol.json",
            "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
            "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
            "training_windows": 256,
            "optimizer_steps": 32,
            "accumulation_windows": 8,
            "actual_learning_rate": P026_LEARNING_RATE,
            "parent_sampler_order_sha256": P026_ORDER_SHA256,
            "schedule_sha256": P064_SCHEDULE_SHA256[arm],
            "parent_history_inventory": _p026_inventory(),
            "flow_parent_model_sha256": FLOW_MODEL_SHA256,
            "flow_parent_state_sha256": FLOW_STATE_SHA256,
            "aerodynamic_parent_model_sha256": P028_AERO_PARENT_MODEL_SHA256,
            "aerodynamic_parent_state_sha256": P028_AERO_PARENT_STATE_SHA256,
            "selection_performed": False,
            "validation_accessed": False,
            "frozen_test_accessed": False,
            "ppo_executed": False,
        }
    elif history_k is not None and not contract.get("flow_repair"):
        required_aero_metadata.update(
            {
                "history_profile": f"p026_k{history_k}",
                "history_k": history_k,
                "model_in_channels": 6 if history_k == 1 else 18,
                "training_protocol_sha256": identity.payload[
                    "training_protocol_sha256"
                ],
                "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
                "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
                "sampler_order_sha256": P026_ORDER_SHA256,
                "history_inventory": _p026_inventory(),
            }
        )
    if contract.get("flow_repair") and not contract.get("p064_b_h25"):
        required_aero_metadata = {
            "status": P026_K1_AERO_KIND,
            "checkpoint_epoch": 1,
            "flow_parent_model_sha256": FLOW_MODEL_SHA256,
            "flow_parent_state_sha256": FLOW_STATE_SHA256,
            "aerodynamic_initial_model_sha256": P026_AERO_INITIAL_MODEL_SHA256,
            "aerodynamic_initial_state_sha256": P026_AERO_INITIAL_STATE_SHA256,
            "optimizer_steps": 171,
            "training_experiment": "FC-P026",
            "accumulation_windows": 8,
            "training_windows": 1368,
            "actual_learning_rate": P026_LEARNING_RATE,
            "training_protocol_file": "training_protocol.json",
            "history_profile": "p026_k1",
            "history_k": 1,
            "model_in_channels": 6,
            "training_protocol_sha256": P028_AERO_PROTOCOL_SHA256,
            "history_state_module_sha256": P026_HISTORY_STATE_SHA256,
            "history_inference_module_sha256": P026_HISTORY_INFERENCE_SHA256,
            "sampler_order_sha256": P026_ORDER_SHA256,
            "history_inventory": _p026_inventory(),
            "selection_performed": False,
            "validation_accessed": False,
            "frozen_test_accessed": False,
            "ppo_executed": False,
        }
    if contract.get('response_aux'):
        required_aero_metadata.update(status='FC_P064_RESPONSE_AUX_AERODYNAMIC_CHECKPOINT',
                                      training_experiment='FC-P064-RESPONSE-AUX',
                                      auxiliary_profile='FC_P064_RESPONSE_AUX_K1_FRESH')
    if contract.get('temporal_increment_aux'):
        required_aero_metadata.update(status='FC_P064_TEMPORAL_INCREMENT_AUX_AERODYNAMIC_CHECKPOINT',
                                      training_experiment='FC-P064-TEMPORAL-INCREMENT')
    if any(
        aerodynamic_metadata.get(key) != value
            for key, value in required_aero_metadata.items()
        ):
            raise ValueError("P028 frozen aerodynamic checkpoint metadata differs")
    elif any(
        aerodynamic_metadata.get(key) != value
        for key, value in required_aero_metadata.items()
    ):
        raise ValueError("dual FNO aerodynamic checkpoint metadata differs")
    if history_k is None:
        return make_dual_fno_adapter(flow_model, aerodynamic_model), identity
    make_history_dual_fno_adapter = _p026_history_adapter_factory()

    return (
        make_history_dual_fno_adapter(flow_model, aerodynamic_model, k=history_k),
        identity,
    )
