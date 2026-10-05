#!/usr/bin/env python3
"""Audit the train-only FC-P008 force-row candidate and its real lineage."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
CANDIDATE_NAME = "fcp008_force_readout_candidate_20261005"
CANDIDATE_KIND = "full_train_force_row_recalibration"
RESULT_STATUS = "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE_COMPLETE_NOT_ADMISSION"
LAUNCH_STATUS = "FC_P008_FORCE_READOUT_LAUNCH_VERIFIED"
COMPLETION_STATUS = "FC_P008_FORCE_READOUT_CANDIDATE_VERIFIED"
LINEAGE_STATUS = "FC_P008_CANDIDATE_LINEAGE_PASS"
APPROVAL_STATUS = "FC_P008_FULL_EXECUTION_APPROVED"
APPROVAL_CONTRACT_COMMIT = "729e5201833c491de5c8648f42abc9d565f7a0eb"
APPROVAL_CONTRACT_SHA256 = "1e20a2bd1ea008571e3c8768d2e8ee75e8d2ff6860069bdf6f39d66fda3a884c"
PARENT_MODEL = "f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4"
PARENT_STATE = "a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a"
BASE_SPLIT = "1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89"
BASE_MANIFEST = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
TRAIN8 = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
TRAIN16 = "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
NORM = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
CONFIG = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
MAPPING = "57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92"
ALPHAS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]
CHANNELS = ["front_cd", "front_cl", "rear_cd", "rear_cl"]
FAMILY_EXPOSURES = {"base20": 720, "train8": 408, "train16": 240}
FAMILY_ENDPOINTS = {"base20": 16000, "train8": 1600, "train16": 2048}
PRECISION = {
    "NVIDIA_TF32_OVERRIDE": None,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "high",
}
FORMAL_PROTOCOL = [
    "validation10_H1_H10_H50_H100_stride25_batch4",
    "dynamic6_H1_H10_H50_H100_stride1_batch8",
    "force_window6",
    "unchanged_development_gate",
]
REQUIRED_SOURCE_FILES = {
    "scripts/build_fcp008_force_readout_candidate.py",
    "scripts/spark_gpu_guard.py",
    "scripts/evaluate_tandem_fno.py",
    "scripts/train_tandem_fno.py",
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


def finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def archive_payload(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected = {"model.pt", "args.json", "metadata.json"}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("official model archive member set differs")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}


def model_state(path: Path) -> dict:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("CPU torch is required for tensor confinement audit") from error
    with zipfile.ZipFile(path) as archive:
        value = torch.load(
            io.BytesIO(archive.read("model.pt")), map_location="cpu", weights_only=False
        )
    if not isinstance(value, dict):
        raise TypeError("official model payload is not a state mapping")
    return value


def training_state(path: Path) -> dict:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("CPU torch is required for training-state audit") from error
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError("official training state is not an object")
    return value


def tensor_sha256(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_tensor_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def verify_force_row_confinement(parent_path: Path, candidate_path: Path) -> list[str]:
    import torch

    parent, candidate = model_state(parent_path), model_state(candidate_path)
    if list(parent) != list(candidate):
        raise ValueError("candidate tensor inventory differs")
    weight = "decoder_net.final_layer.linear.weight"
    bias = "decoder_net.final_layer.linear.bias"
    changed = []
    for name in parent:
        before, after = parent[name], candidate[name]
        if before.dtype != after.dtype or before.shape != after.shape:
            raise ValueError(f"candidate tensor structure differs: {name}")
        if name == weight:
            if tuple(before.shape) != (7, 128) or not torch.equal(before[:3], after[:3]):
                raise ValueError("non-force weight rows changed")
        elif name == bias:
            if tuple(before.shape) != (7,) or not torch.equal(before[:3], after[:3]):
                raise ValueError("non-force bias rows changed")
        elif not torch.equal(before, after):
            raise ValueError(f"non-force tensor changed: {name}")
        if not torch.equal(before, after):
            changed.append(name)
    if changed != [weight, bias]:
        raise ValueError("force-row changed tensor inventory differs")
    return changed


def read_source_manifest(path: Path, snapshot: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("source manifest syntax differs")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in entries:
            raise ValueError("source manifest path is unsafe or duplicated")
        item = snapshot / relative
        if not item.is_file() or sha256(item) != digest:
            raise ValueError(f"source snapshot member differs: {name}")
        entries[name] = digest
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if set(entries) != actual or not REQUIRED_SOURCE_FILES.issubset(entries):
        raise ValueError("source snapshot closure differs")
    return entries


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_static(repo: Path) -> dict[str, str]:
    paths = {
        "approval_contract": (repo / "docs/FC_P008_APPROVAL_20261005.md", APPROVAL_CONTRACT_SHA256),
        "base_manifest": (repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/manifest.json", BASE_MANIFEST),
        "base_train_split": (repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/splits/train.json", BASE_SPLIT),
        "train8_manifest": (repo / "data/curated/tandem_cylinders_dynamic_train8_v1/manifest.json", TRAIN8),
        "train16_manifest": (repo / "data/curated/tandem_cylinders_directppo_train16_v1/manifest.json", TRAIN16),
        "normalization": (repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json", NORM),
        "resolved_config": (repo / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/resolved_config.yaml", CONFIG),
        "source_phase_mapping": (repo / "artifacts/fcp003c_full_train_source_phase_mapping_20261005/mapping.json", MAPPING),
        "parent_model": (repo / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best/FNO.0.2.mdlus", PARENT_MODEL),
        "parent_state": (repo / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best/checkpoint.0.2.pt", PARENT_STATE),
    }
    for name, (path, digest) in paths.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"static identity differs: {name}")
    return {name: digest for name, (_, digest) in paths.items()}


def validate_candidate(repo: Path, candidate: Path, approval_sha256: str) -> dict:
    repo, candidate = repo.resolve(), candidate.resolve()
    if candidate != (repo / "artifacts" / CANDIDATE_NAME).resolve():
        raise ValueError("FC-P008 candidate root differs")
    if not re.fullmatch(r"[0-9a-f]{64}", approval_sha256):
        raise ValueError("reviewed execution approval SHA is required")
    static = validate_static(repo)
    approval_path = candidate / "launch_evidence/fc_p008_execution_approval.json"
    if not approval_path.is_file() or sha256(approval_path) != approval_sha256:
        raise ValueError("execution approval identity differs")
    approval = load(approval_path)
    launch = load(candidate / "launch_receipt.json")
    completion = load(candidate / "completion_receipt.json")
    build = candidate / "candidate_build"
    result = load(build / "result.json")
    if (
        approval.get("status") != APPROVAL_STATUS
        or approval.get("formal_evaluation_authorized") is not False
        or approval.get("validation_accessed") is not False
        or approval.get("frozen_test_accessed") is not False
        or approval.get("ppo_executed") is not False
    ):
        raise ValueError("execution approval contract differs")
    implementation = candidate / "source_snapshot/scripts/build_fcp008_force_readout_candidate.py"
    implementation_sha = sha256(implementation)
    expected_launch = {
        "status": LAUNCH_STATUS,
        "execution_approval_sha256": approval_sha256,
        "implementation_sha256": implementation_sha,
        "runtime_image_id": IMAGE_ID,
        "resolved_config_sha256": CONFIG,
        "normalization_sha256": NORM,
        "parent_model_sha256": PARENT_MODEL,
        "parent_training_state_sha256": PARENT_STATE,
        "source_phase_mapping_sha256": MAPPING,
        "base_manifest_sha256": BASE_MANIFEST,
        "base_train_split_sha256": BASE_SPLIT,
        "train8_manifest_sha256": TRAIN8,
        "train16_manifest_sha256": TRAIN16,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    if any(launch.get(key) != value for key, value in expected_launch.items()):
        raise ValueError("FC-P008 launch receipt differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(launch.get("source_commit", ""))):
        raise ValueError("FC-P008 source commit differs")
    if approval.get("implementation_sha256") != implementation_sha:
        raise ValueError("execution approval implementation differs")
    manifest_path = candidate / "source_snapshot.sha256"
    if launch.get("source_snapshot_manifest_sha256") != sha256(manifest_path):
        raise ValueError("source snapshot manifest identity differs")
    source_files = read_source_manifest(manifest_path, candidate / "source_snapshot")
    if source_files["scripts/build_fcp008_force_readout_candidate.py"] != implementation_sha:
        raise ValueError("source snapshot does not bind implementation")
    expected_result = {
        "status": RESULT_STATUS,
        "source_phase_mapping_sha256": MAPPING,
        "approval_contract_commit": APPROVAL_CONTRACT_COMMIT,
        "approval_contract_sha256": APPROVAL_CONTRACT_SHA256,
        "execution_approval_sha256": approval_sha256,
        "family_exposures": FAMILY_EXPOSURES,
        "family_endpoint_counts": FAMILY_ENDPOINTS,
        "precision_protocol": PRECISION,
        "parent_model_sha256": PARENT_MODEL,
        "parent_training_state_sha256": PARENT_STATE,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "candidate_checkpoint_epoch": 0,
        "implementation_sha256": implementation_sha,
        "optimizer_steps": 0,
        "architecture_changed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "formal_evaluation_authorized": False,
        "state_rows_0_3_byte_identical": True,
        "all_other_tensors_byte_identical": True,
        "calibration_fit_performed": True,
        "no_optimizer_training": True,
        "official_training_state_epoch_field_present": False,
        "fresh_official_load_checkpoint_return_epoch": 0,
    }
    if any(result.get(key) != value for key, value in expected_result.items()):
        raise ValueError("FC-P008 result contract differs")
    if result.get("input_sha256") != {
        "base_manifest": BASE_MANIFEST,
        "base_train_split": BASE_SPLIT,
        "train8_manifest": TRAIN8,
        "train16_manifest": TRAIN16,
        "normalization": NORM,
        "resolved_config": CONFIG,
        "source_phase_mapping": MAPPING,
        "parent_model": PARENT_MODEL,
        "parent_training_state": PARENT_STATE,
        "execution_approval": approval_sha256,
        "implementation": implementation_sha,
    }:
        raise ValueError("FC-P008 result input lineage differs")
    inventory = result.get("source_inventory", {})
    if (
        inventory.get("status") != "FC_P008_TRAIN_ONLY_SOURCE_INVENTORY_PASS"
        or inventory.get("trajectory_count") != 44
        or inventory.get("endpoint_count") != 19648
        or inventory.get("family_endpoint_counts") != FAMILY_ENDPOINTS
        or inventory.get("validation_accessed") is not False
        or inventory.get("frozen_test_accessed") is not False
    ):
        raise ValueError("FC-P008 source inventory differs")
    if result.get("selected_alpha") not in ALPHAS or not finite(result):
        raise ValueError("FC-P008 result selection/finite contract differs")
    if (
        result.get("parent_native_pointwise_wiring_max_abs", math.inf) > 2e-5
        or result.get("candidate_native_pointwise_wiring_max_abs", math.inf) > 2e-5
        or len(result.get("state_output_sha256_by_trajectory", {})) != 44
    ):
        raise ValueError("FC-P008 native replay contract differs")
    if result.get("changed_parameter_names") != [
        "decoder_net.final_layer.linear.weight",
        "decoder_net.final_layer.linear.bias",
    ]:
        raise ValueError("FC-P008 changed parameter names differ")
    checkpoint = build / "candidate"
    model, state = checkpoint / "FNO.0.0.mdlus", checkpoint / "checkpoint.0.0.pt"
    if (
        not model.is_file()
        or not state.is_file()
        or sha256(model) != result.get("candidate_model_sha256")
        or sha256(state) != result.get("candidate_state_sha256")
    ):
        raise ValueError("FC-P008 official checkpoint pair differs")
    if archive_payload(model) != result.get("candidate_model_archive_member_sha256"):
        raise ValueError("FC-P008 official model archive members differ")
    loaded_training_state = training_state(state)
    if "epoch" in loaded_training_state or result.get("official_training_state_epoch_field_present") is not False:
        raise ValueError("FC-P008 official epoch0 state unexpectedly has a top-level epoch")
    if loaded_training_state.get("metadata") != result.get("candidate_training_state_metadata"):
        raise ValueError("FC-P008 official training-state metadata differs")
    if result.get("fresh_official_load_checkpoint_return_epoch") != 0:
        raise ValueError("FC-P008 fresh official load result differs")
    parent_model = repo / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/best/FNO.0.2.mdlus"
    changed = verify_force_row_confinement(parent_model, model)
    if changed != result["changed_parameter_names"]:
        raise ValueError("independent force-row confinement differs")
    parent_tensors = model_state(parent_model)
    candidate_tensors = model_state(model)
    if model_tensor_sha256(parent_tensors) != result.get("parent_tensor_sha256"):
        raise ValueError("FC-P008 parent tensor identity differs")
    candidate_tensor_sha = model_tensor_sha256(candidate_tensors)
    if (
        candidate_tensor_sha != result.get("candidate_tensor_sha256")
        or candidate_tensor_sha != result.get("fresh_official_reload_tensor_sha256")
    ):
        raise ValueError("FC-P008 fresh official reload tensor identity differs")
    required_completion = {
        "candidate_build/result.json",
        "candidate_build/source_inventory.json",
        "candidate_build/train_features.npz",
        "candidate_build/candidate/FNO.0.0.mdlus",
        "candidate_build/candidate/checkpoint.0.0.pt",
        "launch_receipt.json",
        "source_snapshot.sha256",
        "run.log",
    }
    if completion.get("status") != COMPLETION_STATUS:
        raise ValueError("FC-P008 completion status differs")
    hashes = completion.get("sha256")
    if not isinstance(hashes, dict) or not required_completion.issubset(hashes):
        raise ValueError("FC-P008 completion artifact set differs")
    for name, digest in hashes.items():
        path = candidate / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"FC-P008 completion artifact differs: {name}")
    if hashes["candidate_build/result.json"] != sha256(build / "result.json"):
        raise ValueError("FC-P008 result is not bound by completion")
    if sha256(build / "source_inventory.json") != hashes["candidate_build/source_inventory.json"]:
        raise ValueError("FC-P008 source inventory file differs")
    if load(build / "source_inventory.json") != inventory:
        raise ValueError("FC-P008 embedded/source inventory differs")
    return {
        "status": LINEAGE_STATUS,
        "candidate_kind": CANDIDATE_KIND,
        "candidate_root": str(candidate.relative_to(repo)),
        "checkpoint_relative_directory": "candidate_build/candidate",
        "checkpoint_model_file": model.name,
        "checkpoint_state_file": state.name,
        "checkpoint_epoch": 0,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "checkpoint_sha256": sha256(model),
        "checkpoint_state_sha256": sha256(state),
        "checkpoint_generation_payload_sha256": archive_payload(model),
        "parent_model_sha256": PARENT_MODEL,
        "parent_state_sha256": PARENT_STATE,
        "source_snapshot_manifest_sha256": sha256(manifest_path),
        "source_commit": launch.get("source_commit"),
        "implementation_sha256": implementation_sha,
        "result_sha256": sha256(build / "result.json"),
        "launch_receipt_sha256": sha256(candidate / "launch_receipt.json"),
        "completion_receipt_sha256": sha256(candidate / "completion_receipt.json"),
        "execution_approval_sha256": approval_sha256,
        "official_image_id": IMAGE_ID,
        "precision_protocol": PRECISION,
        "data_lineage": static,
        "formal_protocol": FORMAL_PROTOCOL,
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "training_performed": False,
        "validation_or_frozen_accessed": False,
        "ppo_auto_launch": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--execution-approval-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candidate:
        if not args.execution_approval_sha256:
            parser.error("--execution-approval-sha256 is required with --candidate")
        result = validate_candidate(
            args.repo, args.candidate, args.execution_approval_sha256
        )
    else:
        result = {
            "status": "FC_P008_POSTEVAL_STATIC_PREFLIGHT_PASS",
            "data_lineage": validate_static(args.repo.resolve()),
            "candidate_required": True,
        }
    if args.output:
        write_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
