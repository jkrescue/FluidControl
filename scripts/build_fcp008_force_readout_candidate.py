#!/usr/bin/env python3
"""Build the FC-P008 force-row-only FNO candidate from train-only H1 features.

PhysicsNeMo supplies the pinned FNO and checkpoint APIs. Feature extraction,
weighted phase-blocked ridge selection, and force-row calibration are project
methods. The parent is never overwritten and no new CFD is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np

BASE_SPLIT_SHA = "1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89"
BASE_MANIFEST_SHA = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
TRAIN8_MANIFEST_SHA = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
TRAIN16_MANIFEST_SHA = "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
NORM_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
MODEL_SHA = "f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4"
STATE_SHA = "a63186685c998115507c423c5db8d467e845469dc0a5b087b47f341b361a8b4a"
MAPPING_SHA = "57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92"
APPROVAL_CONTRACT_COMMIT = "729e5201833c491de5c8648f42abc9d565f7a0eb"
APPROVAL_CONTRACT_SHA256 = "1e20a2bd1ea008571e3c8768d2e8ee75e8d2ff6860069bdf6f39d66fda3a884c"
ALPHAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
PHASES = ("b00", "b02", "b04", "b06")
CHANNELS = ("front_cd", "front_cl", "rear_cd", "rear_cl")
FAMILY_EXPOSURES = {"base20": 720, "train8": 408, "train16": 240}
FAMILY_ENDPOINTS = {"base20": 16000, "train8": 1600, "train16": 2048}
EXPECTED_FRAMES = {"base20": 801, "train8": 201, "train16": 129}
SOURCE_STATE_ATOL = 3e-7
TIME_ATOL = 2e-5
WIRING_ATOL = 2e-5


@dataclass(frozen=True)
class Trajectory:
    family: str
    name: str
    path: Path
    sha256: str
    frames: int
    source_phase: str = ""
    start_time: float = math.nan
    source_state_max_abs: float = math.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_member_sha256(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("candidate archive has duplicate members")
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(names)
        }


def load_json(path: Path, digest: str) -> dict:
    if sha256(path) != digest:
        raise ValueError(f"JSON SHA differs: {path}")
    return json.loads(path.read_text())


def discover_trajectories(base: Path, train8: Path, train16: Path) -> list[Trajectory]:
    load_json(base / "manifest.json", BASE_MANIFEST_SHA)
    base_split = load_json(base / "splits/train.json", BASE_SPLIT_SHA)
    dynamic = load_json(train8 / "manifest.json", TRAIN8_MANIFEST_SHA)
    direct = load_json(train16 / "manifest.json", TRAIN16_MANIFEST_SHA)
    if base_split.get("split") != "train" or len(base_split.get("cases", [])) != 20:
        raise ValueError("base20 train split differs")
    if dynamic.get("trajectory_counts") != {"train": 8, "validation": 0, "frozen_test": 0}:
        raise ValueError("train8 split differs")
    if direct.get("trajectory_counts") != {"train": 16, "validation": 0, "frozen_test": 0}:
        raise ValueError("train16 split differs")
    if dynamic.get("normalization_sha256") != NORM_SHA or direct.get("normalization_sha256") != NORM_SHA:
        raise ValueError("additional-source normalization differs")
    records = [
        Trajectory("base20", case, base / "train" / f"{case}.h5", base_split["hdf5_sha256"][case], 801)
        for case in base_split["cases"]
    ]
    records.extend(
        Trajectory("train8", Path(name).stem, train8 / "train" / name, digest, 201)
        for name, digest in sorted(dynamic["hdf_sha256"].items())
    )
    records.extend(
        Trajectory("train16", Path(name).stem, train16 / "train" / name, digest, 129)
        for name, digest in sorted(direct["hdf_sha256"].items())
    )
    if len(records) != 44 or {family: sum(r.family == family for r in records) for family in FAMILY_EXPOSURES} != {
        "base20": 20,
        "train8": 8,
        "train16": 16,
    }:
        raise ValueError("44-trajectory inventory differs")
    for record in records:
        if not record.path.is_file() or sha256(record.path) != record.sha256:
            raise ValueError(f"HDF SHA differs: {record.path}")
    return records


def _initial_state(path: Path, expected_frames: int) -> tuple[float, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as hdf:
        if {len(hdf[key]) for key in ("state", "mask", "omega", "force", "time")} != {expected_frames}:
            raise ValueError(f"trajectory length differs: {path}")
        time = float(np.asarray(hdf["time"][0]).reshape(-1)[0])
        state = np.asarray(hdf["state"][0], dtype=np.float32)
    if state.shape[0] != 3 or not np.isfinite(state).all() or not math.isfinite(time):
        raise ValueError("initial state/time differs")
    return time, state


def assign_source_phases(records: list[Trajectory]) -> list[Trajectory]:
    references = {}
    for phase in PHASES:
        candidates = [r for r in records if r.family == "base20" and r.name.endswith(f"_{phase}_zero")]
        if len(candidates) != 1:
            raise ValueError(f"canonical zero missing for {phase}")
        references[phase] = _initial_state(candidates[0].path, candidates[0].frames)
    if {phase: references[phase][0] for phase in PHASES} != {
        "b00": 148.0,
        "b02": 106.0,
        "b04": 120.0,
        "b06": 134.0,
    }:
        raise ValueError("canonical source times differ")
    assigned = []
    for record in records:
        time, state = _initial_state(record.path, record.frames)
        matches = []
        distances = {}
        for phase, (reference_time, reference_state) in references.items():
            distance = float(np.max(np.abs(state - reference_state)))
            distances[phase] = distance
            if abs(time - reference_time) <= TIME_ATOL and distance <= SOURCE_STATE_ATOL:
                matches.append(phase)
        if len(matches) != 1:
            raise ValueError(f"source phase is not unique for {record.name}: {time}, {distances}")
        assigned.append(
            Trajectory(
                **{**record.__dict__, "source_phase": matches[0], "start_time": time, "source_state_max_abs": distances[matches[0]]}
            )
        )
    return assigned


def inventory_payload(records: list[Trajectory]) -> dict:
    counts = {family: sum(r.family == family for r in records) for family in FAMILY_EXPOSURES}
    endpoints = {family: sum(r.frames - 1 for r in records if r.family == family) for family in FAMILY_EXPOSURES}
    if endpoints != FAMILY_ENDPOINTS:
        raise ValueError("19648 endpoint inventory differs")
    return {
        "status": "FC_P008_TRAIN_ONLY_SOURCE_INVENTORY_PASS",
        "trajectory_count": len(records),
        "endpoint_count": sum(endpoints.values()),
        "family_trajectory_counts": counts,
        "family_endpoint_counts": endpoints,
        "canonical_phase_mapping_method": "unique actual time0 within 2e-5 and state0 max-abs within 3e-7 against canonical base zero",
        "records": [
            {
                "family": r.family,
                "name": r.name,
                "hdf_sha256": r.sha256,
                "frames": r.frames,
                "endpoints": r.frames - 1,
                "source_phase": r.source_phase,
                "start_time": r.start_time,
                "source_state_max_abs": r.source_state_max_abs,
            }
            for r in records
        ],
        "validation_accessed": False,
        "frozen_test_accessed": False,
    }


def validate_mapping_artifact(records: list[Trajectory], path: Path) -> dict:
    mapping = load_json(path, MAPPING_SHA)
    if mapping.get("status") != "FCP003C_FULL_TRAIN_SOURCE_PHASE_MAPPING_COMPLETE" or mapping.get("trajectory_count") != 44:
        raise ValueError("source-phase mapping contract differs")
    entries = {(item["family"], item["file"]): item for item in mapping.get("trajectories", [])}
    if len(entries) != 44:
        raise ValueError("source-phase mapping entries differ")
    for record in records:
        item = entries.get((record.family, record.path.name))
        if (
            item is None
            or item.get("canonical_physical_phase") != record.source_phase
            or abs(float(item.get("physical_start_time")) - record.start_time) > TIME_ATOL
            or abs(float(item.get("hdf_state0_max_abs_vs_base_zero")) - record.source_state_max_abs) > 1e-12
        ):
            raise ValueError(f"independent mapping differs for {record.name}")
    return mapping


def endpoint_weights(families: np.ndarray) -> np.ndarray:
    families = np.asarray(families)
    total_exposure = float(sum(FAMILY_EXPOSURES.values()))
    weights = np.empty(len(families), dtype=np.float64)
    for family in FAMILY_EXPOSURES:
        mask = families == family
        if int(mask.sum()) != FAMILY_ENDPOINTS[family]:
            raise ValueError(f"endpoint count differs for {family}")
        weights[mask] = FAMILY_EXPOSURES[family] / total_exposure / FAMILY_ENDPOINTS[family]
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0, abs_tol=2e-15):
        raise ValueError("global endpoint weights do not sum to one")
    return weights


def fit_weighted_ridge(x, y, weights, alpha):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 128 or y.shape != (len(x), 4) or weights.shape != (len(x),):
        raise ValueError("weighted ridge shapes differ")
    if alpha not in ALPHAS or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all():
        raise ValueError("weighted ridge values differ")
    weights = weights / weights.sum()
    mean = weights @ x
    variance = weights @ np.square(x - mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    active = std > 1e-12
    if not active.any():
        raise ValueError("all features constant")
    z = (x[:, active] - mean[active]) / std[active]
    ymean = weights @ y
    yc = y - ymean
    root_weight = np.sqrt(weights)[:, None]
    zw, yw = z * root_weight, yc * root_weight
    if alpha == 0.0:
        scaled, _, rank, singular = np.linalg.lstsq(zw, yw, rcond=1e-10)
    else:
        gram, cross = zw.T @ zw, zw.T @ yw
        scaled = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
        singular = np.linalg.svd(zw, compute_uv=False)
        rank = int(np.linalg.matrix_rank(zw))
    coefficients = np.zeros((129, 4), dtype=np.float64)
    coefficients[:128][active] = scaled / std[active, None]
    coefficients[128] = ymean - mean @ coefficients[:128]
    return coefficients, {
        "alpha": alpha,
        "objective": "weighted mean over endpoints of the mean normalized squared error over four force channels plus alpha/4 times the squared Frobenius norm of the standardized-coordinate four-channel coefficient matrix",
        "normalized_input_weight_sum": float(weights.sum()),
        "constant_feature_indices": np.flatnonzero(~active).tolist(),
        "rank": int(rank),
        "singular_max": float(singular[0]),
        "singular_min": float(singular[-1]),
        "original_coordinate_channel_l2": [float(np.linalg.norm(coefficients[:, i])) for i in range(4)],
    }


def predict(x, coefficients):
    return np.asarray(x, dtype=np.float64) @ coefficients[:128] + coefficients[128]


def choose_alpha(scores: dict[float, float]) -> float:
    if set(scores) != set(ALPHAS) or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("fixed complete alpha scores required")
    return min(ALPHAS, key=lambda alpha: (scores[alpha], -alpha))


def grouped_metrics(error, phases, families, weights):
    error = np.asarray(error, dtype=np.float64)
    payload = {}
    for phase in PHASES:
        payload[phase] = {}
        for family in FAMILY_EXPOSURES:
            mask = (phases == phase) & (families == family)
            if not mask.any():
                payload[phase][family] = None
                continue
            local_weights = weights[mask] / weights[mask].sum()
            payload[phase][family] = {
                channel: {
                    "count": int(mask.sum()),
                    "mae": float(local_weights @ np.abs(error[mask, index])),
                    "rmse": float(np.sqrt(local_weights @ np.square(error[mask, index]))),
                    "bias": float(local_weights @ error[mask, index]),
                    "mse": float(local_weights @ np.square(error[mask, index])),
                }
                for index, channel in enumerate(CHANNELS)
            }
    return payload


def aggregate_metrics(error, weights):
    error = np.asarray(error, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return {
        channel: {
            "count": int(len(error)),
            "mae": float(weights @ np.abs(error[:, index])),
            "rmse": float(np.sqrt(weights @ np.square(error[:, index]))),
            "bias": float(weights @ error[:, index]),
            "mse": float(weights @ np.square(error[:, index])),
        }
        for index, channel in enumerate(CHANNELS)
    }


def phase_blocked_cv(features, targets, phases, families, weights):
    oof = {alpha: np.empty_like(targets, dtype=np.float64) for alpha in ALPHAS}
    fold_reports = {phase: {} for phase in PHASES}
    for phase in PHASES:
        train = phases != phase
        held = ~train
        for alpha in ALPHAS:
            coefficients, fit = fit_weighted_ridge(features[train], targets[train], weights[train], alpha)
            oof[alpha][held] = predict(features[held], coefficients)
            fold_reports[phase][str(alpha)] = fit
    scores, per_channel = {}, {}
    for alpha in ALPHAS:
        squared = np.square(oof[alpha] - targets)
        channel_mse = weights @ squared
        scores[alpha] = float(channel_mse.mean())
        per_channel[str(alpha)] = dict(zip(CHANNELS, [float(v) for v in channel_mse], strict=True))
    selected = choose_alpha(scores)
    return selected, {
        "aggregate_global_endpoint_weighted_equal_channel_mse": {str(k): v for k, v in scores.items()},
        "per_channel_mse": per_channel,
        "selected_alpha": selected,
        "fold_fits": fold_reports,
        "selected_oof_by_phase_family": grouped_metrics(oof[selected] - targets, phases, families, weights),
    }


def verify_force_row_only(before: dict, after: dict, weight_key: str, bias_key: str, expected):
    if set(before) != set(after):
        raise ValueError("state_dict keys changed")
    changed = []
    for key in before:
        if key == weight_key:
            if not np.array_equal(before[key][:3], after[key][:3]) or not np.array_equal(after[key][3:7], expected[:128].T):
                raise ValueError("force weight row update differs")
        elif key == bias_key:
            if not np.array_equal(before[key][:3], after[key][:3]) or not np.array_equal(after[key][3:7], expected[128]):
                raise ValueError("force bias row update differs")
        elif not np.array_equal(before[key], after[key]):
            raise ValueError(f"non-force tensor changed: {key}")
        if not np.array_equal(before[key], after[key]):
            changed.append(key)
    if changed != [key for key in before if key in (weight_key, bias_key)]:
        raise ValueError("changed-tensor inventory differs")
    return changed


def tensor_sha256(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_tensor_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def default_precision_contract(torch_module) -> dict:
    settings = {
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
    }
    if (
        settings["NVIDIA_TF32_OVERRIDE"] is not None
        or settings["cuda_matmul_allow_tf32"] is not True
        or settings["cudnn_allow_tf32"] is not True
        or settings["float32_matmul_precision"] != "high"
    ):
        raise ValueError(f"historical default TF32/high protocol differs: {settings}")
    return settings


def extract_trajectory(model, layer, record, stats, device, chunk_size, *, collect_features):
    import h5py
    import torch

    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    force_mean = np.asarray(stats["all_force_mean"], dtype=np.float64).reshape(1, 4)
    force_std = np.asarray(stats["all_force_std"], dtype=np.float64).reshape(1, 4)
    features, targets, native = [], [], []
    state_digest = hashlib.sha256()
    wiring_max = 0.0
    ideal_native_original_max = 0.0
    with h5py.File(record.path, "r") as hdf:
        mask_np = np.asarray(hdf["mask"][0], dtype=np.float32)
        if not np.array_equal(np.asarray(hdf["mask"][:]), np.broadcast_to(mask_np, hdf["mask"].shape)):
            raise ValueError(f"mask changes: {record.name}")
        mask = torch.as_tensor(mask_np, device=device).reshape(1, 1, *mask_np.shape[-2:])
        for first in range(1, record.frames, chunk_size):
            last = min(first + chunk_size, record.frames)
            state_np = np.asarray(hdf["state"][first - 1 : last - 1], dtype=np.float32)
            omega_now = np.asarray(hdf["omega"][first - 1 : last], dtype=np.float32).reshape(-1) / 0.75
            force_np = np.asarray(hdf["force"][first:last], dtype=np.float32)
            if len(omega_now) != (last - first + 1):
                raise ValueError("action endpoint slice differs")
            state = (torch.as_tensor(state_np, device=device) - state_mean) / state_std
            batch, _, height, width = state.shape
            mask_batch = mask.expand(batch, -1, -1, -1)
            state = state * mask_batch
            now = torch.as_tensor(omega_now[:-1], device=device).reshape(batch, 1, 1, 1).expand(-1, 1, height, width)
            following = torch.as_tensor(omega_now[1:], device=device).reshape(batch, 1, 1, 1).expand(-1, 1, height, width)
            inputs = torch.cat((state, mask_batch, now, following), dim=1)
            captured_input, captured_output = [], []
            pre = layer.register_forward_pre_hook(lambda _module, args: captured_input.append(args[0]))
            post = layer.register_forward_hook(lambda _module, _args, output: captured_output.append(output))
            try:
                with torch.no_grad():
                    raw = model(inputs)
            finally:
                pre.remove()
                post.remove()
            pixels = height * width
            hidden = captured_input[0]
            point_output = captured_output[0]
            if hidden.shape != (batch * pixels, 128) or point_output.shape != (batch * pixels, 7):
                raise ValueError("official final-layer topology differs")
            grid_output = point_output.reshape(batch, height, width, 7).permute(0, 3, 1, 2)
            wiring_max = max(wiring_max, float((grid_output - raw).abs().max().item()))
            raw_force = (raw[:, 3:] * mask_batch).sum((-2, -1)) / mask_batch.sum((-2, -1))
            state_array = raw[:, :3].detach().cpu().contiguous().numpy()
            state_digest.update(state_array.tobytes())
            native.append(raw_force.detach().cpu().numpy().astype(np.float32))
            targets.append(((force_np.astype(np.float64) - force_mean) / force_std).astype(np.float32))
            if collect_features:
                hidden = hidden.reshape(batch, pixels, 128)
                flat_mask = mask_batch.permute(0, 2, 3, 1).reshape(batch, pixels, 1)
                pooled = (hidden * flat_mask).sum(1) / flat_mask.sum(1)
                ideal_original = pooled @ layer.linear.weight[3:7].T + layer.linear.bias[3:7]
                ideal_native_original_max = max(
                    ideal_native_original_max, float((ideal_original - raw_force).abs().max().item())
                )
                features.append(pooled.detach().cpu().numpy().astype(np.float32))
    if wiring_max > WIRING_ATOL:
        raise ValueError(f"native pointwise-head wiring differs: {wiring_max}")
    arrays = [*targets, *native]
    if collect_features:
        arrays.extend(features)
    if not arrays or any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"nonfinite extracted values: {record.name}")
    return {
        "features": np.concatenate(features) if collect_features else None,
        "targets": np.concatenate(targets),
        "native": np.concatenate(native),
        "state_sha256": state_digest.hexdigest(),
        "wiring_max_abs": wiring_max,
        "ideal_native_original_max_abs": ideal_native_original_max if collect_features else None,
    }


def extract_all(model, records, stats, device, chunk_size, *, collect_features):
    layer = model.decoder_net.final_layer
    features, targets, native, phases, families, names = [], [], [], [], [], []
    state_sha, wiring_max, ideal_max = {}, 0.0, 0.0
    for index, record in enumerate(records):
        result = extract_trajectory(model, layer, record, stats, device, chunk_size, collect_features=collect_features)
        if collect_features:
            features.append(result["features"])
        targets.append(result["targets"])
        native.append(result["native"])
        count = record.frames - 1
        phases.extend([record.source_phase] * count)
        families.extend([record.family] * count)
        names.extend([record.name] * count)
        state_sha[record.name] = result["state_sha256"]
        wiring_max = max(wiring_max, result["wiring_max_abs"])
        ideal_max = max(ideal_max, result["ideal_native_original_max_abs"] or 0.0)
        finite = all(
            np.isfinite(value).all()
            for value in (result["targets"], result["native"], result["features"])
            if value is not None
        )
        if not finite:
            raise ValueError(f"nonfinite replay: {record.name}")
        print(json.dumps({"event": "fcp008_replay_progress", "pass": "features" if collect_features else "candidate_native", "trajectory_index": index, "trajectory": record.name, "endpoints": count, "finite": finite}), flush=True)
    return {
        "features": np.concatenate(features) if collect_features else None,
        "targets": np.concatenate(targets),
        "native": np.concatenate(native),
        "phases": np.asarray(phases),
        "families": np.asarray(families),
        "names": np.asarray(names),
        "state_sha256_by_trajectory": state_sha,
        "wiring_max_abs": wiring_max,
        "ideal_native_original_max_abs": ideal_max,
    }


def atomic_json(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "train8", "train16", "normalization", "config", "checkpoint-dir", "source-phase-mapping", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() or isinstance(args.chunk_size, bool) or not 1 <= args.chunk_size <= 32:
        raise ValueError("exclusive output and chunk-size [1,32] required")
    records = assign_source_phases(discover_trajectories(args.base, args.train8, args.train16))
    inventory = inventory_payload(records)
    validate_mapping_artifact(records, args.source_phase_mapping)
    if args.inventory_only:
        inventory["implementation_sha256"] = sha256(Path(__file__))
        inventory["approval_contract_commit"] = APPROVAL_CONTRACT_COMMIT
        inventory["approval_contract_sha256"] = APPROVAL_CONTRACT_SHA256
        inventory["source_phase_mapping_sha256"] = MAPPING_SHA
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, inventory)
        print(inventory["status"], inventory["trajectory_count"], inventory["endpoint_count"])
        return
    if args.approval is None or not args.approval.is_file():
        raise ValueError("reviewed FC-P008 execution approval is required")
    approval = json.loads(args.approval.read_text())
    if (
        approval.get("status") != "FC_P008_FULL_EXECUTION_APPROVED"
        or approval.get("implementation_sha256") != sha256(Path(__file__))
        or approval.get("approval_contract_sha256") != APPROVAL_CONTRACT_SHA256
    ):
        raise ValueError("FC-P008 approval does not bind this implementation")
    if sha256(args.normalization) != NORM_SHA or sha256(args.config) != CONFIG_SHA:
        raise ValueError("normalization/config differs")
    models = list(args.checkpoint_dir.glob("FNO.0.*.mdlus"))
    states = list(args.checkpoint_dir.glob("checkpoint.0.*.pt"))
    if len(models) != 1 or sha256(models[0]) != MODEL_SHA or len(states) != 1 or sha256(states[0]) != STATE_SHA:
        raise ValueError("fixed C parent differs")
    import torch
    from evaluate_tandem_fno import load_composed_config
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from train_tandem_fno import build_model, configured_force_indices

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda:
        raise RuntimeError("single-GPU CUDA execution required")
    torch.cuda.set_per_process_memory_fraction(0.15, device=dist.device)
    precision = default_precision_contract(torch)
    config = load_composed_config(args.config)
    if tuple(configured_force_indices(config)) != (0, 1, 2, 3):
        raise ValueError("four-force output contract differs")
    model = build_model(config).to(dist.device)
    if load_checkpoint(args.checkpoint_dir, models=model, device=dist.device) != 2:
        raise ValueError("expected C epoch2")
    model.eval()
    parent_tensor_sha = model_tensor_sha256(model)
    stats = json.loads(args.normalization.read_text())
    if tuple(stats.get("all_force_channels", ())) != CHANNELS:
        raise ValueError("force-channel order differs")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "source_inventory.json", inventory)
    parent = extract_all(model, records, stats, dist.device, args.chunk_size, collect_features=True)
    if len(parent["features"]) != 19648 or not np.isfinite(parent["features"]).all():
        raise ValueError("feature extraction count/finite differs")
    weights = endpoint_weights(parent["families"])
    selected_alpha, cv = phase_blocked_cv(parent["features"], parent["targets"], parent["phases"], parent["families"], weights)
    coefficients, full_fit = fit_weighted_ridge(parent["features"], parent["targets"], weights, selected_alpha)
    ideal = predict(parent["features"], coefficients)
    np.savez_compressed(
        args.output / "train_features.npz",
        features=parent["features"].astype(np.float32),
        targets_normalized=parent["targets"].astype(np.float32),
        parent_native_normalized=parent["native"].astype(np.float32),
        phases=parent["phases"], families=parent["families"], trajectory_names=parent["names"],
        global_endpoint_weights=weights, selected_coefficients=coefficients,
    )
    layer = model.decoder_net.final_layer.linear
    if tuple(layer.weight.shape) != (7, 128) or tuple(layer.bias.shape) != (7,):
        raise ValueError("final affine shape differs")
    weight_key = next(name for name, value in model.named_parameters() if value is layer.weight)
    bias_key = next(name for name, value in model.named_parameters() if value is layer.bias)
    before_arrays = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    before_tensor_shas = {name: tensor_sha256(value) for name, value in model.state_dict().items()}
    with torch.no_grad():
        layer.weight[3:7].copy_(torch.as_tensor(coefficients[:128].T, dtype=layer.weight.dtype, device=layer.weight.device))
        layer.bias[3:7].copy_(torch.as_tensor(coefficients[128], dtype=layer.bias.dtype, device=layer.bias.device))
    after_arrays = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    changed = verify_force_row_only(before_arrays, after_arrays, weight_key, bias_key, coefficients.astype(np.float32))
    after_tensor_shas = {name: tensor_sha256(value) for name, value in model.state_dict().items()}
    if {name for name in before_tensor_shas if before_tensor_shas[name] != after_tensor_shas[name]} != {weight_key, bias_key}:
        raise ValueError("tensor SHA change set differs")
    assigned_tensor_sha = model_tensor_sha256(model)
    checkpoint_dir = args.output / "candidate"
    save_checkpoint(
        checkpoint_dir,
        models=model,
        epoch=0,
        metadata={
            "status": "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE",
            "candidate_checkpoint_epoch": 0,
            "parent_checkpoint_epoch": 2,
            "calibration_generation": 1,
            "parent_model_sha256": MODEL_SHA,
            "selected_alpha": selected_alpha,
            "changed_parameters": changed,
            "validation_accessed": False,
            "frozen_test_accessed": False,
            "ppo_executed": False,
        },
    )
    candidate_models = list(checkpoint_dir.glob("FNO.0.*.mdlus"))
    candidate_states = list(checkpoint_dir.glob("checkpoint.0.*.pt"))
    if len(candidate_models) != 1 or len(candidate_states) != 1:
        raise ValueError("official candidate checkpoint pair missing")
    training_state = torch.load(candidate_states[0], map_location="cpu", weights_only=False)
    if (
        "epoch" in training_state
        or training_state.get("metadata", {}).get("status")
        != "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE"
        or training_state.get("metadata", {}).get("candidate_checkpoint_epoch") != 0
    ):
        raise ValueError("official candidate training-state metadata differs")
    reloaded = build_model(config).to(dist.device)
    fresh_load_return_epoch = load_checkpoint(
        checkpoint_dir, models=reloaded, device=dist.device
    )
    if fresh_load_return_epoch != 0:
        raise ValueError("expected candidate calibration epoch0")
    reloaded.eval()
    reloaded_arrays = {name: value.detach().cpu().numpy().copy() for name, value in reloaded.state_dict().items()}
    verify_force_row_only(before_arrays, reloaded_arrays, weight_key, bias_key, coefficients.astype(np.float32))
    if model_tensor_sha256(reloaded) != assigned_tensor_sha:
        raise ValueError("official reloaded candidate tensor SHA differs")
    del model
    candidate = extract_all(reloaded, records, stats, dist.device, args.chunk_size, collect_features=False)
    if parent["state_sha256_by_trajectory"] != candidate["state_sha256_by_trajectory"]:
        raise ValueError("state outputs changed after force-row assignment")
    if not np.array_equal(parent["targets"], candidate["targets"]):
        raise ValueError("native replay targets changed")
    native_error = candidate["native"].astype(np.float64) - candidate["targets"].astype(np.float64)
    parent_native_error = parent["native"].astype(np.float64) - parent["targets"].astype(np.float64)
    ideal_native = candidate["native"].astype(np.float64) - ideal
    if any(not np.isfinite(array).all() for array in (native_error, parent_native_error, ideal_native)):
        raise ValueError("candidate native replay is nonfinite")
    cache_path = args.output / "train_features.npz"
    cache = dict(np.load(cache_path, allow_pickle=False))
    cache["candidate_native_normalized"] = candidate["native"].astype(np.float32)
    np.savez_compressed(cache_path, **cache)
    force_std = np.asarray(stats["all_force_std"], dtype=np.float64).reshape(1, 4)
    input_sha256 = {
        "base_manifest": sha256(args.base / "manifest.json"),
        "base_train_split": sha256(args.base / "splits/train.json"),
        "train8_manifest": sha256(args.train8 / "manifest.json"),
        "train16_manifest": sha256(args.train16 / "manifest.json"),
        "normalization": sha256(args.normalization),
        "resolved_config": sha256(args.config),
        "source_phase_mapping": sha256(args.source_phase_mapping),
        "parent_model": sha256(models[0]),
        "parent_training_state": sha256(states[0]),
        "execution_approval": sha256(args.approval),
        "implementation": sha256(Path(__file__)),
    }
    result = {
        "status": "FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE_COMPLETE_NOT_ADMISSION",
        "scope": "project force-row calibration using official pinned FNO/checkpoint APIs; not validation, PPO, or control evidence",
        "source_inventory": inventory,
        "source_phase_mapping_sha256": MAPPING_SHA,
        "approval_contract_commit": APPROVAL_CONTRACT_COMMIT,
        "approval_contract_sha256": APPROVAL_CONTRACT_SHA256,
        "execution_approval_sha256": sha256(args.approval),
        "family_exposures": FAMILY_EXPOSURES,
        "family_endpoint_counts": FAMILY_ENDPOINTS,
        "endpoint_weight_formula": "(family_exposure/1368)/family_endpoint_count; fold fit renormalizes retained rows; OOF pools original global weights once",
        "precision_protocol": precision,
        "input_sha256": input_sha256,
        "parent_model_sha256": MODEL_SHA,
        "parent_training_state_sha256": STATE_SHA,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "candidate_checkpoint_epoch": 0,
        "official_training_state_epoch_field_present": False,
        "fresh_official_load_checkpoint_return_epoch": fresh_load_return_epoch,
        "fresh_official_reload_tensor_sha256": model_tensor_sha256(reloaded),
        "calibration_fit_performed": True,
        "no_optimizer_training": True,
        "parent_tensor_sha256": parent_tensor_sha,
        "candidate_tensor_sha256": model_tensor_sha256(reloaded),
        "changed_parameter_names": changed,
        "state_rows_0_3_byte_identical": True,
        "all_other_tensors_byte_identical": True,
        "cv": cv,
        "selected_alpha": selected_alpha,
        "full_train_fit": full_fit,
        "full_train_parent_native_metrics_normalized": aggregate_metrics(parent_native_error, weights),
        "full_train_parent_native_by_phase_family": grouped_metrics(parent_native_error, parent["phases"], parent["families"], weights),
        "full_train_parent_native_metrics_physical": aggregate_metrics(parent_native_error * force_std, weights),
        "full_train_ideal_metrics_normalized": aggregate_metrics(ideal - parent["targets"], weights),
        "full_train_native_metrics_normalized": aggregate_metrics(native_error, weights),
        "full_train_native_metrics_physical": aggregate_metrics(native_error * force_std, weights),
        "full_train_native_by_phase_family": grouped_metrics(native_error, parent["phases"], parent["families"], weights),
        "ideal_vs_native_normalized": {
            "aggregate": aggregate_metrics(ideal_native, weights),
            "max_abs": float(np.max(np.abs(ideal_native))),
        },
        "parent_native_pointwise_wiring_max_abs": parent["wiring_max_abs"],
        "candidate_native_pointwise_wiring_max_abs": candidate["wiring_max_abs"],
        "parent_pooled_affine_vs_native_max_abs_diagnostic_only": parent["ideal_native_original_max_abs"],
        "state_output_sha256_by_trajectory": candidate["state_sha256_by_trajectory"],
        "feature_cache_sha256": sha256(args.output / "train_features.npz"),
        "candidate_model_sha256": sha256(candidate_models[0]),
        "candidate_state_sha256": sha256(candidate_states[0]),
        "candidate_model_archive_member_sha256": zip_member_sha256(candidate_models[0]),
        "candidate_training_state_metadata": training_state["metadata"],
        "implementation_sha256": sha256(Path(__file__)),
        "optimizer_steps": 0,
        "architecture_changed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "formal_evaluation_authorized": False,
    }
    atomic_json(args.output / "result.json", result)
    print(result["status"], selected_alpha, result["candidate_model_sha256"])


if __name__ == "__main__":
    main()
