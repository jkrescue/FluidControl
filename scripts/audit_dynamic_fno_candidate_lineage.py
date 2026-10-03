#!/usr/bin/env python3
"""Build fail-closed lineage for reviewed dynamic or train16 FNO candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

DEV30_MANIFEST_SHA = "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
TRAIN8_MANIFEST_SHA = "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
TRAIN16_MANIFEST_SHA = "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
NORMALIZATION_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
FULL40_MANIFEST_SHA = "1c9086b4d08da57e84fb4f6a22376f0935596727952a50c55acb1a130e77e90e"
PHYSICSNEMO_IMAGE_ID = "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
TRAIN16_PROBE = Path(
    "artifacts/tandem_cylinders/directppo_train16_official_datapipe_probe_20261004.json"
)
MODEL = re.compile(r"FNO\.0\.(\d+)\.mdlus")
STATE = re.compile(r"checkpoint\.0\.(\d+)\.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha256(repo: Path, commit: str, relative: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _one(path: Path, pattern: re.Pattern[str]) -> tuple[Path, int]:
    rows = []
    for item in path.iterdir():
        match = pattern.fullmatch(item.name)
        if item.is_file() and match:
            rows.append((item, int(match.group(1))))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one matching file under {path}")
    return rows[0]


def _finite_history(path: Path, *, epochs: int, horizon: int) -> tuple[list[dict], int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or [row.get("epoch") for row in value] != list(
        range(1, epochs + 1)
    ):
        raise ValueError("training history epoch sequence differs")
    for row in value:
        numeric = [item for item in row.values() if isinstance(item, (int, float))]
        if any(isinstance(item, bool) or not math.isfinite(float(item)) for item in numeric):
            raise ValueError("training history contains non-finite metrics")
        if (
            row.get("teacher_forcing_ratio") != 0.0
            or row.get("train_rollout_steps") != horizon
            or row.get("validation_rollout_steps") != 100
        ):
            raise ValueError("free-autoregressive horizon contract differs")
    selected = min(value, key=lambda row: float(row["selection_score"]))
    if sum(
        math.isclose(
            float(row["selection_score"]),
            float(selected["selection_score"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for row in value
    ) != 1:
        raise ValueError("best selection score is not unique")
    return value, int(selected["epoch"])


def build(repo: Path, candidate_root: Path) -> dict:
    repo = repo.resolve()
    root = candidate_root.resolve()
    artifacts = (repo / "artifacts").resolve()
    if artifacts not in root.parents:
        raise ValueError("candidate must remain under the project artifact root")
    is_train16 = root.name == "tandem_fno_control_train16_h100_20261004"
    if not is_train16 and not root.name.startswith("tandem_fno_dynamic_train8_h"):
        raise ValueError("candidate is not a reviewed FNO artifact kind")

    config_path = root / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config.get("training", {})
    data = config.get("data", {})
    horizon = int(training.get("rollout_steps", -1))
    epochs = int(training.get("epochs", -1))
    if is_train16:
        candidate_kind = "control_train16_h100"
        expected_roots = ["/workspace/train8", "/workspace/train16"]
        expected_windows = (720, 408, 240)
        source_config_relative = "conf/tandem_fno_control_train16_h100.yaml"
        if (
            horizon != 100
            or epochs != 2
            or training.get("batch_size") != 2
            or not math.isclose(
                float(training.get("learning_rate", math.nan)),
                1.0e-5,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or training.get("train_stride") != 20
            or training.get("additional_train_stride") != 2
            or training.get("initial_checkpoint") != "/workspace/parent"
        ):
            raise ValueError("train16 candidate training contract differs")
    else:
        candidate_kind = f"dynamic_train8_h{horizon}"
        expected_roots = ["/workspace/train8"]
        expected_windows = {50: (760, 608), 100: (720, 408)}.get(horizon, ())
        source_config_relative = f"conf/tandem_fno_dynamic_train8_h{horizon}.yaml"
    if not expected_windows or epochs != ({50: 4, 100: 2}.get(horizon)):
        raise ValueError("candidate horizon/epoch contract differs")
    if (
        training.get("validation_rollout_steps") != 100
        or float(training.get("teacher_forcing_start", math.nan)) != 0.0
        or float(training.get("teacher_forcing_end", math.nan)) != 0.0
        or training.get("seed") != 20261003
        or data.get("root") != "/workspace/base"
        or data.get("additional_train_roots") != expected_roots
        or data.get("force_indices") != [0, 1, 2, 3]
    ):
        raise ValueError("candidate resolved config differs")

    history, selected_epoch = _finite_history(
        root / "training_history.json", epochs=epochs, horizon=horizon
    )
    model, model_epoch = _one(root / "best", MODEL)
    state, state_epoch = _one(root / "best", STATE)
    if model_epoch != selected_epoch or state_epoch != selected_epoch:
        raise ValueError("best files do not match the unique selected epoch")
    checkpoint_model = root / "checkpoints" / model.name
    checkpoint_state = root / "checkpoints" / state.name
    if (
        sha256(model) != sha256(checkpoint_model)
        or sha256(state) != sha256(checkpoint_state)
    ):
        raise ValueError("best files differ from their checkpoint generation")

    base = repo / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
    train8 = repo / "data/curated/tandem_cylinders_dynamic_train8_v1"
    train16 = repo / "data/curated/tandem_cylinders_directppo_train16_v1"
    formal = repo / "data/curated/tandem_cylinders_matched_start_full40_v1"
    if sha256(base / "manifest.json") != DEV30_MANIFEST_SHA:
        raise ValueError("dev30 manifest differs")
    if sha256(train8 / "manifest.json") != TRAIN8_MANIFEST_SHA:
        raise ValueError("train8 manifest differs")
    if sha256(formal / "manifest.json") != FULL40_MANIFEST_SHA:
        raise ValueError("formal full40 manifest differs")
    normalizations = [
        sha256(base / "normalization.json"),
        sha256(train8 / "normalization.json"),
        sha256(formal / "normalization.json"),
    ]
    if is_train16:
        train16_manifest = load_json(train16 / "manifest.json")
        if (
            sha256(train16 / "manifest.json") != TRAIN16_MANIFEST_SHA
            or train16_manifest.get("status")
            != "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED"
            or train16_manifest.get("profile") != "directppo_train16_v1"
            or train16_manifest.get("trajectory_counts")
            != {"train": 16, "validation": 0, "frozen_test": 0}
            or train16_manifest.get("frames_per_trajectory") != 129
            or train16_manifest.get("normalization_sha256") != NORMALIZATION_SHA
            or train16_manifest.get("validation_or_frozen_accessed") is not False
        ):
            raise ValueError("train16 finalized manifest contract differs")
        normalizations.append(sha256(train16 / "normalization.json"))
    if normalizations != [NORMALIZATION_SHA] * (4 if is_train16 else 3):
        raise ValueError("base/additional/formal normalization identity differs")

    sources_path = root / "training_data_sources.json"
    sources = load_json(sources_path)
    additional = sources.get("additional_sources")
    expected_additional = [
        ("/workspace/train8", expected_windows[1], TRAIN8_MANIFEST_SHA)
    ]
    if is_train16:
        expected_additional.append(
            ("/workspace/train16", expected_windows[2], TRAIN16_MANIFEST_SHA)
        )
    if (
        sources.get("base_root") != "/workspace/base"
        or sources.get("base_windows") != expected_windows[0]
        or sources.get("total_windows") != sum(expected_windows)
        or not isinstance(additional, list)
        or len(additional) != len(expected_additional)
    ):
        raise ValueError("training data-source receipt differs")
    for row, (expected_root, windows, manifest_sha) in zip(
        additional, expected_additional, strict=True
    ):
        if (
            row.get("root") != expected_root
            or row.get("windows") != windows
            or row.get("stride") != 2
            or row.get("manifest_sha256") != manifest_sha
            or row.get("normalization_sha256") != NORMALIZATION_SHA
        ):
            raise ValueError("training data-source receipt differs")
    if is_train16 and (
        additional[0].get("release_status")
        != "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED"
        or additional[0].get("trajectory_count") != 8
        or additional[1].get("release_status")
        != "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED"
        or additional[1].get("trajectory_count") != 16
    ):
        raise ValueError("train16 training-source release identity differs")

    parent_path = root / "parent_receipt.json"
    parent = load_json(parent_path)
    parent_hashes = parent.get("sha256")
    if parent.get("status") != "IMMUTABLE_PARENT_COPIED" or not isinstance(
        parent_hashes, dict
    ) or len(parent_hashes) != 2:
        raise ValueError("immutable parent receipt differs")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in parent_hashes.values()
    ):
        raise ValueError("immutable parent SHA is invalid")
    local_parent = root / "immutable_parent"
    if local_parent.is_dir():
        for name, digest in parent_hashes.items():
            if not (local_parent / name).is_file() or sha256(local_parent / name) != digest:
                raise ValueError("local immutable parent differs from receipt")

    source_path = root / "source_receipt.json"
    source = load_json(source_path) if source_path.is_file() else None
    if is_train16 and source is None:
        raise ValueError("train16 candidate requires a launch-time source receipt")
    source_commit = None
    source_tree = None
    training_image_id = None
    if source is not None:
        source_commit = source.get("source_snapshot_commit")
        source_tree = source.get("source_snapshot_tree")
        training_image_id = source.get("physicsnemo_image_id")
        if (
            not re.fullmatch(r"[0-9a-f]{40}", str(source_commit))
            or not re.fullmatch(r"[0-9a-f]{40}", str(source_tree))
            or training_image_id != PHYSICSNEMO_IMAGE_ID
            or source.get("data", {}).get("frozen_test_transferred_or_opened")
            is not False
        ):
            raise ValueError("source snapshot receipt differs")

        actual_tree = subprocess.run(
            ["git", "rev-parse", f"{source_commit}^{{tree}}"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_tree != source_tree:
            raise ValueError("source snapshot tree differs from launch commit")

    train16_manifest_sha = TRAIN16_MANIFEST_SHA if is_train16 else None
    if is_train16:
        source_data = source.get("data", {})
        dev30_source = source_data.get("dev30", {})
        train8_source = source_data.get("train8", source_data.get("dynamic_train8", {}))
        train16_source = source_data.get(
            "train16", source_data.get("directppo_train16", {})
        )
        source_parent = source.get("parent", {})
        if (
            source.get("status") != "CONTROL_TRAIN16_H100_STAGED_NOT_EXECUTED"
            or source.get("candidate_kind") != candidate_kind
            or dev30_source.get("manifest_sha256") != DEV30_MANIFEST_SHA
            or dev30_source.get("normalization_sha256") != NORMALIZATION_SHA
            or train8_source.get("manifest_sha256") != TRAIN8_MANIFEST_SHA
            or train8_source.get("normalization_sha256") != NORMALIZATION_SHA
            or train8_source.get("file_count") != 8
            or train16_source.get("manifest_sha256") != train16_manifest_sha
            or train16_source.get("normalization_sha256") != NORMALIZATION_SHA
            or train16_source.get("file_count") != 16
            or source_parent.get("sha256") != parent_hashes
            or source_parent.get("checkpoint_epoch") != 2
            or source_parent.get("candidate_kind") != "dynamic_train8_h100"
            or source_parent.get("optimizer_loaded") is not False
            or source_parent.get("new_optimizer") != "AdamW"
        ):
            raise ValueError("train16 launch source/data/parent receipt differs")
        snapshot_hashes = source.get("source_snapshot_sha256")
        snapshot_root = root / "source_snapshot"
        if not isinstance(snapshot_hashes, dict) or not snapshot_hashes:
            raise ValueError("train16 source snapshot hash map is absent")
        for relative, digest in snapshot_hashes.items():
            if (
                not isinstance(relative, str)
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
            ):
                raise ValueError("train16 source snapshot entry is invalid")
            path = snapshot_root / relative
            if not path.is_file() or sha256(path) != digest:
                raise ValueError("train16 source snapshot differs from receipt")
        actual_snapshot_files = {
            str(path.relative_to(snapshot_root))
            for path in snapshot_root.rglob("*")
            if path.is_file()
        }
        if actual_snapshot_files != set(snapshot_hashes):
            raise ValueError("train16 source snapshot file set differs")
        required_snapshot_files = {
            "scripts/train_tandem_fno.py",
            "scripts/train_tandem_fno_rollout.py",
            "scripts/probe_directppo_train16_datapipe.py",
            "scripts/run_fno_control_train16_spark.sh",
            "src/fluid_control/augmented_datapipe.py",
            "src/fluid_control/tandem_datapipe.py",
            "conf/tandem_fno_control_train16_h100.yaml",
        }
        if not required_snapshot_files <= set(snapshot_hashes):
            raise ValueError("train16 source snapshot lacks required implementation files")

        probe_path = repo / TRAIN16_PROBE
        probe = load_json(probe_path)
        probe_sources = probe.get("additional_sources")
        probe_samples = probe.get("samples")
        expected_probe_sources = [
            (
                "/workspace/train8",
                408,
                TRAIN8_MANIFEST_SHA,
                8,
                "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED",
            ),
            (
                "/workspace/train16",
                240,
                train16_manifest_sha,
                16,
                "DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED",
            ),
        ]
        if (
            probe.get("status")
            != "DIRECTPPO_TRAIN16_OFFICIAL_DATAPIPE_PROBE_PASS"
            or probe.get("training_executed") is not False
            or probe.get("validation_or_frozen_hdf_opened") is not False
            or probe.get("rollout_steps") != 100
            or probe.get("base_windows") != 720
            or probe.get("total_windows") != 1368
            or not isinstance(probe_sources, list)
            or len(probe_sources) != 2
            or not isinstance(probe_samples, list)
            or len(probe_samples) != 4
            or {row.get("metadata", {}).get("dataset_index") for row in probe_samples}
            != {0, 1, 2}
            or probe.get("script_sha256")
            != snapshot_hashes["scripts/probe_directppo_train16_datapipe.py"]
            or probe.get("implementation_sha256")
            != {
                relative: snapshot_hashes[relative]
                for relative in (
                    "src/fluid_control/augmented_datapipe.py",
                    "src/fluid_control/tandem_datapipe.py",
                )
            }
            or source.get("datapipe_probe_sha256") != sha256(probe_path)
        ):
            raise ValueError("train16 official DataPipe probe differs")
        expected_shapes = {
            "state": [1, 3, 128, 256],
            "target_state": [1, 100, 3, 128, 256],
            "target_force": [1, 100, 4],
            "omega": [1, 101, 1],
            "mask": [1, 1, 128, 256],
        }
        for row in probe_samples:
            dataset_index = row.get("metadata", {}).get("dataset_index")
            tolerance = 2.0e-6 if dataset_index == 2 else 2.0e-5
            if (
                row.get("metadata", {}).get("split") != "train"
                or row.get("metadata", {}).get("rollout_steps") != 100
                or any(row.get("shapes", {}).get(key) != value for key, value in expected_shapes.items())
                or not math.isfinite(float(row.get("max_abs_omega", math.nan)))
                or not math.isfinite(float(row.get("max_delta_omega", math.nan)))
                or float(row["max_abs_omega"]) > 0.75 + tolerance
                or float(row["max_delta_omega"]) > 0.1 + tolerance
                or not math.isclose(
                    float(row.get("action_slew_representation_tolerance", math.nan)),
                    tolerance,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("train16 official DataPipe sample contract differs")
        for row, (expected_root, windows, manifest_sha, count, status) in zip(
            probe_sources, expected_probe_sources, strict=True
        ):
            if (
                row.get("root") != expected_root
                or row.get("windows") != windows
                or row.get("stride") != 2
                or row.get("manifest_sha256") != manifest_sha
                or row.get("normalization_sha256") != NORMALIZATION_SHA
                or row.get("trajectory_count") != count
                or row.get("release_status") != status
            ):
                raise ValueError("train16 official DataPipe probe source differs")

    source_config = repo / source_config_relative
    launch_training_paths = [
        "scripts/train_tandem_fno.py",
        "scripts/train_tandem_fno_rollout.py",
        "src/fluid_control/augmented_datapipe.py",
        source_config_relative,
    ]
    current_recovery_and_validation = [
        repo / "scripts/train_tandem_fno.py",
        repo / "scripts/train_tandem_fno_rollout.py",
        repo / "scripts/evaluate_tandem_fno.py",
        repo / "src/fluid_control/augmented_datapipe.py",
        source_config,
    ]
    return {
        "status": "DYNAMIC_FNO_CANDIDATE_LINEAGE_PASS",
        "candidate_kind": candidate_kind,
        "candidate_root": str(root.relative_to(repo)),
        "selected_epoch": selected_epoch,
        "selection_rule": "unique minimum training_history.selection_score",
        "checkpoint_model": str(model.relative_to(repo)),
        "checkpoint_sha256": sha256(model),
        "checkpoint_state": str(state.relative_to(repo)),
        "checkpoint_state_sha256": sha256(state),
        "resolved_config_sha256": sha256(config_path),
        "training_history_sha256": sha256(root / "training_history.json"),
        "training_data_sources_sha256": sha256(sources_path),
        "parent_receipt_sha256": sha256(parent_path),
        "source_receipt_sha256": sha256(source_path) if source else None,
        "datapipe_probe_sha256": (
            sha256(repo / TRAIN16_PROBE) if is_train16 else None
        ),
        "source_snapshot_commit": source_commit,
        "source_snapshot_tree": source_tree,
        "source_snapshot_recorded_at_launch": source is not None,
        "source_snapshot_limitation": (
            None
            if source is not None
            else "launch-time git commit was not recorded; exact archived config, logs, model and parent remain bound, but current implementation SHAs are not claimed as training-time code"
        ),
        "training_implementation_sha256_at_launch": (
            {
                relative: git_blob_sha256(repo, source_commit, relative)
                for relative in launch_training_paths
            }
            if source_commit is not None
            else None
        ),
        "current_recovery_and_validation_implementation_sha256": {
            str(path.relative_to(repo)): sha256(path)
            for path in current_recovery_and_validation
        },
        "implementation_identity_policy": (
            "launch-commit blobs identify training when recorded; current worktree "
            "hashes identify only recovery/formal validation and are never relabelled "
            "as the historical training implementation"
        ),
        "data_lineage": {
            "dev30_manifest_sha256": DEV30_MANIFEST_SHA,
            "train8_manifest_sha256": TRAIN8_MANIFEST_SHA,
            "train16_manifest_sha256": train16_manifest_sha,
            "formal_full40_manifest_sha256": FULL40_MANIFEST_SHA,
            "normalization_sha256": NORMALIZATION_SHA,
            "base_windows": expected_windows[0],
            "train8_windows": expected_windows[1],
            "train16_windows": expected_windows[2] if is_train16 else None,
        },
        "parent_initialization_policy": (
            "immutable H100-e2 model and state files copied as parent; training code "
            "loads only model weights from initial_checkpoint and creates a new AdamW"
            if is_train16
            else "candidate-specific immutable parent receipt"
        ),
        "training_image_id_recorded_at_launch": training_image_id,
        "required_formal_validation_image_id": PHYSICSNEMO_IMAGE_ID,
        "validation_horizon_steps": 100,
        "surrogate_ppo_episode_steps_required": 100,
        "frozen_test_opened_or_enumerated": False,
        "history": history,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = build(args.repo, args.candidate_root)
    write_exclusive(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
