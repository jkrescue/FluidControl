"""Create and verify one immutable-provenance, train-only b02 HDF view."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch


EXPECTED_RESULT = "B02_CONTROLLED_TRAIN_HDF_COMPLETE_NOT_TRAINING"
NORM_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_window(handle, start, stats, action_scale=.75):
    mask = torch.from_numpy(handle["mask"][start]).float()
    states = torch.from_numpy(handle["state"][start:start + 101]).float()
    mean = torch.tensor(stats["state_mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(stats["state_std"], dtype=torch.float32)[:, None, None]
    states = (states - mean[None]) / std[None]
    states *= mask[None]
    omega = torch.from_numpy(handle["omega"][start:start + 101]).float() / action_scale
    force = torch.from_numpy(handle["force"][start + 1:start + 101]).float()
    force = (force - torch.tensor(stats["all_force_mean"])) / torch.tensor(stats["all_force_std"])
    return states, omega, force, mask


def verify_windows(root: Path, hdf: Path):
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    dataset = TandemRolloutDataset(root, "train", 100, stride=1,
                                   num_workers=1, force_indices=(0, 1, 2, 3))
    try:
        require(len(dataset) == 701, "H100 start count")
        stats = json.loads((root / "normalization.json").read_text())
        checks = []
        with h5py.File(hdf, "r") as handle:
            for index, start in ((0, 0), (700, 700)):
                sample, metadata = dataset[index]
                states, omega, force, mask = expected_window(handle, start, stats)
                require(torch.equal(sample["state"], states[0]), "state")
                require(torch.equal(sample["target_state"], states[1:]), "target state")
                require(torch.equal(sample["omega"], omega), "omega")
                require(torch.equal(sample["target_force"], force), "force")
                require(torch.equal(sample["mask"], mask), "mask")
                require(metadata["step"] == start and metadata["rollout_steps"] == 100,
                        "metadata")
                checks.append({"start": start, "endpoint": start + 100,
                               "metadata": metadata})
        return checks
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion-result", type=Path, required=True)
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute, "explicit --execute required")
    require(not args.output.exists() and not args.output.is_symlink(), "exclusive output")
    result = json.loads(args.conversion_result.read_text())
    require(result.get("status") == EXPECTED_RESULT, "conversion result")
    require(result.get("frames") == 801 and result.get("split") == "train", "conversion shape")
    require(result.get("official_reader_verified") is True and
            result.get("normalization_refit") is False, "conversion verification")
    require(sha(args.normalization) == NORM_SHA == result.get("normalization_sha256"),
            "normalization identity")
    hdf_sha = sha(args.hdf)
    require(result["hdf"]["sha256"] == hdf_sha, "HDF identity")
    before = args.hdf.stat()
    args.output.mkdir()
    train = args.output / "train"
    train.mkdir()
    linked = train / "b02_canonical_ppo_train.h5"
    os.link(args.hdf, linked)
    shutil.copyfile(args.normalization, args.output / "normalization.json")
    manifest = {
        "status": "B02_CONTROLLED_TRAIN_VIEW_NOT_TRAINING_APPROVAL",
        "split": "train", "trajectories": 1, "frames": 801,
        "max_abs_omega": .75, "normalization_refit": False,
        "normalization_sha256": NORM_SHA,
        "hdf_sha256": {linked.name: hdf_sha},
        "conversion_receipt": str(args.conversion_result),
        "conversion_receipt_sha256": sha(args.conversion_result),
        "label_provenance": "actual CFD endpoint observation coefficients",
        "validation_accessed": False, "frozen_test_accessed": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checks = verify_windows(args.output, linked)
    after = args.hdf.stat()
    require(sha(args.hdf) == hdf_sha and before.st_mode == after.st_mode and
            before.st_mtime_ns == after.st_mtime_ns, "source HDF unchanged")
    require(not any((args.output / name).exists() for name in
                    ("validation", "test", "frozen_test")), "train only")
    receipt = {
        "status": "B02_TRAIN_VIEW_CPU_VERIFIED_NOT_TRAINING",
        "hdf_sha256": hdf_sha, "normalization_sha256": NORM_SHA,
        "official_reader_frames": 801, "windows": 701,
        "window_checks": checks, "hardlink": linked.stat().st_ino == args.hdf.stat().st_ino,
        "source_unchanged": True, "training_authorized": False,
    }
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for path in (args.output / "normalization.json", args.output / "manifest.json",
                 args.output / "verification.json"):
        path.chmod(0o444)
    for path in (train, args.output):
        path.chmod(0o555)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
