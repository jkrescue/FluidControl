"""Compose verified train-only trajectories using official PhysicsNeMo MultiDataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from physicsnemo.datapipes import MultiDataset

from .tandem_datapipe import TandemRolloutDataset


def compose_training_data(base, roots, *, rollout_steps, stride, workers, force_indices):
    """Keep the parent normalization and train split across all dataset sources.

    Each extra dataset must be a finalized train-only Curator profile. Sampling
    density is controlled by the documented window stride, not by modifying or
    duplicating the CFD files. MultiDataset preserves per-source metadata.
    """
    if not roots:
        return base, []
    reference = (base.root / "normalization.json").read_bytes()
    reference_sha = hashlib.sha256(reference).hexdigest()
    datasets = [base]
    records = []
    try:
        for item in roots:
            root = Path(item).resolve()
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "DYNAMIC_TRAIN8_TRAIN_ONLY_CURATED":
                raise ValueError(f"additional training dataset is not finalized: {root}")
            if manifest.get("trajectory_counts") != {"train": 8, "validation": 0, "frozen_test": 0}:
                raise ValueError("additional dataset must have exactly eight train-only trajectories")
            if any((root / name).exists() for name in ("validation", "test", "frozen_test")):
                raise ValueError("additional training root exposes a non-training split")
            if (root / "normalization.json").read_bytes() != reference:
                raise ValueError("additional trajectories must use identical parent normalization")
            if float(manifest.get("max_abs_omega", -1)) != base.action_scale:
                raise ValueError("additional trajectory action normalization differs")
            paths = sorted((root / "train").glob("*.h5"))
            declared = manifest.get("hdf_sha256", {})
            expected = {f"dynamic_train8_b{phase:02d}_{kind}.h5"
                        for phase in (0, 2, 4, 6) for kind in ("prbs", "multisine")}
            if {p.name for p in paths} != expected or set(declared) != expected:
                raise ValueError("additional train phase/file set differs")
            for path in paths:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        digest.update(block)
                if digest.hexdigest() != declared[path.name]:
                    raise ValueError(f"additional training HDF SHA differs: {path.name}")
            dataset = TandemRolloutDataset(
                root, "train", rollout_steps, stride=stride,
                num_workers=workers, force_indices=force_indices,
            )
            datasets.append(dataset)
            records.append({
                "root": str(root), "windows": len(dataset), "stride": stride,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "normalization_sha256": reference_sha,
            })
        return MultiDataset(*datasets, output_strict=True), records
    except Exception:
        for dataset in datasets[1:]:
            dataset.close()
        raise
