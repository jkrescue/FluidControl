from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.audit_fc_p002_action_coverage import build, summarize_case


def make_case(path: Path, omega: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("omega", data=np.asarray(omega))
        handle.create_dataset("time", data=np.arange(len(omega)) * 0.1)


def test_metric_definitions_handle_zero_crossing(tmp_path: Path) -> None:
    path = tmp_path / "case.h5"
    make_case(path, [-0.1, 0.0, 0.1, 0.1])
    result = summarize_case(path)
    assert result["change_fraction"] == pytest.approx(2 / 3)
    assert result["total_variation"] == pytest.approx(0.2)
    assert result["longest_constant_dwell_du"] == pytest.approx(0.2)
    assert result["sign_crossings_zeros_removed"] == 1


def test_build_rejects_frozen_path(tmp_path: Path) -> None:
    roots = []
    hashes = []
    for index, name in enumerate(("train20", "train8", "train16", "frozen_test")):
        root = tmp_path / name / "train"
        make_case(root / "case.h5", [0.0, 0.1])
        manifest = root.parent / "manifest.json"
        manifest.write_text(json.dumps({"index": index}))
        roots.append(root)
        hashes.append(hashlib.sha256(manifest.read_bytes()).hexdigest())
    args = Namespace(
        train20=roots[0], train8=roots[1], train16=roots[2], dynamic6=roots[3],
        dev30_manifest_sha=hashes[0], train8_manifest_sha=hashes[1],
        train16_manifest_sha=hashes[2], dynamic6_manifest_sha=hashes[3],
    )
    with pytest.raises(ValueError, match="frozen data are forbidden"):
        build(args)
