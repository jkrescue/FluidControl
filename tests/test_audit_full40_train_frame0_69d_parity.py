from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from audit_full40_train_frame0_69d_parity import CASES, audit


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root: Path, *, wrong_source_sha: bool = False) -> None:
    data = root / "data/curated/tandem_cylinders_matched_start_full40_v1"
    cases = root / "cfd/tandem_cylinders/cases"
    source = cases / "tandem_backward_dt005"
    front = source / "postProcessing/forceFront/0/coefficient.dat"
    rear = source / "postProcessing/forceRear/0/coefficient.dat"
    front.parent.mkdir(parents=True)
    rear.parent.mkdir(parents=True)
    front.write_text("# Time Cd x x Cl\n0 1 0 0 2\n", encoding="utf-8")
    rear.write_text("# Time Cd x x Cl\n0 3 0 0 4\n", encoding="utf-8")
    (data / "train").mkdir(parents=True)
    (data / "splits").mkdir()
    times = (148.0, 106.0, 120.0, 134.0)
    hdf_sha = {}
    for case, time in zip(CASES, times, strict=True):
        config = {
            "case": case,
            "split": "train",
            "source_restart_case": source.name,
            "source_restart_time": time,
            "source_force_sha256": {
                "forceFront": "0" * 64 if wrong_source_sha else sha256(front),
                "forceRear": sha256(rear),
            },
        }
        config_path = cases / case / "case_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        hdf = data / "train" / f"{case}.h5"
        with h5py.File(hdf, "w") as handle:
            handle.attrs["case"] = case
            handle.attrs["split"] = "train"
            handle.attrs["config_json"] = json.dumps(config)
            handle.create_dataset("time", data=np.asarray([[time]]))
            handle.create_dataset("omega", data=np.asarray([[0.0]]))
            handle.create_dataset("state", data=np.zeros((1, 3, 2, 2), np.float32))
            handle.create_dataset("force", data=np.zeros((1, 4), np.float32))
            handle.create_dataset("x", data=np.asarray([16.0, 18.0]))
            handle.create_dataset("y", data=np.asarray([5.0, 10.0]))
        hdf_sha[case] = sha256(hdf)
    split = data / "splits/train.json"
    split.write_text(
        json.dumps({"split": "train", "cases": list(CASES), "hdf5_sha256": hdf_sha}),
        encoding="utf-8",
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "matched_start_full40_v1",
                "split_manifests": {
                    "train": {"path": "splits/train.json", "sha256": sha256(split)}
                },
            }
        ),
        encoding="utf-8",
    )


def comparison(case: str) -> dict:
    return {
        "status": "TANDEM_69D_OBSERVATION_PARITY_OK",
        "comparisons": [
            {
                "frame": 0,
                "time": 0.0,
                "probe_max_abs_error": 0.0,
                "front_force_max_abs_error": 0.0,
                "rear_force_max_abs_error": 0.0,
                "omega_abs_error": 0.0,
                "raw_front_force_sources": [case],
            }
        ],
    }


class Full40TrainFrame0ParityTests(unittest.TestCase):
    def test_uses_exact_four_train_cases_and_only_frame_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root)
            with patch(
                "audit_full40_train_frame0_69d_parity.compare",
                side_effect=lambda data, cases, split, case, frames: comparison(case),
            ) as mocked, patch(
                "audit_full40_train_frame0_69d_parity.bilinear_probes",
                return_value=np.zeros((32, 2), dtype=np.float64),
            ), patch(
                "audit_full40_train_frame0_69d_parity.total_drag_observation_at",
                return_value=(np.zeros(69, dtype=np.float64), {}),
            ):
                result = audit(root)
            self.assertEqual(result["status"], "FULL40_TRAIN_FRAME0_69D_PARITY_PASS")
            self.assertEqual([row["case"] for row in result["cases"]], list(CASES))
            self.assertEqual(
                [(call.args[2], call.args[3], call.args[4]) for call in mocked.call_args_list],
                [("train", case, [0]) for case in CASES],
            )
            self.assertTrue(
                all(len(row["raw_observation_69d"]) == 69 for row in result["cases"])
            )

    def test_rejects_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root)
            manifest = root / "data/curated/tandem_cylinders_matched_start_full40_v1/manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["split_manifests"]["train"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest identity differs"):
                audit(root)

    def test_rejects_raw_force_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root, wrong_source_sha=True)
            with self.assertRaisesRegex(ValueError, "raw force SHA differs"):
                audit(root)


if __name__ == "__main__":
    unittest.main()
