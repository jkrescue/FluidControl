from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "orchestrate_matched_start_curator.py"
    spec = importlib.util.spec_from_file_location("matched_start_curator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MatchedStartCuratorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def build_repo(
        self,
        root: Path,
        mismatch: str | None = None,
        force_hash_mismatch: bool = False,
    ) -> dict:
        phase = root / self.module.PHASE_MANIFEST
        phase.parent.mkdir(parents=True)
        phase.write_text('{"status":"PREDECLARED"}\n')
        phase_sha = digest(phase)
        baseline = root / "cfd/tandem_cylinders/cases/tandem_backward_dt005"
        source_force_sha256 = {}
        for object_name in self.module.FORCE_OBJECTS:
            force = baseline / "postProcessing" / object_name / "0/coefficient.dat"
            force.parent.mkdir(parents=True, exist_ok=True)
            force.write_text("# Time Cd Cl\n0 1.25 -0.5\n", encoding="utf-8")
            source_force_sha256[object_name] = digest(force)
        if force_hash_mismatch:
            source_force_sha256["forceRear"] = "0" * 64
        entries = []
        for case in self.module.expected_cases():
            match = self.module.CASE_RE.fullmatch(case)
            assert match is not None
            phase_id, branch = match.groups()
            target = {"m075": -0.75, "zero": 0.0, "p075": 0.75}[branch]
            case_root = root / "cfd/tandem_cylinders/cases" / case
            start = case_root / "0"
            start.mkdir(parents=True)
            fields = {}
            token = "different" if mismatch == case else "shared-" + phase_id
            provenance = case_root / "source_restart_provenance"
            provenance.mkdir()
            for name in ("U", "U_0", "p", "phi", "phi_0"):
                path = start / name
                if name == "U":
                    path.write_text(
                        "rearCylinder\n{\n type rotatingWallVelocity;\n omega table\n"
                        f"(\n (0 {target})\n (80 {target})\n);\n}}\n"
                    )
                else:
                    path.write_text(f"{token}-{name}\n")
                # source_state_sha256 describes the pre-action restart. The
                # branch case U file itself differs because its omega table is
                # part of the boundary condition.
                source_path = provenance / name
                source_path.write_text(f"{token}-{name}\n")
                fields[name] = digest(source_path)
            config = {
                "case": case, "split": "train", "phase_bin": int(phase_id),
                "action_target": target, "action_points": [[0.0, target], [80.0, target]],
                "start_time": 0.0, "end_time": 80.0, "analysis_window": [20.0, 80.0],
                "expected_field_frames": 801,
                "expected_aligned_force_samples_with_source_t0": 16001,
                "phase_manifest_sha256": phase_sha, "source_state_sha256": fields,
                "source_restart_case": "tandem_backward_dt005",
                "source_state_provenance_dir": "source_restart_provenance",
                "source_force_sha256": source_force_sha256,
            }
            config_path = case_root / "case_config.json"
            config_path.write_text(json.dumps(config))
            raw_paths = [
                config_path, *[start / name for name in fields],
                *[provenance / name for name in fields],
            ]
            manifest = (
                root / "artifacts/matched_start_acquisition/worker_transfer_manifests"
                / f"{case}.sha256"
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                "".join(
                    f"{digest(path)}  {case}/{path.relative_to(case_root)}\n"
                    for path in raw_paths
                )
            )
            receipt = (
                root / "artifacts/matched_start_acquisition/transfer_verified"
                / f"{case}.json"
            )
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({
                "status": "RAW_TRANSFER_VERIFIED", "case": case,
                "phase_manifest_sha256": phase_sha,
                "worker_raw_manifest": str(manifest.relative_to(root)),
                "worker_raw_manifest_sha256": digest(manifest),
                "raw_file_count": len(raw_paths),
                "coverage": "raw OpenFOAM solver files only; VTK is not included",
                "next_required_stage": "SPARK_PINNED_FOAMTOVTK_801_FRAMES",
                "curator_guard": "do not curate until a separate VTK_READY receipt exists",
            }))
            entries.append({"case": case})
        return {
            "schema_version": 1, "profile": self.module.PROFILE,
            "status": "BOUND_APPROVED_FOR_CURATION",
            "phase_manifest": str(self.module.PHASE_MANIFEST),
            "phase_manifest_sha256": phase_sha, "cases": entries,
        }

    def test_complete_bound_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.module.preflight(root, self.build_repo(root))
            self.assertEqual(report["status"], "CURATOR_PREFLIGHT_OK")
            self.assertEqual(len(report["cases"]), 9)
            self.assertEqual(report["training_use"], "FORBIDDEN_COMMISSIONING_ONLY")
            for row in report["cases"]:
                self.assertEqual(
                    set(row["source_force_provenance"]), {"forceFront", "forceRear"}
                )

    def test_generator_scalar_contract_uses_exact_real_keys(self) -> None:
        config = {
            "phase_bin": 2,
            "expected_field_frames": 801,
            "expected_aligned_force_samples_with_source_t0": 16001,
        }
        self.module.validate_generator_scalar_contract(config, "case", "02")
        old_force_key = dict(config)
        del old_force_key["expected_aligned_force_samples_with_source_t0"]
        old_force_key["force_samples"] = 16001
        with self.assertRaisesRegex(ValueError, "expected_aligned_force"):
            self.module.validate_generator_scalar_contract(old_force_key, "case", "02")
        string_phase = dict(config, phase_bin="b02")
        with self.assertRaisesRegex(ValueError, "phase_bin must be an integer"):
            self.module.validate_generator_scalar_contract(string_phase, "case", "02")

    def test_phase_branch_restart_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mismatch = "matched_start_acquisition_train_b00_p075"
            with self.assertRaisesRegex(ValueError, "five-field restart hashes differ"):
                self.module.preflight(root, self.build_repo(root, mismatch=mismatch))

    def test_unbound_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.build_repo(root)
            contract["status"] = "UNBOUND_DO_NOT_CURATE"
            with self.assertRaisesRegex(ValueError, "not bound"):
                self.module.preflight(root, contract)

    def test_declared_baseline_force_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.build_repo(root, force_hash_mismatch=True)
            with self.assertRaisesRegex(ValueError, "baseline source-force SHA mismatch"):
                self.module.preflight(root, contract)

    def test_baseline_force_mutation_after_case_generation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.build_repo(root)
            force = (
                root
                / "cfd/tandem_cylinders/cases/tandem_backward_dt005/"
                "postProcessing/forceFront/0/coefficient.dat"
            )
            force.write_text("# Time Cd Cl\n0 99 99\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline source-force SHA mismatch"):
                self.module.preflight(root, contract)


if __name__ == "__main__":
    unittest.main()
