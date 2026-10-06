"""Prepare a non-authorizing metadata draft; never read HDF/model payloads."""

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

from train_exploratory_h5_ppo import PROTOCOL

from exploratory_h5_hydrogym import TRAIN_CASES

REPO = Path("/workspace/fluid_control")
STAGE = Path(__file__).resolve().parent
BASE = REPO / ".venv-curator-py312/lib/python3.12/site-packages"
OVERLAY = REPO / ".runtime/exploratory-h5-ppo-py312"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(path, known=None):
    return {"path": str(path), "sha256": digest(path) if known is None else known}


def main():
    data = REPO / "data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
    split = json.loads((data / "splits/train.json").read_text())
    cases = REPO / "cfd/tandem_cylinders/cases"
    inputs = {
        "manifest": row(
            REPO
            / "artifacts/fcp026_history_training_k1_20261005/candidate/dual_model_manifest.json"
        ),
        "config": row(
            REPO
            / "artifacts/fcp027_diagnostic_source_20261006_immutable/training_config.yaml"
        ),
        "normalization": row(data / "normalization.json"),
        "baseline": row(
            REPO
            / "artifacts/matched_start_full40_extension/train20_physics_summary.json"
        ),
        "data_manifest": row(data / "manifest.json"),
        "train_split": row(data / "splits/train.json"),
    }
    provenance = {}
    case_metadata = []
    for case in TRAIN_CASES:
        config = cases / case / "case_config.json"
        value = json.loads(config.read_text())
        assert value["split"] == "train" and value["action_target"] == 0.0
        provenance[str(config)] = row(config)
        source = cases / value["source_restart_case"]
        for pattern in (
            "postProcessing/forceFront/*/coefficient.dat",
            "postProcessing/forceRear/*/coefficient.dat",
            "postProcessing/wakeProbes/*/U",
        ):
            for path in sorted(source.glob(pattern)):
                # Force SHA inherited from prior case provenance, probes are small text only.
                key = "forceFront" if "forceFront" in str(path) else "forceRear"
                known = None if path.name == "U" else value["source_force_sha256"][key]
                provenance[str(path)] = row(path, known)
        case_metadata.append(
            {
                "case": case,
                "physical_start_time": value["source_restart_time"],
                "source_restart_case": value["source_restart_case"],
            }
        )
    source_paths = [
        *sorted((REPO / "src/fluid_control").glob("*.py")),
        *sorted(STAGE.glob("*.py")),
        *[
            REPO / "scripts" / name
            for name in (
                "train_tandem_fno.py",
                "train_full40_hydrogym_ppo_canonical.py",
                "p026_state_history.py",
                "p026_history_inference.py",
                "probe_k1_uma_inference.py",
            )
        ],
        REPO / ".tools/hydrogym/hydrogym/__init__.py",
        REPO / ".tools/hydrogym/hydrogym/core.py",
    ]
    runtime_paths = [
        BASE / "physicsnemo/models/fno/fno.py",
        BASE / "physicsnemo/utils/checkpoint.py",
    ]
    runtime_paths += sorted(OVERLAY.rglob("*.py"))
    runtime_paths += sorted(OVERLAY.glob("*.dist-info/METADATA"))
    bindings = {
        "exploratory_h5_hydrogym": STAGE / "exploratory_h5_hydrogym.py",
        "train_tandem_fno": REPO / "scripts/train_tandem_fno.py",
        "train_full40_hydrogym_ppo_canonical": REPO
        / "scripts/train_full40_hydrogym_ppo_canonical.py",
        "fluid_control.dual_fno": REPO / "src/fluid_control/dual_fno.py",
        "fluid_control.full40_canonical_hydrogym": REPO
        / "src/fluid_control/full40_canonical_hydrogym.py",
        "fluid_control.tandem_hydrogym": REPO / "src/fluid_control/tandem_hydrogym.py",
        "hydrogym": REPO / ".tools/hydrogym/hydrogym/__init__.py",
        "hydrogym.core": REPO / ".tools/hydrogym/hydrogym/core.py",
        "gymnasium": OVERLAY / "gymnasium/__init__.py",
        "stable_baselines3": OVERLAY / "stable_baselines3/__init__.py",
        "stable_baselines3.common.on_policy_algorithm": OVERLAY
        / "stable_baselines3/common/on_policy_algorithm.py",
        "stable_baselines3.common.vec_env.dummy_vec_env": OVERLAY
        / "stable_baselines3/common/vec_env/dummy_vec_env.py",
        "physicsnemo.models.fno.fno": BASE / "physicsnemo/models/fno/fno.py",
        "physicsnemo.utils.checkpoint": BASE / "physicsnemo/utils/checkpoint.py",
    }
    packages = (
        "torch",
        "numpy",
        "nvidia-physicsnemo",
        "h5py",
        "gymnasium",
        "stable-baselines3",
        "Farama-Notifications",
        "cloudpickle",
        "pandas",
        "matplotlib",
    )
    result = {
        "status": "PREPARATION_ONLY_NOT_EXECUTION_APPROVAL",
        "execution_authorized": False,
        "protocol": PROTOCOL,
        "inputs": inputs,
        "train_cases": list(TRAIN_CASES),
        "train_hdf": [
            row(data / "train" / f"{c}.h5", split["hdf5_sha256"][c])
            for c in TRAIN_CASES
        ],
        "train_provenance_files": list(provenance.values()),
        "case_metadata": case_metadata,
        "data_root": str(data),
        "cases_root": str(cases),
        "source_files": {str(p): digest(p) for p in source_paths},
        "runtime_sources": {str(p): digest(p) for p in runtime_paths},
        "runtime_packages": {p: metadata.version(p) for p in packages},
        "runtime_requirements": {p: metadata.requires(p) for p in packages},
        "import_bindings": {m: str(p) for m, p in bindings.items()},
        "python": str(REPO / ".venv-curator-py312/bin/python"),
        "pythonpath": [
            str(STAGE),
            str(OVERLAY),
            str(REPO / ".tools/hydrogym"),
            str(REPO / "src"),
            str(REPO / "scripts"),
        ],
        "supervisor_base": str(REPO / "scripts/probe_k1_uma_inference.py"),
        "runner": str(STAGE / "train_exploratory_h5_ppo.py"),
        "supervision_output": str(
            REPO / "artifacts/exploratory_h5_ppo_training_20261006"
        ),
        "output": str(REPO / "artifacts/exploratory_h5_ppo_training_20261006/payload"),
        "cpu_lifecycle_invocation": "10d916f24e9f4715b0059cdd32019db7",
        "interpretation": "4096-step exploratory surrogate PPO, no scientific admission or CFD execution",
    }
    with Path(sys.argv[1]).open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
