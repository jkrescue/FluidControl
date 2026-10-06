#!/usr/bin/env python3
"""Bounded train-only FC-P030 start0/H100 diagnostic.

The scientific arithmetic lives in ``p030_train_horizon_core``.  This entry
owns only reviewed input identity, official DataPipe/model loading, and
streaming one selected trajectory at a time.  It never reads validation or
frozen-test data and never saves or trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

APPROVED = "P030_TRAIN_HORIZON_DIAGNOSTIC_EXECUTION_APPROVED"
COMPLETE = "P030_TRAIN_HORIZON_DIAGNOSTIC_COMPLETE_NOT_ADMISSION"
NORMALIZATION_SHA256 = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
PHASE_MAP_SHA256 = "57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92"
TRAIN_AUDIT_SHA256 = "03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9"
FAMILIES = {
    "base": (20, 20, 801, 0),
    "train8": (8, 2, 201, 1),
    "train16": (16, 2, 129, 2),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path: Path | str, expected: str) -> Path:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular input absent: {path}")
    require(sha256(path) == expected, f"input SHA differs: {path}")
    return path


def load_spec(path: Path, expected_sha256: str) -> dict:
    checked(path, expected_sha256)
    spec = json.loads(path.read_text())
    require(spec.get("status") == APPROVED, "P030 execution is not approved")
    require(spec.get("execution_authorized") is True, "P030 authorization flag is false")
    require(set(spec.get("candidates", {})) == {"k1", "p029"}, "exact K1/P029 candidates required")
    require(set(spec.get("data", {})) == set(FAMILIES), "exact train families required")
    require(spec.get("p029_terminal_proof", {}).get("reviewed_by_lead") is True,
            "reviewed P029 terminal proof required")
    require(spec.get("p029_official_cpu_reload", {}).get("status") ==
            "FC_P029_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
            "P029 CPU reload binding required")
    require(spec.get("comparison_contract") == {
        "split": "train", "starts": [0], "windows": 44, "rollout_steps": 100,
        "report_leads": [1, 10, 25, 50, 100], "selection_performed": False,
    }, "P030 comparison contract differs")
    return spec


def runtime_dependencies():
    from evaluate_tandem_fno import field_error_sums, load_composed_config, relative_field_metrics
    from p026_state_history import build_input
    from p030_train_horizon_core import (
        assert_h1_force_identity, grouped_and_paired, rollout_window, validate_selection,
    )
    from train_tandem_fno import build_model, predict
    from fluid_control.dual_fno import load_dual_fno
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    return (field_error_sums, load_composed_config, relative_field_metrics, build_input,
            assert_h1_force_identity, grouped_and_paired, rollout_window, validate_selection, build_model,
            predict, load_dual_fno, TandemRolloutDataset)


def phase_rows(mapping: dict, spec: dict) -> dict[tuple[str, str], dict]:
    require(mapping.get("status") == "FCP003C_FULL_TRAIN_SOURCE_PHASE_MAPPING_COMPLETE",
            "phase-map status differs")
    rows = {}
    for row in mapping["trajectories"]:
        family = "base" if row["family"] == "base20" else row["family"]
        key = (family, row["file"])
        require(family in FAMILIES and key not in rows, "phase-map identity differs")
        rows[key] = {
            "canonical_phase": row["canonical_physical_phase"],
            "action_profile": action_profile(family, Path(row["file"]).stem),
        }
    expected = {(family, name) for family in FAMILIES for name in spec["data"][family]["train_files"]}
    require(set(rows) == expected and len(rows) == 44, "phase map must cover exact44")
    return rows


def action_profile(family: str, case: str) -> str:
    """Report deterministic existing action families; never select by outcome."""
    if family == "base":
        value = case.rsplit("_", 1)[-1]
        require(value in {"zero", "m0375", "m075", "p0375", "p075"},
                "base action profile differs")
        return value
    if family == "train8":
        value = case.rsplit("_", 1)[-1]
        require(value in {"prbs", "multisine"}, "train8 action profile differs")
        return value
    if family == "train16":
        require("_directppo2048_v1_" in case, "train16 case identity differs")
        return "historical_directppo_exploration"
    raise RuntimeError("unknown action-profile family")


def original_membership(datasets: dict[str, object]) -> list[tuple[str, int, int]]:
    result = []
    for family, dataset in datasets.items():
        dataset_index = FAMILIES[family][3]
        result.extend((dataset.paths[file_index].stem, int(start), dataset_index)
                      for file_index, start in dataset.index)
    return result


def validate_original_membership(datasets: dict[str, object]) -> list[tuple[str, int, int]]:
    membership = original_membership(datasets)
    expected = {"base": 720, "train8": 408, "train16": 240}
    actual = {family: len(dataset.index) for family, dataset in datasets.items()}
    require(actual == expected and len(membership) == 1368 and len(set(membership)) == 1368,
            "original 1368 H100 sampler membership differs")
    return membership


def selection_rows(datasets: dict[str, object], spec: dict, phases: dict) -> list[dict]:
    rows = []
    for family, dataset in datasets.items():
        frames, dataset_index = FAMILIES[family][2], FAMILIES[family][3]
        entry = spec["data"][family]
        for path in dataset.paths:
            provenance = phases[(family, path.name)]
            rows.append({
                "family": family, "dataset_index": dataset_index, "case": path.stem,
                "start": 0, "rollout_steps": 100,
                "canonical_phase": provenance["canonical_phase"],
                "action_profile": provenance["action_profile"],
                "source_manifest_sha256": entry["manifest_sha256"],
                "hdf_sha256": entry["train_files"][path.name],
                "split": "train", "frames": frames,
            })
    return rows


def sample_start0(dataset, file_index: int) -> tuple[dict, dict]:
    matches = [i for i, identity in enumerate(dataset.index) if identity == (file_index, 0)]
    require(len(matches) == 1, "exact one start0 index required")
    sample, metadata = dataset[matches[0]]
    require(metadata == {"case": dataset.paths[file_index].stem, "step": 0,
                         "rollout_steps": 100, "split": "train"},
            "official start0 metadata differs")
    reader = dataset._reader(file_index)
    times = torch.stack([reader[j][0]["time"].float().reshape(1) for j in range(101)])
    result = {key: value[None] for key, value in sample.items() if key != "time"}
    result["time"] = times[None]
    return result, metadata


def make_inputs_with_reviewed_helper(build_input, state, mask, now, nxt):
    """Batch-one K1 packing through the reviewed shared helper."""
    require(state.shape[0] == 1 and mask.shape[0] == 1, "P030 uses batch one")
    packed = build_input(state[0, None], mask[0], now[0].reshape(1), nxt[0])
    return packed[None]


def same_state_dict(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    return left.state_dict().keys() == right.state_dict().keys() and all(
        torch.equal(left.state_dict()[key], right.state_dict()[key]) for key in left.state_dict()
    )


def atomic_json(path: Path, value: dict) -> None:
    """Publish once into an existing exclusive output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), "exclusive result already exists")
    temporary = path.with_name("." + path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_selection(output: Path, rows: list[dict], selection: dict,
                      source_spec_sha256: str) -> dict:
    require(len(source_spec_sha256) == 64 and
            all(char in "0123456789abcdef" for char in source_spec_sha256),
            "exact source spec SHA required")
    path = output.parent / "selection.json"
    atomic_json(path, {"rows": rows, "selection": selection,
                       "source_spec_sha256": source_spec_sha256})
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def persist_raw_records(output: Path, records: dict, rows: list[dict],
                        selection_manifest: dict, source_spec_sha256: str) -> dict:
    path = output.parent / "raw_records.json"
    atomic_json(path, {"records": records, "rows": rows,
                       "selection_manifest": selection_manifest,
                       "source_spec_sha256": source_spec_sha256})
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def validated_runtime_dependencies(spec: dict):
    """Verify the complete small source/proof closure before importing it."""
    for item in spec["source_files"]:
        checked(item["path"], item["sha256"])
    checked(spec["config"]["path"], spec["config"]["sha256"])
    checked(spec["train_audit"]["path"], TRAIN_AUDIT_SHA256)
    checked(spec["source_phase_mapping"]["path"], PHASE_MAP_SHA256)
    checked(spec["train16_predeclaration"]["path"], spec["train16_predeclaration"]["sha256"])
    checked(spec["p029_terminal_proof"]["path"], spec["p029_terminal_proof"]["sha256"])
    checked(spec["p029_official_cpu_reload"]["path"], spec["p029_official_cpu_reload"]["sha256"])
    return runtime_dependencies()


def execute(spec: dict, output: Path, source_spec_sha256: str) -> dict:
    started = time.monotonic()

    def guard(startup: bool = False) -> None:
        values = {line.split(":")[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
        require(values["MemFree"] >= (30 if startup else 20) * 2**30, "MemFree guard")
        require(values["MemAvailable"] >= (50 if startup else 20) * 2**30, "MemAvailable guard")
        require(time.monotonic() - started <= 900, "P030 900s deadline")
        if torch.cuda.is_initialized():
            require(torch.cuda.mem_get_info()[0] >= 20 * 2**30, "CUDA20GiB floor")

    guard(True)
    deps = validated_runtime_dependencies(spec)
    (field_error_sums, load_config, relative_field_metrics, build_input,
     assert_h1_force_identity, grouped_and_paired, rollout_window, validate_selection,
     build_model, predict, load_dual_fno, Dataset) = deps
    cfg = load_config(Path(spec["config"]["path"]))
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(.06)
    device = torch.device("cuda")
    models = {}
    identities = {}
    for name in ("k1", "p029"):
        entry = spec["candidates"][name]
        models[name], identities[name] = load_dual_fno(
            Path(entry["manifest"]), cfg, device, build_model=build_model,
            expected_manifest_sha256=entry["manifest_sha256"])
    require(identities["k1"].payload["kind"] == "FC_P026_K1_HISTORY_FORCE_FNO",
            "K1 candidate kind differs")
    require(identities["p029"].payload["kind"] == "FC_P029_CONTROL_AWARE_FLOW_REPAIR",
            "P029 candidate kind differs")
    for identity in identities.values():
        require(identity.payload["config_sha256"] == spec["config"]["sha256"] and
                identity.payload["normalization_sha256"] == NORMALIZATION_SHA256,
                "candidate config/normalization differs")
    require(same_state_dict(models["k1"].aerodynamic_model, models["p029"].aerodynamic_model),
            "K1/P029 aerodynamic tensors differ")
    models["p029"].aerodynamic_model.cpu()
    snapshots = [(value, value._version) for adapter in models.values()
                 for value in list(adapter.parameters()) + list(adapter.buffers())]

    datasets = {}
    try:
        audit = json.loads(Path(spec["train_audit"]["path"]).read_text())
        planned = {f"data/curated/{Path(spec['data'][family]['root']).name}/train/{name}": digest
                   for family in FAMILIES
                   for name, digest in spec["data"][family]["train_files"].items()}
        require(audit.get("train_hdf_sha256") == planned and len(planned) == 44,
                "reviewed exact44 train map differs")
        for family, (_, stride, _, _) in FAMILIES.items():
            entry = spec["data"][family]
            root = Path(entry["root"])
            checked(root / "manifest.json", entry["manifest_sha256"])
            checked(root / "normalization.json", entry["normalization_sha256"])
            require(entry["normalization_sha256"] == NORMALIZATION_SHA256,
                    "train normalization differs")
            datasets[family] = Dataset(root, "train", 100, stride=stride,
                                       num_workers=0, force_indices=(0, 1, 2, 3))
            require(len(datasets[family].paths) == FAMILIES[family][0] and
                    {p.name for p in datasets[family].paths} == set(entry["train_files"]),
                    "train inventory differs")
            require(float(datasets[family].action_scale) == .75,
                    "original action_scale must remain 0.75")
            for path in datasets[family].paths:
                checked(path, entry["train_files"][path.name])
        phases = phase_rows(json.loads(Path(spec["source_phase_mapping"]["path"]).read_text()), spec)
        rows = selection_rows(datasets, spec, phases)
        selection = validate_selection(rows, validate_original_membership(datasets))
        selection_manifest = persist_selection(output, rows, selection, source_spec_sha256)
        records = {"k1": [], "p029": []}
        h1_force_identity = []
        for count, row in enumerate(rows, 1):
            guard()
            dataset = datasets[row["family"]]
            file_index = next(i for i, path in enumerate(dataset.paths) if path.stem == row["case"])
            sample, _ = sample_start0(dataset, file_index)
            sample = {key: value.to(device) for key, value in sample.items()}
            common = dict(sample=sample, predict_fn=predict,
                          make_inputs_fn=lambda q, m, a, b: make_inputs_with_reviewed_helper(
                              build_input, q, m, a, b),
                          field_error_sums_fn=field_error_sums,
                          state_mean=dataset.state_mean.to(device), state_std=dataset.state_std.to(device),
                          force_mean=dataset.force_mean.to(device), force_std=dataset.force_std.to(device),
                          identity={"case": row["case"], "start": 0,
                                    "dataset_index": row["dataset_index"]}, guard=guard)
            for name in ("k1", "p029"):
                records[name].append(rollout_window(models[name].flow_model,
                    models["k1"].aerodynamic_model, **common))
            assert_h1_force_identity(records["k1"][-1], records["p029"][-1])
            k1_h1 = records["k1"][-1]["predicted_force_physical_by_lead"][0]
            h1_force_identity.append(k1_h1)
            print(json.dumps({"event": "origin_complete", "case": row["case"],
                              "count": count}), flush=True)
        raw_records = persist_raw_records(
            output, records, rows, selection_manifest, source_spec_sha256)
        summary = grouped_and_paired(records["k1"], records["p029"], rows,
                                     relative_field_metrics_fn=relative_field_metrics)
        require(all(value._version == version for value, version in snapshots),
                "model parameter/buffer version changed")
        result = {
            "status": COMPLETE, "source_spec": spec, "selection": selection,
            "selection_manifest": selection_manifest,
            "raw_records": raw_records,
            "records": records, "summary": summary,
            "h1_force_identity_exact": True, "h1_force_identity": h1_force_identity,
            "flow_transition_counts": {"k1": 4400, "p029": 4400},
            "aerodynamic_evaluation_counts": {"k1": 4400, "p029": 4400},
            "total_model_forward_calls": 17600,
            "optimizer_created": False, "model_saved": False,
            "validation_accessed": False, "frozen_test_accessed": False,
            "scientific_admission": False, "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(output, result)
        return result
    finally:
        for dataset in datasets.values():
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec, args.spec_sha256)
    if not args.execute:
        print(json.dumps({"status": "P030_TRAIN_HORIZON_DIAGNOSTIC_PREPARATION_ONLY_NOT_APPROVED",
                          "execution_authorized": False}, sort_keys=True))
        return
    require(not args.output.exists(), "exclusive output already exists")
    execute(spec, args.output, args.spec_sha256)


if __name__ == "__main__":
    main()
