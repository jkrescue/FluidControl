import hashlib
import importlib.util
import ast
import json
import os
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_p030_train_horizon.py"
SPEC = importlib.util.spec_from_file_location("p030_driver_test", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

CORE_PATH = Path(os.environ.get(
    "P030_CORE_SOURCE",
    Path(__file__).resolve().parents[1] / "scripts" / "p030_train_horizon_core.py",
))
CORE_SPEC = importlib.util.spec_from_file_location("p030_real_core_test", CORE_PATH)
CORE = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(CORE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimal_spec(tmp_path):
    spec = {
        "status": MOD.APPROVED,
        "execution_authorized": True,
        "candidates": {"k1": {}, "p029": {}},
        "data": {key: {} for key in MOD.FAMILIES},
        "p029_terminal_proof": {"reviewed_by_lead": True},
        "p029_official_cpu_reload": {
            "status": "FC_P029_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION"},
        "comparison_contract": {
            "split": "train", "starts": [0], "windows": 44, "rollout_steps": 100,
            "report_leads": [1, 10, 25, 50, 100], "selection_performed": False,
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, spec


def test_spec_is_exact_and_non_authorizing_status_rejected(tmp_path):
    path, spec = minimal_spec(tmp_path)
    assert MOD.load_spec(path, digest(path)) == spec
    spec["status"] = "P030_TRAIN_HORIZON_DIAGNOSTIC_PREPARATION_ONLY_NOT_APPROVED"
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="not approved"):
        MOD.load_spec(path, digest(path))


@pytest.mark.parametrize(
    "family,case,expected",
    [
        ("base", "matched_start_acquisition_train_b00_m0375", "m0375"),
        ("train8", "dynamic_train8_b04_prbs", "prbs"),
        ("train16", "direct_cfd_directppo2048_v1_env0_ep0002_b00",
         "historical_directppo_exploration"),
    ],
)
def test_action_profiles_are_identity_derived(family, case, expected):
    assert MOD.action_profile(family, case) == expected


class Reader:
    def __getitem__(self, index):
        return {"time": torch.tensor(float(index) / 10)}, {}


class Dataset:
    index = [(0, 0), (0, 20)]
    paths = [Path("case.h5")]

    def __getitem__(self, index):
        assert index == 0
        sample = {
            "state": torch.zeros(3, 2, 3),
            "target_state": torch.zeros(100, 3, 2, 3),
            "omega": torch.zeros(101, 1),
            "target_force": torch.zeros(100, 4),
            "mask": torch.ones(1, 2, 3),
            "time": torch.tensor(999.0),
        }
        return sample, {"case": "case", "step": 0, "rollout_steps": 100, "split": "train"}

    def _reader(self, file_index):
        assert file_index == 0
        return Reader()


def test_start0_uses_all_101_official_reader_timestamps():
    sample, metadata = MOD.sample_start0(Dataset(), 0)
    assert metadata["step"] == 0
    assert tuple(sample["time"].shape) == (1, 101, 1)
    assert sample["time"][0, 0, 0] == 0
    assert sample["time"][0, 100, 0] == 10
    assert sample["state"].shape[0] == 1


def test_history_builder_is_used_for_exact_batch_one():
    calls = []

    def builder(states, mask, actions, nxt):
        calls.append((states.shape, mask.shape, actions.shape, nxt.shape))
        return torch.zeros(6, 2, 3)

    result = MOD.make_inputs_with_reviewed_helper(
        builder, torch.zeros(1, 3, 2, 3), torch.ones(1, 1, 2, 3),
        torch.zeros(1, 1), torch.zeros(1, 1))
    assert result.shape == (1, 6, 2, 3)
    assert calls == [(torch.Size([1, 3, 2, 3]), torch.Size([1, 2, 3]),
                      torch.Size([1]), torch.Size([1]))]


def test_aerodynamic_state_must_match_exactly():
    left, right = torch.nn.Linear(2, 3), torch.nn.Linear(2, 3)
    right.load_state_dict(left.state_dict())
    assert MOD.same_state_dict(left, right)
    with torch.no_grad():
        right.weight[0, 0] += 1
    assert not MOD.same_state_dict(left, right)


def test_bad_source_rejected_before_any_runtime_import(tmp_path, monkeypatch):
    imported = []
    monkeypatch.setattr(MOD, "runtime_dependencies", lambda: imported.append(True))
    spec = {"source_files": [{"path": str(tmp_path / "missing.py"), "sha256": "0" * 64}]}
    with pytest.raises(RuntimeError, match="regular input absent"):
        MOD.validated_runtime_dependencies(spec)
    assert imported == []


def test_exact_1368_original_membership_required():
    class D:
        def __init__(self, prefix, count):
            self.paths = [Path(f"{prefix}{index}.h5") for index in range(count)]
            self.index = []

    datasets = {"base": D("b", 20), "train8": D("t", 8), "train16": D("p", 16)}
    for family, count in (("base", 720), ("train8", 408), ("train16", 240)):
        dataset = datasets[family]
        dataset.index = [(index % len(dataset.paths), index // len(dataset.paths)) for index in range(count)]
    assert len(MOD.validate_original_membership(datasets)) == 1368
    datasets["train16"].index.pop()
    with pytest.raises(RuntimeError, match="1368"):
        MOD.validate_original_membership(datasets)


def test_selection_is_atomically_persisted_and_sha_bound(tmp_path):
    output = tmp_path / "result.json"
    rows = [{"case": "fixed"}]
    selection = {"windows": 44}
    binding = MOD.persist_selection(output, rows, selection, "a" * 64)
    path = tmp_path / "selection.json"
    assert binding == {"path": str(path.resolve()), "sha256": digest(path)}
    assert json.loads(path.read_text()) == {
        "rows": rows, "selection": selection, "source_spec_sha256": "a" * 64}
    with pytest.raises(RuntimeError, match="already exists"):
        MOD.persist_selection(output, rows, selection, "a" * 64)


def test_existing_dashboard_progress_event_schema_is_emitted():
    source = SCRIPT.read_text()
    assert '"event": "origin_complete"' in source
    assert '"count": count' in source


def test_actual_core_groups_all_44_rows_and_raw_records_survive(tmp_path):
    rows, left, right = [], [], []
    for index in range(44):
        dataset_index = 0 if index < 20 else 1 if index < 28 else 2
        identity = {"case": f"case{index:02d}", "start": 0,
                    "dataset_index": dataset_index}
        rows.append({**identity, "family": "base" if index < 20 else
                     "train8" if index < 28 else "train16",
                     "canonical_phase": ("b00", "b02", "b04", "b06")[index % 4],
                     "action_profile": "fixed"})
        record = {
            "identity": identity, "rollout_steps": 100,
            "field_sums_by_lead": [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
                                     for _ in range(100)],
            "predicted_force_physical_by_lead": [[1.0, 0.0, 2.0, 0.1]
                                                   for _ in range(100)],
            "target_force_physical_by_lead": [[1.0, 0.0, 2.0, 0.0]
                                                for _ in range(100)],
        }
        left.append(record)
        right.append(json.loads(json.dumps(record)))
    selection = tmp_path / "selection.json"; selection.write_text("{}")
    raw = MOD.persist_raw_records(
        tmp_path / "result.json", {"k1": left, "p029": right}, rows,
        {"path": str(selection), "sha256": digest(selection)}, "a" * 64)
    assert raw["sha256"] == digest(tmp_path / "raw_records.json")

    def relative(sums):
        error, reference = sums
        ratios = [(e / r) ** .5 if r > 0 else None for e, r in zip(error, reference)]
        return {"field_squared_error_sums_u_v_p": error.tolist(),
                "field_reference_squared_sums_u_v_p": reference.tolist(),
                "field_relative_l2_u_v_p": ratios,
                "velocity_relative_l2": ((error[:2].sum() / reference[:2].sum()) ** .5)}

    summary = CORE.grouped_and_paired(left, right, rows,
                                      relative_field_metrics_fn=relative)
    assert summary["all"]["k1"]["at_lead"]["100"]["count"] == 44
    assert summary["paired_at_lead"]["100"]["summary"]["rear_cl_absolute_error"] == {
        "count": 44, "negative": 0, "zero": 44, "positive": 0, "mean_delta": 0.0}


def test_production_execute_persists_raw_then_passes_rows_to_core():
    tree = ast.parse(SCRIPT.read_text())
    execute = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "execute")
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)]
    persisted = [node for node in calls
                 if isinstance(node.func, ast.Name) and node.func.id == "persist_raw_records"]
    grouped = [node for node in calls
               if isinstance(node.func, ast.Name) and node.func.id == "grouped_and_paired"]
    assert len(persisted) == len(grouped) == 1
    assert persisted[0].lineno < grouped[0].lineno
    assert len(grouped[0].args) >= 3
    assert isinstance(grouped[0].args[2], ast.Name) and grouped[0].args[2].id == "rows"
