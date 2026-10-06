import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_p030_train_horizon.py"
SPEC = importlib.util.spec_from_file_location("p030_driver_test", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


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
