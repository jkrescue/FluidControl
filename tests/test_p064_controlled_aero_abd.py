import importlib.util
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch


class DatasetBase:
    def __init__(self, num_workers=0):
        self.num_workers = num_workers

    def __getitem__(self, index):
        return self._load(index)

    def close(self):
        pass


physicsnemo = types.ModuleType("physicsnemo")
datapipes = types.ModuleType("physicsnemo.datapipes")
datapipes.DatasetBase = DatasetBase
physicsnemo.datapipes = datapipes
sys.modules.setdefault("physicsnemo", physicsnemo)
sys.modules.setdefault("physicsnemo.datapipes", datapipes)


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


new = load(ROOT / "src/fluid_control/p064_controlled_aero_abd.py", "p064_abd")
old = load(ROOT / "reference/p064_controlled_aero_abc.py", "p064_abc")
runner = load(ROOT / "scripts/train_fcp064_controlled_aero_abd.py", "p064_runner_d")
old_runner = load(ROOT / "reference/train_fcp064_controlled_aero_abc.py", "p064_runner_abc")
loader = load(ROOT / "src/fluid_control/dual_fno.py", "p064_loader_d")
old_loader = load(ROOT / "reference/dual_fno.py", "p064_loader_abc")
evaluator = load(ROOT / "scripts/evaluate_p064_d_development_h1_h5.py", "p064_eval_d")
ORDER = json.loads((ROOT / "tests/parent_order.json").read_text())


def test_old_arms_are_byte_equivalent_schedules():
    for arm in "ABC":
        assert [row.__dict__ for row in new.compile_schedule(ORDER, arm)] == [
            row.__dict__ for row in old.compile_schedule(ORDER, arm)
        ]
        assert new.schedule_sha256(new.compile_schedule(ORDER, arm)) == old.schedule_sha256(
            old.compile_schedule(ORDER, arm)
        )


def test_d_fixed_replacement_contract():
    b = new.compile_schedule(ORDER, "B")
    d = new.compile_schedule(ORDER, "D")
    assert len(d) == 256
    assert sum(row.source == "original44" for row in d) == 192
    assert sum(row.source == "controlled_b00" for row in d) == 32
    assert sum(row.source == "controlled_b02" for row in d) == 32
    for update in range(32):
        block = d[8 * update : 8 * (update + 1)]
        assert block[0].source == "controlled_b00"
        assert block[4].source == "controlled_b02"
        assert all(block[j].source == "original44" for j in (1, 2, 3, 5, 6, 7))
    assert [row.original_global_index for row in d] == [row.original_global_index for row in b]
    starts_b00 = [row.b00_start for row in d if row.source == "controlled_b00"]
    starts_b02 = [row.b00_start for row in d if row.source == "controlled_b02"]
    assert starts_b00 == starts_b02 == list(new.evenly_spaced_indices(701, 32))
    assert starts_b00[0] == 0 and starts_b00[-1] == 700
    b_starts = {row.b00_start for row in b if row.b00_start is not None}
    assert len(set(starts_b00) & b_starts) == 4


class Fake:
    rollout_steps = 100
    force_indices = (0, 1, 2, 3)
    action_scale = 0.75
    state_mean = torch.tensor([1.0])
    state_std = torch.tensor([2.0])
    force_mean = torch.tensor([3.0])
    force_std = torch.tensor([4.0])

    def __init__(self, dataset_index, length=701):
        self.dataset_index = dataset_index
        self.index = [(0, start) for start in range(length)]
        self.closed = False

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return {"value": torch.tensor(index)}, {
            "case": f"case{self.dataset_index}", "start": index,
            "dataset_index": self.dataset_index, "split": "train", "rollout_steps": 100,
        }

    def close(self):
        self.closed = True


class Original(Fake):
    def __init__(self):
        super().__init__(0, 1368)
        self._datasets = [Fake(0), Fake(1), Fake(2)]


def test_d_routes_distinct_controlled_sources_and_identity():
    original, reference = Original(), Fake(0)
    b00, b02 = Fake(3), Fake(4)
    dataset = new.ScheduledABCDataset(
        original, reference, ORDER, "D", b00_dataset=b00, b02_dataset=b02
    )
    _, m0 = dataset[0]
    _, m4 = dataset[4]
    _, m1 = dataset[1]
    def collate(metadata):
        return {key: [value] for key, value in metadata.items()}

    assert new.scheduled_training_identity(collate(m0))["dataset_index"] == 3
    assert new.scheduled_training_identity(collate(m4))["dataset_index"] == 4
    assert m0["ab_source"] == "controlled_b00" and m0["ab_b00_start"] == 0
    assert m4["ab_source"] == "controlled_b02" and m4["ab_b02_start"] == 0
    assert m1["ab_source"] == "original44"
    dataset.close()
    assert original.closed and b00.closed and b02.closed


def test_d_rejects_missing_or_malformed_b02():
    with pytest.raises(ValueError, match="requires b02"):
        new.ScheduledABCDataset(Original(), Fake(0), ORDER, "D", b00_dataset=Fake(3))
    malformed = Fake(4, 700)
    with pytest.raises(ValueError, match="b02 must be one"):
        new.ScheduledABCDataset(
            Original(), Fake(0), ORDER, "D", b00_dataset=Fake(3), b02_dataset=malformed
        )


def test_protocol_records_predeclared_d_contract():
    summary = new.protocol_summary(ORDER)
    assert summary["training_windows_per_arm"] == 256
    assert summary["optimizer_steps_per_arm"] == 32
    assert summary["b00_windows_arm_d"] == summary["b02_windows_arm_d"] == 32
    assert summary["replacement_within_each_update_arm_d"] == [0, 4]
    assert sum(summary["arm_d_original_family_windows"].values()) == 192
    assert summary["selection_performed"] is False
    assert summary["gpu_execution_authorized"] is False


def test_runner_protocol_preserves_abc_and_declares_d_only_delta():
    for arm in "ABC":
        assert runner.protocol(arm, new, ORDER) == old_runner.protocol(arm, old, ORDER)
    d = runner.protocol("D", new, ORDER)
    assert d["training_windows"] == 256 and d["optimizer_steps"] == 32
    assert d["b00_windows"] == d["b02_windows"] == 32
    assert d["b00_weight"] == d["b02_weight"] == 0.125
    assert d["replacement_within_each_update"] == [0, 4]
    assert d["controlled_source_profile"] == "32_b00_k1_projected_plus_32_b02_canonical_policy"
    assert d["validation_accessed"] is d["frozen_test_accessed"] is False


def test_loader_old_contracts_unchanged_and_d_is_explicit():
    for arm in "ABC":
        kind = old_loader.P064_SYSTEM_KIND[arm]
        assert loader._experiment_contract(kind) == old_loader._experiment_contract(kind)
    contract = loader._experiment_contract(loader.P064_SYSTEM_KIND["D"])
    assert contract["status"] == "FC_P064_ARM_D_DUAL_FNO_MANIFEST_VERIFIED"
    assert contract["aero_kind"] == "FC_P064_ARM_D_CONTROLLED_AERO_CHECKPOINT"
    assert contract["p064_arm"] == "D"
    assert contract["optimizer_steps"] == 32
    assert contract["flow_frozen"] is True and contract["aero_frozen"] is False


def test_loader_validates_exact_d_protocol_and_rejects_wrong_mix(tmp_path):
    protocol = runner.protocol("D", new, ORDER)
    protocol_path = tmp_path / "training_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2))
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    aero_arch = dict(loader.ARCHITECTURE)
    aero_arch["in_channels"] = 6
    payload = {
        "history_input": loader._p026_history_input(1),
        "parent_history_inventory": loader._p026_inventory(),
        "history_state_module_sha256": loader.P026_HISTORY_STATE_SHA256,
        "history_inference_module_sha256": loader.P026_HISTORY_INFERENCE_SHA256,
        "flow_architecture": loader.ARCHITECTURE,
        "aerodynamic_architecture": aero_arch,
        "training_protocol_file": "training_protocol.json",
        "training_protocol_sha256": protocol_sha,
        "training_semantics": protocol,
    }
    loader._validate_p026_protocol(
        tmp_path, payload, {"p026_history_k": 1, "p064_arm": "D"}
    )
    bad = json.loads(json.dumps(protocol))
    bad["b02_windows"] = 31
    protocol_path.write_text(json.dumps(bad, indent=2))
    payload["training_protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    payload["training_semantics"] = bad
    with pytest.raises(ValueError, match="protocol"):
        loader._validate_p026_protocol(
            tmp_path, payload, {"p026_history_k": 1, "p064_arm": "D"}
        )


def test_development_entry_routes_explicit_d_kind(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"kind": loader.P064_SYSTEM_KIND["D"]}))

    class Identity:
        payload = {"kind": loader.P064_SYSTEM_KIND["D"]}

    def fake_load(path, cfg, device, *, build_model, expected_manifest_sha256):
        assert path == manifest and expected_manifest_sha256 == "future-d-sha"
        return types.SimpleNamespace(flow_model="flow", aerodynamic_model="aero"), Identity()

    models = evaluator.load_evaluation_pair(
        {"candidate_label": "D", "inputs": {"manifest": {"sha256": "future-d-sha"}}},
        manifest, object(), "cpu", fake_load, object(), object()
    )
    assert models[:2] == ("flow", "aero")
    manifest.write_text(json.dumps({"kind": loader.P064_SYSTEM_KIND["C"]}))
    with pytest.raises(ValueError, match="explicit P064 arm kind"):
        evaluator.load_evaluation_pair(
            {"candidate_label": "D", "inputs": {"manifest": {"sha256": "future-d-sha"}}},
            manifest, object(), "cpu", fake_load, object(), object()
        )


def test_runner_validates_future_b02_train_view_contract(tmp_path):
    root = tmp_path / "b02_view"
    train = root / "train"
    train.mkdir(parents=True)
    hdf = train / "b02.h5"
    hdf.write_bytes(b"synthetic-engineering-fixture")
    hdf_sha = hashlib.sha256(hdf.read_bytes()).hexdigest()
    normalization = root / "normalization.json"
    normalization.write_text("{}")
    norm_sha = hashlib.sha256(normalization.read_bytes()).hexdigest()
    receipt = tmp_path / "result.json"
    receipt_data = {
        "status": "B02_CONTROLLED_TRAIN_HDF_COMPLETE_NOT_TRAINING",
        "frames": 801, "trajectories": 1, "split": "train",
        "normalization_sha256": norm_sha, "normalization_refit": False,
        "official_reader_verified": True, "source_unchanged": True,
        "model_loaded": False, "cfd_executed": False, "optimizer_steps": 0,
        "scientific_admission": False, "owned_containers_cleaned": True,
        "hdf": {"path": str(hdf), "sha256": hdf_sha},
    }
    receipt.write_text(json.dumps(receipt_data))
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "max_abs_omega": 0.75, "conversion_receipt_sha256": receipt_sha,
        "validation_accessed": False, "frozen_test_accessed": False,
        "normalization_sha256": norm_sha, "hdf_sha256": {hdf.name: hdf_sha},
    }))
    args = types.SimpleNamespace(
        b02_data_root=root, b02_manifest=manifest,
        b02_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        b02_conversion_receipt=receipt,
        b02_conversion_receipt_sha256=receipt_sha, b02_source_hdf=hdf,
    )
    assert runner.validate_b02_view(args)["normalization_sha256"] == norm_sha
    receipt_data["official_reader_verified"] = False
    receipt.write_text(json.dumps(receipt_data))
    args.b02_conversion_receipt_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()
    # Keep the manifest internally bound so the failure exercises the receipt gate.
    payload = json.loads(manifest.read_text())
    payload["conversion_receipt_sha256"] = args.b02_conversion_receipt_sha256
    manifest.write_text(json.dumps(payload))
    args.b02_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="official-reader contract"):
        runner.validate_b02_view(args)
