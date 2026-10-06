"""Synthetic metadata/AST contracts; no policy, FNO, dataset or GPU execution."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("diverse_trainer", SCRIPTS / "train_exploratory_diverse_h5_ppo.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
REPO = Path(os.environ.get("PPO_BASE_REPO", "/workspace/fluid_control"))


def test_same_hyperparameters():
    oldspec = importlib.util.spec_from_file_location("original_trainer", REPO / "scripts/train_exploratory_h5_ppo.py")
    old = importlib.util.module_from_spec(oldspec)
    oldspec.loader.exec_module(old)
    extra = {"reset_panel", "reset_count", "reset_order", "reset_selection"}
    assert {k: v for k, v in m.PROTOCOL.items() if k not in extra} == old.PROTOCOL


def test_fixed_panel():
    p = m.fixed_panel()
    assert len(p) == len({(x["case"], x["frame"]) for x in p}) == 24
    assert sum(x["frame"] == 0 for x in p) == 4
    assert all("train" in x["case"] for x in p)
    assert [x["frame"] for x in p[:6]] == [0, 62, 62, 62, 62, 62]


def fixture():
    data = Path("/synthetic/train-root")
    cases = [f"matched_start_acquisition_train_b{p}_zero" for p in ("00", "02", "04", "06")]
    allcases = [f"matched_start_acquisition_train_b{p}_{f}" for p in ("00", "02", "04", "06")
                for f in ("m075", "m0375", "zero", "p0375", "p075")]
    inputs = {k: {"sha256": v} for k, v in {"manifest": m.K1_SHA, "config": m.CONFIG_SHA,
              "normalization": m.NORM_SHA, "baseline": m.BASELINE_SHA}.items()}
    inputs["normalization"]["path"] = str(data / "normalization.json")
    return {"status": m.STATUS, "execution_authorized": True, "protocol": copy.deepcopy(m.PROTOCOL),
            "inputs": inputs, "train_cases": cases, "train_hdf": [{"path": str(data / "train" / f"{c}.h5")} for c in allcases],
            "reset_panel": m.fixed_panel(), "packet_verification": {"path": "x", "sha256": "a" * 64},
            "data_root": str(data), "runtime_packages": {"stable-baselines3": "2.7.1", "gymnasium": "1.2.3"},
            "import_bindings": dict.fromkeys(m.REQUIRED_IMPORTS, "/synthetic/module.py")}


def test_explicit_profile():
    assert m.validate_spec(fixture())


@pytest.mark.parametrize("bad", ["old_status", "unauthorized", "horizon", "missing_case", "order", "validation"])
def test_reject_spec(bad):
    s = fixture()
    if bad == "old_status": s["status"] = "EXPLORATORY_H5_PPO_EXECUTION_APPROVED"
    if bad == "unauthorized": s["execution_authorized"] = False
    if bad == "horizon": s["protocol"]["episode_steps"] = 6
    if bad == "missing_case": s["train_hdf"].pop()
    if bad == "order": s["reset_panel"].reverse()
    if bad == "validation": s["train_hdf"][0]["path"] = "/synthetic/train-root/validation/x.h5"
    with pytest.raises(ValueError): m.validate_spec(s)


def test_actual_receipt_contract(tmp_path):
    p = tmp_path / "receipt.json"
    proof = {"status": "DIVERSE_H5_REAL_RESET_PACKETS_VERIFIED_CPU_NOT_TRAINING", "packets": m.fixed_panel(),
             "packet_count": 24, "scientific_admission": False, "no_model_loaded": True,
             "no_optimizer": True, "no_gpu": True, "no_cfd": True, "source_sha256": {"adapter.py": "a" * 64}}
    p.write_text(json.dumps(proof))
    s = {"packet_verification": {"path": str(p), "sha256": m.sha(p)},
         "import_bindings": {"exploratory_diverse_h5_resets": "/new/adapter.py"},
         "source_files": {"/new/adapter.py": "a" * 64}}
    assert m.read_packet_verification(s) == proof
    s["source_files"]["/new/adapter.py"] = "b" * 64
    with pytest.raises(ValueError): m.read_packet_verification(s)


def calls(path, name):
    return [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == name)
             or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def test_actual_optimizer_and_learning_calls_unchanged():
    old = REPO / "scripts/train_exploratory_h5_ppo.py"
    new = SCRIPTS / "train_exploratory_diverse_h5_ppo.py"
    for name in ("PPO", "learn", "register_step_post_hook", "load_dual_fno", "set_per_process_memory_fraction"):
        assert calls(old, name) == calls(new, name)


def test_supervisor_only_explicit_profile_labels_change():
    old = (REPO / "scripts/supervise_exploratory_h5_ppo.py").read_text()
    new = (SCRIPTS / "supervise_exploratory_diverse_h5_ppo.py").read_text()
    assert old.replace("EXPLORATORY_H5_PPO", "EXPLORATORY_DIVERSE_H5_PPO").replace(
        "EXPLORATORY_H5_SUPERVISED", "EXPLORATORY_DIVERSE_H5_SUPERVISED") == new


def test_actual_caller_wires_packet_comparison_before_environments():
    source = (SCRIPTS / "train_exploratory_diverse_h5_ppo.py").read_text()
    assert source.index("proof = read_packet_verification(spec)") < source.index("import numpy as np")
    assert source.index('same_json(observed_packets, proof["packets"])') < source.index("env = VecNormalize(")
    assert "wrap_phase_cycle(anchor, packets[offset:offset + 6])" in source
    assert '"reset_packets.json"' in source and 'w.reset_counts' in source
