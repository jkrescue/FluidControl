"""Software fixtures: actual CLI mains/metrics, mocked official loading; no admission."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


class Tiny(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(7 * channels).reshape(7, channels).float() / 100000)

    def forward(self, x):
        return torch.einsum("oc,bchw->bohw", self.weight, x)


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    # Fence any future caller default output into this exclusive software fixture.
    monkeypatch.chdir(tmp_path)
    class DM:
        device = torch.device("cpu")
        distributed = cuda = False
        rank = 0
        initialize = staticmethod(lambda: None)

    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("physicsnemo")
    module("physicsnemo.distributed", DistributedManager=DM)
    module("physicsnemo.utils", load_checkpoint=lambda *a, **kw: 1)
    module("train_tandem_fno", build_model=lambda cfg: None,
           configured_force_indices=lambda cfg: (0, 1, 2, 3))
    stage = Path(__file__).resolve().parents[1] / "scripts"
    def load(name):
        spec = importlib.util.spec_from_file_location(name, stage / (name + ".py"))
        value = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, value)
        spec.loader.exec_module(value)
        return value
    ev, dg = load("evaluate_tandem_fno"), load("diagnose_fno_force_window")
    monkeypatch.setattr(ev, "load_composed_config", lambda p: types.SimpleNamespace())
    data = tmp_path / "data"
    (data / "validation").mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps({"max_abs_omega": .75}))
    (data / "normalization.json").write_text(json.dumps({
        "state_mean": [.1, -.2, .3], "state_std": [2., 3., 4.],
        "all_force_channels": ["front_cd", "front_cl", "rear_cd", "rear_cl"],
        "all_force_mean": [1., 0., 1., 0.], "all_force_std": [2., 3., 4., 5.]}))
    t = np.arange(101, dtype=np.float32) / 10
    for name in dg.CASES:
        with h5py.File(data / "validation" / (name + ".h5"), "w") as f:
            f["state"] = np.broadcast_to((1 + t[:, None, None, None] / 100), (101, 3, 4, 4)).copy()
            f["mask"] = np.ones((101, 1, 4, 4), dtype=np.float32)
            f["omega"] = t * .01
            f["force"] = np.stack((1 + t*.01, np.sin(t), 1+t*.02, np.cos(t)), axis=1)
            f["time"] = t
            f["x"] = np.linspace(17, 24, 4)
            f["y"] = np.linspace(5, 10, 4)
    monkeypatch.setattr(dg, "MANIFEST_SHA", dg.sha256(data / "manifest.json"))
    monkeypatch.setattr(dg, "NORMALIZATION_SHA", dg.sha256(data / "normalization.json"))
    import fluid_control.dual_fno as dual
    from p026_history_inference import make_history_dual_fno_adapter
    def run(which, profile, suffix):
        k = 4 if profile == "p026_k4" else 1
        flow, aero = Tiny(6), Tiny(6 if k == 1 else 18)
        network = (dual.make_dual_fno_adapter(flow, aero) if profile == "legacy_k1"
                   else make_history_dual_fno_adapter(flow, aero, k=k))
        role = types.SimpleNamespace(epoch=1, directory=tmp_path.resolve(), model_sha256="a"*64, state_sha256="b"*64)
        identity = types.SimpleNamespace(aerodynamic=role, flow=role,
            manifest_sha256="c"*64, manifest_path=tmp_path / "dual.json",
            payload={"history_input": ev.HISTORY_PROFILES.get(profile)})
        monkeypatch.setattr(dual, "load_dual_fno", lambda *a, **kw: (network, identity))
        monkeypatch.setattr(dual, "validate_dual_runtime_files", lambda *a, **kw: None)
        out = tmp_path / (suffix + ".json")
        args = [which, "--data", str(data), "--normalization-data", str(data),
                "--config", str(tmp_path / "config.yaml"), "--checkpoint-dir", str(tmp_path),
                "--dual-fno-manifest", str(identity.manifest_path),
                "--expected-dual-fno-manifest-sha256", "c"*64,
                "--dual-training-config", str(tmp_path / "training.yaml"),
                "--fno-history-profile", profile, "--output", str(out)]
        if which == "evaluate":
            args += ["--split", "validation", "--horizons", "1", "10", "50", "100", "--segment-stride", "25",
                     "--visualizations-per-horizon", "0", "--visualization-dir", str(tmp_path / "plots")]
        else:
            args += ["--expected-model-sha", "a"*64]
        monkeypatch.setattr(sys, "argv", args)
        (ev if which == "evaluate" else dg).main()
        return json.loads(out.read_text())
    return run, data, ev


@pytest.mark.parametrize("which", ["evaluate", "diagnose"])
def test_actual_main_k1_matches_legacy_metrics(fixture, which):
    run, _, _ = fixture
    old = run(which, "legacy_k1", which + "old")
    new = run(which, "p026_k1", which + "new")
    assert old["cases"] == new["cases"]
    if which == "evaluate":
        assert old["summary"] == new["summary"]
    else:
        assert old["pairs"] == new["pairs"]
        assert new["ppo_authorized"] is False


def test_actual_diagnostic_main_k4_ignores_all_future_truth(fixture):
    run, data, _ = fixture
    old = run("diagnose", "p026_k4", "before")
    for path in (data / "validation").glob("*.h5"):
        with h5py.File(path, "r+") as f:
            f["state"][1:] = 12345
    new = run("diagnose", "p026_k4", "poison")
    for a, b in zip(old["cases"], new["cases"]):
        assert a["predicted_forces"] == b["predicted_forces"]
        assert a["prediction_window"] == b["prediction_window"]
        assert a["field_diagnostics"] != b["field_diagnostics"]


@pytest.mark.parametrize("start", [0, 2, 5])
def test_each_start_independently_excludes_every_future_observation(fixture, start):
    _, _, ev = fixture
    from p026_history_inference import trajectory_history, make_history_dual_fno_adapter
    states = torch.arange(15*3*4*4).reshape(15, 3, 4, 4).float()/100
    actions = torch.linspace(-.2, .2, 15)
    def rollout(values):
        net = make_history_dual_fno_adapter(Tiny(6), Tiny(18), k=4)
        history = trajectory_history(values, actions, torch.tensor([start]), k=4)
        outputs = []
        for offset in range(4):
            raw, predicted, history = ev.history_dual_step(net, history, torch.ones(1,1,4,4), actions[[start+offset+1]])
            outputs.append((raw.detach(), predicted.detach()))
        return outputs
    expected = rollout(states)
    poison = states.clone()
    poison[start+1:] = 1e6
    for a, b in zip(expected, rollout(poison)):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
