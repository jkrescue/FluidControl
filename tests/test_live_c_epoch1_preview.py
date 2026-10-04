import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"
spec = importlib.util.spec_from_file_location("dashboard_preview", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root, final=False):
    base = root / (module.C_FINAL_PREVIEW if final else module.C_EPOCH1_PREVIEW)
    base.mkdir(parents=True)
    receipt = {
        "status": ("FCP003C_FINAL_CANDIDATE_FLOW_VISUALIZATION_COMPLETE" if final
                   else "FCP003C_INTERIM_EPOCH1_FLOW_VISUALIZATION_PREVIEW_COMPLETE"),
        "interim_epoch": 1, "checkpoint_epoch": 2,
        "checkpoint_sha256": module.C_FINAL_SHA if final else module.C_EPOCH1_SHA,
        "formal_gate_modified": False, "ppo_launched": False,
        "training_performed": False, "frozen_test_accessed": False,
        "output_sha256": {},
    }
    names = ["evaluation.json", "segments.json"] + [
        f"figures/full40_dynamic_validation_b01_{profile}/horizon_{h}_start_0000.png"
        for profile in (("zero", "minus", "plus") if final else ("plus",))
        for h in ("001", "010", "050", "100")
    ]
    for name in names:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture, not scientific evidence")
        receipt["output_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (base / "receipt.json").write_text(json.dumps(receipt))
    return base, receipt


def test_absent_is_not_ready(tmp_path):
    assert module._c_epoch1_preview(tmp_path) == {"ready": False}


def test_complete_bound_preview(tmp_path):
    fixture(tmp_path)
    assert module._c_epoch1_preview(tmp_path)["ready"]


def test_changed_plot_rejected(tmp_path):
    base, _ = fixture(tmp_path)
    next(base.rglob("*.png")).write_bytes(b"modified")
    assert not module._c_epoch1_preview(tmp_path)["ready"]


def test_wrong_checkpoint_rejected(tmp_path):
    base, receipt = fixture(tmp_path)
    receipt["checkpoint_sha256"] = "0" * 64
    (base / "receipt.json").write_text(json.dumps(receipt))
    assert not module._c_epoch1_preview(tmp_path)["ready"]


def test_missing_safety_flag_rejected(tmp_path):
    base, receipt = fixture(tmp_path)
    del receipt["frozen_test_accessed"]
    (base / "receipt.json").write_text(json.dumps(receipt))
    assert not module._c_epoch1_preview(tmp_path)["ready"]


def test_plot_escape_rejected(tmp_path):
    base, _ = fixture(tmp_path)
    path = next(base.rglob("*.png"))
    outside = tmp_path / "outside.png"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    assert not module._c_epoch1_preview(tmp_path)["ready"]


def test_final_requires_all_three_profiles(tmp_path):
    base, _ = fixture(tmp_path, final=True)
    assert module._c_final_preview(tmp_path)["ready"]
    (base / "figures/full40_dynamic_validation_b01_zero/horizon_100_start_0000.png").unlink()
    assert not module._c_final_preview(tmp_path)["ready"]


def test_interim_does_not_authorize_final(tmp_path):
    fixture(tmp_path)
    assert not module._c_final_preview(tmp_path)["ready"]


def test_final_cannot_use_epoch1_identity(tmp_path):
    base, receipt = fixture(tmp_path, final=True)
    receipt["checkpoint_sha256"] = module.C_EPOCH1_SHA
    (base / "receipt.json").write_text(json.dumps(receipt))
    assert not module._c_final_preview(tmp_path)["ready"]


def test_final_changed_metrics_rejected(tmp_path):
    base, _ = fixture(tmp_path, final=True)
    (base / "evaluation.json").write_text('{}')
    assert not module._c_final_preview(tmp_path)["ready"]
