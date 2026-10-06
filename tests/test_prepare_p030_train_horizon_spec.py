import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_p030_train_horizon_spec.py"
SPEC = importlib.util.spec_from_file_location("p030_spec_test", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def base():
    return {
        "status": "P029_MATCHED_H10_COMPARISON_EXECUTION_APPROVED",
        "execution_authorized": True,
        "comparison_profile": "p029",
        "candidates": {"1": {"manifest": "/k1", "manifest_sha256": "1" * 64},
                       "4": {"manifest": "/k4", "manifest_sha256": "4" * 64}},
        "p029_candidate": {"manifest": "/p029", "manifest_sha256": "2" * 64},
        "p029_terminal_proof": {"reviewed_by_lead": True, "sha256": "3" * 64},
        "config": {}, "data": {}, "source_phase_mapping": {},
        "train16_predeclaration": {}, "train_audit": {},
    }


def test_pending_spec_keeps_actual_arms_but_never_authorizes(tmp_path):
    source = {"status": MOD.CLOSURE,
              "files_sha256": {"scripts/a.py": "a" * 64}}
    reload_path = tmp_path / "reload.json"; reload_path.write_text("{}")
    reload_receipt = {"status": "FC_P029_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
                      "official_dual_reload_verified": True, "gpu_used": False,
                      "optimizer_created": False, "model_saved": False,
                      "scientific_admission": False}
    result = MOD.prepare(base(), source, tmp_path, reload_receipt, reload_path, "f" * 64)
    assert result["status"] == MOD.PENDING
    assert result["execution_authorized"] is False
    assert result["lead_review"] == {"reviewed": False, "execution_authorized": False}
    assert result["candidates"] == {"k1": base()["candidates"]["1"],
                                     "p029": base()["p029_candidate"]}
    assert result["comparison_contract"]["starts"] == [0]
    assert result["comparison_contract"]["rollout_steps"] == 100


def test_unreviewed_terminal_proof_is_rejected(tmp_path):
    value = base()
    value["p029_terminal_proof"]["reviewed_by_lead"] = False
    with pytest.raises(ValueError, match="not reviewed"):
        MOD.prepare(value, {"status": MOD.CLOSURE, "files_sha256": {"x": "0" * 64}}, tmp_path,
                    {}, tmp_path / "x", "0" * 64)


def test_source_member_escape_is_rejected(tmp_path):
    reload_receipt = {"status": "FC_P029_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION",
                      "official_dual_reload_verified": True, "gpu_used": False,
                      "optimizer_created": False, "model_saved": False,
                      "scientific_admission": False}
    with pytest.raises(ValueError, match="escapes"):
        MOD.prepare(base(), {"status": MOD.CLOSURE,
                             "files_sha256": {"../outside": "0" * 64}}, tmp_path,
                    reload_receipt, tmp_path / "reload", "0" * 64)
