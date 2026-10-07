import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

STAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b_driver", STAGE / "scripts/run_p064_b_causal_history_h5_feedback.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
REPO = Path("/workspace/fluid_control")

def test_exact_b_load_and_reject_other_kind():
    seen = {}
    ident = SimpleNamespace(manifest_sha256=m.B_MANIFEST_SHA256, payload={"kind":m.B_KIND})
    def loader(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(flow_model="flow", aerodynamic_model="aero"), ident
    assert m.load_bound_b("manifest", "cfg", "cpu", load_dual_fno=loader, build_model="official") == ("flow", "aero", ident)
    assert seen == {"build_model":"official", "expected_manifest_sha256":m.B_MANIFEST_SHA256}
    ident.payload["kind"] = "FC_P064_ARM_A_CONTROLLED_AERO_FORCE_FNO"
    with pytest.raises(Exception, match="exact retained B"):
        m.load_bound_b("manifest", "cfg", "cpu", load_dual_fno=loader, build_model="official")

def test_transport_and_inner_numeric_callbacks_unchanged():
    old = ast.parse((REPO / "scripts/run_exploratory_causal_history_h5_feedback.py").read_text())
    new = ast.parse(Path(m.__file__).read_text())
    def named(tree, name):
        return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    for name in ("PairSolvers", "build_pair", "observe", "plan", "solve_pair", "forces", "summarize"):
        assert ast.dump(named(old,name), include_attributes=False) == ast.dump(named(new,name), include_attributes=False), name

def test_cpu_and_no_fallback_protocol():
    text = Path(m.__file__).read_text()
    assert 'device = torch.device("cpu")' in text
    assert 'torch.set_float32_matmul_precision("high")' in text
    assert 'original_fail_stop_no_ppo_fallback' in text
    assert m.STEPS == 10 and m.SOURCE_TIME == 148
