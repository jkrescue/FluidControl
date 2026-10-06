import ast
import hashlib
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/serve_live_research_dashboard.py"


def helper():
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_real_cfd_t228_comparison")
    scope = {"hashlib": hashlib, "json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[node.name]


def test_missing_and_tampered_fail_closed(tmp_path):
    fn = helper()
    assert fn(tmp_path) == {"verified": False}
    base = tmp_path / "artifacts/p064_real_cfd_t228_comparison_20261007"
    base.mkdir(parents=True)
    (base / "figure_manifest.json").write_text('{}')
    assert fn(tmp_path) == {"verified": False}


def test_actual_bound_picture():
    result = helper()(SOURCE.parents[1])
    assert result["verified"] is True
    assert result["model_prediction"] is False
    assert result["includes_e083"] is False
    assert result["time"] == 228


def test_route_and_caption_preserve_historical_prediction():
    text = SOURCE.read_text()
    assert 'parse_qs(parsed.query).get("v") == [evidence["sha256"]]' in text
    assert 'path == "/real-cfd-t228-comparison.png"' in text
    assert 'id="real-cfd-t228"' in text
    assert '瞬时图不用于计算减阻，当前E083尚未包含' in text
    assert 'id="projected-replay-fields"' in text
