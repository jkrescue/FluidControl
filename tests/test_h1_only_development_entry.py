import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/evaluate_p064_h1_only_development_h1_h5.py"
KIND = "FC_P064_H1_ONLY_K1_FRESH_FORCE_FNO"


def load_pair_function():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "load_evaluation_pair"
    )
    namespace = {
        "json": json,
        "require": lambda ok, reason: ok
        or (_ for _ in ()).throw(ValueError(reason)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["load_evaluation_pair"]


def test_h1_only_kind_routes_to_exact_loader(tmp_path):
    manifest = tmp_path / "dual_model_manifest.json"
    manifest.write_text(json.dumps({"kind": KIND}))
    calls = []

    def loader(path, cfg, device, *, build_model, expected_manifest_sha256):
        calls.append((path, cfg, device, build_model, expected_manifest_sha256))
        adapter = SimpleNamespace(flow_model="flow", aerodynamic_model="aero")
        return adapter, SimpleNamespace(payload={"kind": KIND})

    spec = {
        "candidate_label": "F_H1_ONLY",
        "inputs": {"manifest": {"sha256": "e" * 64}},
    }
    result = load_pair_function()(spec, manifest, "cfg", "cuda", loader, "build", None)
    assert result[:2] == ("flow", "aero")
    assert calls == [(manifest, "cfg", "cuda", "build", "e" * 64)]


@pytest.mark.parametrize("label", ["K1", "A", "B", "C", "D", "E"])
def test_old_or_ambiguous_candidate_labels_are_rejected(tmp_path, label):
    manifest = tmp_path / "dual_model_manifest.json"
    manifest.write_text(json.dumps({"kind": KIND}))
    with pytest.raises(ValueError, match="H1-only candidate only"):
        load_pair_function()(
            {"candidate_label": label, "inputs": {"manifest": {"sha256": "e" * 64}}},
            manifest,
            None,
            None,
            None,
            None,
            None,
        )


def test_wrong_manifest_kind_is_rejected_before_model_load(tmp_path):
    manifest = tmp_path / "dual_model_manifest.json"
    manifest.write_text(json.dumps({"kind": "FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO"}))
    called = []

    def loader(*args, **kwargs):
        called.append(True)
        raise AssertionError("must not load")

    with pytest.raises(ValueError, match="explicit P064 arm kind"):
        load_pair_function()(
            {
                "candidate_label": "F_H1_ONLY",
                "inputs": {"manifest": {"sha256": "e" * 64}},
            },
            manifest,
            None,
            None,
            loader,
            None,
            None,
        )
    assert not called


def test_numerical_rollout_and_fixed_panel_code_are_unchanged():
    old = Path(
        "/workspace/fluid_control/artifacts/"
        "p064_b00_b02_coverage_d_source_20261007_immutable/scripts/"
        "evaluate_p064_d_development_h1_h5.py"
    ).read_text()
    new = SOURCE.read_text()

    def functions(text):
        return {
            node.name: ast.dump(node)
            for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef)
        }

    before, after = functions(old), functions(new)
    for name in (
        "conversion_records",
        "rollout",
        "tensor_digest",
        "worker",
        "supervise",
        "verify_execution",
    ):
        assert before[name] == after[name]
    assert "enumerate(('b01','b03'))" in new
    assert "'endpoints':80" in new
