import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cfd/tandem_cylinders/predeclare_directppo_train16.py"


def load():
    spec = importlib.util.spec_from_file_location("directppo_train16_predecl", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_case_matrix_excludes_reset_only_episodes():
    module = load()
    payload = module.build(ROOT)
    assert payload["split_contract"] == {"train": 16, "validation": 0, "frozen_test": 0}
    assert payload["excluded_reset_only_episodes"] == [1, 10]
    assert all(row["episode"] in range(2, 10) for row in payload["cases"].values())
    assert all(row["expected_frames"] == 129 for row in payload["cases"].values())
    assert all(len(row["action_points"]) == 129 for row in payload["cases"].values())
    assert payload["validation_or_frozen_accessed"] is False
    assert "not a final-policy on-policy dataset" in payload["scope"]
