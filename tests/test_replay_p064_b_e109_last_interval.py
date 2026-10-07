from pathlib import Path

import numpy as np
import pytest

from scripts import replay_p064_b_e109_last_interval as replay


def field(path: Path, values, vector=False):
    if vector:
        payload = "\n".join(f"({a} {b} {c})" for a, b, c in values)
    else:
        payload = "\n".join(str(x) for x in values)
    path.write_text(f"FoamFile {{ location wrong; }}\ninternalField nonuniform List<scalar> 2(\n{payload}\n);\nboundaryField {{}}")


def test_internal_comparison_ignores_header_but_rejects_state_change(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    for name in ("U", "U_0", "p", "phi", "phi_0"):
        field(left / name, [1.0, 2.0])
        field(right / name, [1.0, 2.0])
    assert all(row["max_abs_difference"] == 0 for row in replay.compare_restart_numeric(left, right).values())
    field(right / "p", [1.0, 2.001])
    with pytest.raises(ValueError, match="numeric replay"):
        replay.compare_restart_numeric(left, right)


def test_replay_status_is_not_extension():
    assert replay.STATUS.endswith("RECOVERY_REPLAY_EXECUTION_APPROVED")


def test_execute_copy_seam_reads_and_enforces_inventory(tmp_path):
    class Base:
        @staticmethod
        def tree(path):
            return {"path": str(Path(path).name)}
    class Transport:
        calls = []
        @classmethod
        def substitute(cls, path, key, value):
            cls.calls.append((Path(path), key, value))
    source, output = tmp_path / "source", tmp_path / "output"
    output.mkdir()
    inventory = {}
    for role, dirname in (("ppo", "case_mpc"), ("zero", "case_zero")):
        expected = {}
        for part in ("327.9", "328", "constant", "system"):
            root = source / dirname / part
            root.mkdir(parents=True)
            (root / "marker").write_text(part)
            expected[part] = {"path": part}
        inventory[role] = expected
    cases, trees = replay.copy_bound_cases(Base, Transport, source, output, inventory)
    assert set(cases) == {"ppo", "zero"} and trees == inventory
    assert (cases["ppo"] / "327.9/marker").read_text() == "327.9"
    assert not (cases["ppo"] / "328").exists()
    assert len(Transport.calls) == 2
    wrong = {**inventory, "ppo": {**inventory["ppo"], "328": {"path": "changed"}}}
    second = tmp_path / "second"; second.mkdir()
    with pytest.raises(ValueError, match="bound E109"):
        replay.copy_bound_cases(Base, Transport, source, second, wrong)
