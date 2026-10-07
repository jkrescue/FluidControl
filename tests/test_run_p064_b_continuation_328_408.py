from pathlib import Path

import numpy as np
import pytest

from scripts import run_p064_b_continuation_328_408 as run


def e109():
    obs = np.zeros(69, np.float32); obs[-1] = np.float32(.19905773401260382)
    return {"cycles": 800, "rows": [None]*799 + [{"step": 800, "start_time": 327.9,
        "end_time": 328.0, "output_observation": obs.tolist(),
        "applied_omega": .19905773401260382}]}


def test_terminal_seed_keeps_double_limiter_separate():
    obs, previous = run.terminal_seed(e109())
    assert previous == .19905773401260382
    assert float(obs[-1]) == .19905772805213928 and float(obs[-1]) != previous


def test_prospective_windows_cover_complete_tail_without_warmup_drop():
    assert run.WINDOWS[:4] == (("tail_block_1",328.,348.), ("tail_block_2",348.,368.),
                              ("tail_block_3",368.,388.), ("tail_block_4",388.,408.))
    assert ("tail_full_80",328.,408.) in run.WINDOWS
    assert ("joined_full_160",248.,408.) in run.WINDOWS
    assert ("joined_post_transition_140",268.,408.) in run.WINDOWS


def test_fixed_window_is_left_open_and_exact():
    times = np.arange(328., 348.0001, .005)
    data = np.column_stack((times, np.ones_like(times), np.zeros_like(times)))
    selected = run.fixed_window(data, 328., 348.)
    assert len(selected) == 4000 and selected[0,0] > 328. and selected[-1,0] == pytest.approx(348.)


def test_copy_pair_uses_branch_specific_t328(tmp_path):
    class Base:
        @staticmethod
        def tree(path): return {"part": Path(path).name, "role": Path(path).parent.name}
    class Transport:
        @staticmethod
        def substitute(*_): pass
    source, output, inventory = tmp_path/"source", tmp_path/"output", {}
    output.mkdir()
    for role, dirname in (("ppo","case_mpc"),("zero","case_zero")):
        inventory[role] = {}
        for part in ("328","constant","system"):
            root=source/dirname/part; root.mkdir(parents=True); (root/"marker").write_text(role+part)
            inventory[role][part] = {"part":part,"role":dirname}
    cases, _ = run.bound_copy_pair(Base,Transport,source,output,inventory)
    assert (cases["ppo"]/"328/marker").read_text()=="ppo328"
    assert (cases["zero"]/"328/marker").read_text()=="zero328"
