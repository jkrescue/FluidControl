import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import train_fcp026_history as p
import p026_state_history as history


def test_protocol_matched_except_explicit_history():
    a, b = p.protocol(1), p.protocol(4)
    assert a.pop("history_input")["aerodynamic_input_channels"] == 6
    assert b.pop("history_input")["aerodynamic_input_channels"] == 18
    assert a == b
    assert (
        a["optimizer_steps"] == 171
        and a["training_windows"] == 1368
        and a["learning_rate"] == 1.5625e-7
    )


def test_real_inventory_all_windows_preserved():
    children = []
    for count, frames, stride in [(20, 801, 20), (8, 201, 2), (16, 129, 2)]:
        children.append(
            SimpleNamespace(
                index=[
                    (i, start)
                    for i in range(count)
                    for start in range(0, frames - 100, stride)
                ]
            )
        )
    dataset = SimpleNamespace(_datasets=children)
    assert p.inventory(dataset) == p.protocol(4)["inventory"]
    children[0].index.pop()
    with pytest.raises(ValueError):
        p.inventory(dataset)


@pytest.mark.parametrize(
    "start,expected", [(0, [0, 0, 0]), (1, [0, 0, 0]), (2, [0, 0, 1]), (4, [1, 2, 3])]
)
def test_preceding_only_reads_past_preserves_batch(start, expected):
    calls = []

    class Reader:
        def __getitem__(self, i):
            calls.append(i)
            return (
                dict(
                    state=torch.full((3, 1, 1), float(i)),
                    mask=torch.ones(1, 1, 1),
                    omega=torch.tensor([float(i)]),
                ),
                {},
            )

    child = SimpleNamespace(
        paths=[Path("case.h5")],
        state_mean=torch.zeros(3, 1, 1),
        state_std=torch.ones(3, 1, 1),
        action_scale=1.0,
        _reader=lambda _: Reader(),
    )
    dataset = SimpleNamespace(_datasets=[child])
    ident = dict(dataset_index=0, case="case", start=start, split="train")
    sample = dict(
        state=torch.full((3, 1, 1), float(start)),
        mask=torch.ones(1, 1, 1),
        omega=torch.tensor([[float(start)], [99.0]]),
        target_state=torch.full((100, 3, 1, 1), 999.0),
    )
    before = copy.deepcopy(sample)
    states, actions, meta = p.preceding(dataset, ident, sample, 4, history)
    assert (
        states[0, :, 0, 0, 0].tolist() == expected
        and actions.flatten().tolist() == expected
    )
    assert all(i < start for i in calls)
    assert all(torch.equal(sample[key], before[key]) for key in sample)
    one = p.preceding(dataset, ident, sample, 1, history)
    assert one[0].shape == (1, 0, 3, 1, 1)


def test_parameter_partition_and_displacement():
    lift = torch.ones(24, 20, 1, 1)
    result = p.split_norm({"lift": lift, "other": torch.tensor([3.0])}, 4, "lift")
    assert result["new_history_l2"] == pytest.approx(288**0.5)
    assert result["inherited_l2"] == pytest.approx((192 + 9) ** 0.5)
    assert p.split_norm({"lift": lift}, 1, "lift")["new_history_l2"] == 0


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(1)) for _ in range(28)]
        )


def test_fresh_exact_adam_states_steps_and_finite():
    m = Model()
    o = torch.optim.AdamW(
        m.parameters(), lr=1.5625e-7, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
    )
    p.check_optimizer(m, o, 0)
    sum(x.square().sum() for x in m.parameters()).backward()
    o.step()
    p.check_optimizer(m, o, 1)
    with pytest.raises(ValueError):
        p.check_optimizer(m, o, 171)
    next(iter(o.state.values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        p.check_optimizer(m, o, 1)


def test_eight_grad_average_matches_independent_reference():
    path = Path(
        "/workspace/fluid_control/scripts/train_fcp015_window_accumulation.py"
    )
    assert p.sha(path) == p.P015_SHA
    spec = importlib.util.spec_from_file_location("test_accumulation", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    m = Model()
    other = copy.deepcopy(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5625e-7)
    ref = torch.optim.AdamW(other.parameters(), lr=1.5625e-7)

    def run(i):
        loss = sum(x.sum() * (i + 1) for x in m.parameters())
        loss.backward()
        return dict(h1_balanced=1.0, ar_balanced=1.0, total=1.0)

    helper.accumulation_step(m, opt, range(8), run, lambda _: {})
    for x in other.parameters():
        x.grad = torch.full_like(x, 4.5)
    torch.nn.utils.clip_grad_norm_(list(other.parameters()), 1.0)
    ref.step()
    for a, b in zip(m.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_parent_roles_cannot_share_generic_directory(tmp_path):
    args = SimpleNamespace(flow_parent=tmp_path, aerodynamic_parent=tmp_path)
    with pytest.raises(ValueError, match="separate"):
        p.validate_parent_paths(args, {})


def test_exact_parent_filename_and_hash_contract(tmp_path):
    directories = {role: tmp_path / role for role in ("flow", "aerodynamic")}
    manifest = {}
    for role, directory in directories.items():
        directory.mkdir()
        epoch = 0 if role == "flow" else 1
        model = directory / f"FNO.0.{epoch}.mdlus"
        model.write_bytes(role.encode())
        state = directory / f"checkpoint.0.{epoch}.pt"
        state.write_bytes(b"state")
        manifest[role] = dict(
            model_file=model.name,
            state_file=state.name,
            model_sha256=p.sha(model),
            state_sha256=p.sha(state),
        )
    args = SimpleNamespace(
        flow_parent=directories["flow"], aerodynamic_parent=directories["aerodynamic"]
    )
    p.validate_parent_paths(args, manifest)
    (directories["flow"] / "FNO.0.1.mdlus").write_bytes(b"wrong")
    with pytest.raises(ValueError):
        p.validate_parent_paths(args, manifest)


def test_protocol_serialized_bytes_have_recorded_digest():
    import json, hashlib

    data = p.protocol(4)
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == p.canonical_sha(data)


def test_no_fixture_status_in_candidate_and_separate_architectures():
    import inspect

    source = inspect.getsource(p.execute)
    assert "ENGINEERING_FIXTURE" not in source
    assert (
        'manifest["flow_architecture"]' in source
        and 'manifest["aerodynamic_architecture"]' in source
    )
    assert "official frozen-flow fresh reload differs" in source


def test_warm_padded_summaries_keep_matching_denominators():
    rows = []
    for index, start, value in [
        (160, 320, 1.0),
        (816, 90, 2.0),
        (923, 100, 3.0),
        (975, 0, 4.0),
        (1077, 0, 5.0),
        (1233, 0, 6.0),
    ]:
        stats = {
            key: value
            for key in (
                "bias_mse",
                "rms_error_mse",
                "absolute_rms_error",
                "centered_residual_mse",
            )
        }
        rows.append(
            dict(
                global_index=index,
                identity=dict(start=start),
                panel=dict(
                    objective=dict(h1_balanced=value, ar_balanced=value),
                    domains=dict(h1=stats, ar=stats),
                ),
            )
        )
    result = p.grouped_panel(rows)
    assert result["warm"]["window_count"] == 3 and result["warm"]["nonzero_count"] == 2
    assert (
        result["padded"]["window_count"] == 3 and result["padded"]["nonzero_count"] == 3
    )
    assert result["warm"]["domains"]["h1"]["mean_original_objective"] == 2.0
    assert result["warm"]["domains"]["ar"]["nonzero_statistics"]["bias_mse"] == 2.5
    assert (
        result["padded"]["domains"]["h1"]["nonzero_statistics"]["centered_residual_mse"]
        == 5.0
    )
