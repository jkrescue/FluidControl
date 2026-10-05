from pathlib import Path
import pytest
import torch
import p026_state_history as h


class Reader:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __getitem__(self, i):
        self.calls.append(i)
        return self.frames[i], {}


class Base:
    def __init__(self, start):
        self.index = [(0, start)]
        self.split = "train"
        self.paths = [Path("train/case.h5")]
        self.state_mean = torch.tensor([1.0, 2.0, 3.0])[:, None, None]
        self.state_std = torch.tensor([2.0, 3.0, 4.0])[:, None, None]
        self.action_scale = 2.0
        self.reader = Reader(
            [
                dict(
                    state=torch.full((3, 2, 2), float(i + 10)),
                    mask=torch.ones(1, 2, 2),
                    omega=torch.tensor([i + 0.25]),
                )
                for i in range(10)
            ]
        )
        frame = self.reader.frames[start]
        self.sample = dict(
            state=(frame["state"] - self.state_mean) / self.state_std,
            mask=frame["mask"],
            omega=torch.tensor([[start + 0.25], [start + 1.25]]) / 2,
            target_state=torch.full((100, 3, 2, 2), 999.0),
            target_force=torch.full((100, 4), 888.0),
        )
        self.meta = dict(case="case", step=start, split="train", rollout_steps=100)

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.sample, self.meta

    def _reader(self, i):
        assert i == 0
        return self.reader


@pytest.mark.parametrize(
    "start,indices,padding",
    [
        (0, [0, 0, 0, 0], 3),
        (1, [0, 0, 0, 1], 2),
        (2, [0, 0, 1, 2], 1),
        (3, [0, 1, 2, 3], 0),
        (6, [3, 4, 5, 6], 0),
    ],
)
def test_exact_past_reader_padding_and_original_targets(start, indices, padding):
    base = Base(start)
    sample, meta, history = h.HistoryWindowAdapter(base, 4)[0]
    assert sample is base.sample and meta is base.meta
    assert torch.all(sample["target_state"] == 999) and torch.all(
        sample["target_force"] == 888
    )
    assert (
        history["metadata"]["frame_indices"] == indices
        and history["metadata"]["padded_frames"] == padding
    )
    assert base.reader.calls == sorted(set(indices) - {start})
    assert all(i < start for i in base.reader.calls)
    for i, frame in enumerate(indices):
        torch.testing.assert_close(
            history["states"][i],
            (base.reader.frames[frame]["state"] - base.state_mean) / base.state_std,
        )
        assert history["actions"][i].item() == (frame + 0.25) / 2
    assert history["metadata"]["action_semantics"] == h.ACTION_SEMANTICS


def test_k1_exact_existing_layout_shared_action_semantics():
    base = Base(3)
    sample, _, history = h.HistoryWindowAdapter(base, 1)[0]
    actual = h.build_input(
        history["states"], sample["mask"], history["actions"], sample["omega"][1]
    )
    expected = torch.cat(
        (
            sample["state"],
            sample["mask"],
            sample["omega"][0, :, None, None].expand(1, 2, 2),
            sample["omega"][1, :, None, None].expand(1, 2, 2),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert base.reader.calls == []


def test_k4_layout_and_selected_next_action_only():
    base = Base(4)
    sample, _, history = h.HistoryWindowAdapter(base, 4)[0]
    actual = h.build_input(history["states"], sample["mask"], history["actions"], 0.9)
    assert actual.shape == (18, 2, 2)
    assert torch.equal(actual[9:12], sample["state"]) and torch.equal(
        actual[12:13], sample["mask"]
    )
    assert torch.all(actual[16] == sample["omega"][0].item()) and torch.all(
        actual[17] == torch.tensor(0.9)
    )
    sample["target_state"].fill_(-999)
    sample["target_force"].fill_(-888)
    assert torch.equal(
        actual,
        h.build_input(history["states"], sample["mask"], history["actions"], 0.9),
    )


def test_autoregressive_shift_keeps_graph_and_uses_prediction():
    states = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    actions = torch.arange(4, dtype=torch.float32)[:, None]
    predicted = torch.full((3, 2, 2), 7.0, requires_grad=True)
    shifted, a = h.shift_history(states, actions, predicted, 8.0)
    assert torch.equal(shifted[:-1], states[1:]) and torch.equal(shifted[-1], predicted)
    assert a.flatten().tolist() == [1.0, 2.0, 3.0, 8.0]
    shifted.sum().backward()
    assert torch.equal(predicted.grad, torch.ones_like(predicted))
    assert states.flatten()[0] == 0 and actions[0] == 0


@pytest.mark.parametrize("k", [1, 4])
def test_warmstart_preserves_old_affine_output_and_coordinates(k):
    torch.manual_seed(3)
    old = {h.LIFT: torch.randn(24, 8, 1, 1), "bias": torch.randn(24)}
    template = {
        h.LIFT: torch.empty(24, 8 if k == 1 else 20, 1, 1),
        "bias": torch.empty(24),
    }
    mapped = h.history_warmstart_state(old, template, k)
    states = torch.randn(k, 3, 2, 2)
    mask = torch.ones(1, 2, 2)
    actions = torch.randn(k, 1)
    next_action = torch.tensor(0.4)
    coords = torch.randn(2, 2, 2)
    newx = torch.cat((h.build_input(states, mask, actions, next_action), coords))[None]
    oldx = torch.cat(
        (
            states[-1],
            mask,
            actions[-1, :, None, None].expand(1, 2, 2),
            next_action.expand(1, 2, 2),
            coords,
        )
    )[None]
    torch.testing.assert_close(
        torch.nn.functional.conv2d(newx, mapped[h.LIFT], mapped["bias"]),
        torch.nn.functional.conv2d(oldx, old[h.LIFT], old["bias"]),
        rtol=1e-6,
        atol=2e-6,
    )
    if k == 4:
        assert (
            torch.count_nonzero(mapped[h.LIFT][:, :9]) == 0
            and torch.count_nonzero(mapped[h.LIFT][:, 13:16]) == 0
        )
        assert torch.equal(mapped[h.LIFT][:, 18:20], old[h.LIFT][:, 6:8])
    else:
        assert torch.equal(mapped[h.LIFT], old[h.LIFT])


def test_split_and_mask_mismatch_fail_closed():
    base = Base(3)
    base.meta["split"] = "validation"
    with pytest.raises(ValueError):
        h.HistoryWindowAdapter(base, 4)[0]
    base = Base(3)
    base.reader.frames[0]["mask"].zero_()
    with pytest.raises(ValueError):
        h.HistoryWindowAdapter(base, 4)[0]


def test_shape_nonfinite_and_remap_failures():
    with pytest.raises(ValueError):
        h.history_indices(-1, 4)
    with pytest.raises(ValueError):
        h.history_indices(0, 3)
    with pytest.raises(FloatingPointError):
        h.build_input(
            torch.full((4, 3, 2, 2), float("nan")),
            torch.ones(1, 2, 2),
            torch.ones(4, 1),
            0.0,
        )
    with pytest.raises(ValueError):
        h.history_warmstart_state(
            {h.LIFT: torch.ones(24, 8, 1, 1)}, {h.LIFT: torch.ones(24, 18, 1, 1)}, 4
        )


def test_preserved_window_denominator_and_padding_subsets():
    totals = []
    for trajectories, frames, stride in [(20, 801, 20), (8, 201, 2), (16, 129, 2)]:
        starts = range(0, frames - 100, stride)
        totals.append(
            (
                trajectories * len(starts),
                trajectories * sum(any(h.history_indices(s, 4)[1]) for s in starts),
            )
        )
    assert totals == [(720, 20), (408, 16), (240, 32)]
    assert sum(n for n, _ in totals) == 1368 and sum(c for _, c in totals) == 68
