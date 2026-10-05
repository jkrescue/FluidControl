"""Synthetic terminal integrity tests; no actual candidate/data/model loads."""
import copy
import json
from pathlib import Path

import pytest
import torch

import audit_fcp026_candidate as audit


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def terminal():
    return dict(LoadState="loaded", ActiveState="active", SubState="exited", Result="success",
                ExecMainCode="1", ExecMainStatus="0", MainPID="0", InvocationID="a"*32)


def test_exact_retained_terminal():
    audit.validate_terminal(terminal(), "a"*32)


@pytest.mark.parametrize("key,value", [("ActiveState", "activating"), ("SubState", "start"),
    ("MainPID", "123"), ("Result", "exit-code"), ("ExecMainStatus", "1"),
    ("ExecMainCode", "2"), ("InvocationID", "b"*32), ("LoadState", "not-found")])
def test_terminal_rejects_live_failed_collected_wrong_identity(key, value):
    props = terminal()
    props[key] = value
    with pytest.raises(ValueError):
        audit.validate_terminal(props, "a"*32)


def fixture_records(k=1):
    records, identities = [], []
    for update in range(1, 172):
        batch = []
        for consumed in range((update-1)*8, update*8):
            ident = dict(dataset_index=0, case="software_fixture", start=consumed*2,
                         split="train", rollout_steps=100)
            raw = list(range(ident["start"]-k+1, ident["start"]+1))
            row = dict(identity=ident, history=dict(frame_indices=[max(0, i) for i in raw],
                padding_mask=[i < 0 for i in raw], full_observed_history=all(i >= 0 for i in raw)),
                flow_history_sha256="f"*64, h1_channel_mse=[1.]*4, ar_channel_mse=[2.]*4,
                h1_balanced=1., ar_balanced=2., total=1.5, chunk_size=10, chunks=10)
            batch.append(row)
            identities.append(ident)
        records.append(dict(update=update, consumed_windows=update*8, windows=8, optimizer_steps=1,
            preclip_mean_gradient_norm=2., applied_clip_scale=1/(2+1e-6),
            gradient_audit=dict(new_history_l2=0., inherited_l2=2.),
            parameter_update=dict(new_history_l2=0., inherited_l2=.01),
            cumulative_displacement=dict(new_history_l2=0., inherited_l2=.01*update),
            records=batch, mean_objective=dict(h1_balanced=1., ar_balanced=2., total=1.5)))
    return records, dict(ordered_identities=identities)


@pytest.mark.parametrize("k", [1, 4])
def test_exact_consumption_and_causal_history(k):
    records, inventory = fixture_records(k)
    assert len(audit.validate_records(records, k, inventory)) == 1368


@pytest.mark.parametrize("mutation", ["count", "order", "future", "clip", "objective", "history_norm", "nan"])
def test_records_fail_closed(mutation):
    records, inventory = fixture_records()
    if mutation == "count":
        records.pop()
    elif mutation == "order":
        records[0]["records"].reverse()
    elif mutation == "future":
        records[0]["records"][0]["history"]["frame_indices"] = [1]
    elif mutation == "clip":
        records[0]["applied_clip_scale"] = 1
    elif mutation == "objective":
        records[0]["records"][0]["ar_balanced"] = 1
    elif mutation == "history_norm":
        records[0]["gradient_audit"]["new_history_l2"] = .1
    else:
        records[0]["preclip_mean_gradient_norm"] = float("nan")
    with pytest.raises(ValueError):
        audit.validate_records(records, 1, inventory)


def test_progress_requires_actual_complete_order_not_result_flag():
    _, inv = fixture_records()
    events = []
    for update in range(171):
        events += [dict(event="training_window_complete", history_k=1, consumed=i+1, global_index=i)
                   for i in range(update*8, (update+1)*8)]
        events.append(dict(event="accumulation_update_complete", history_k=1, update=update+1))
    text = "\n".join(json.dumps(x) for x in events)
    audit.validate_progress(text, 1, inv)
    with pytest.raises(ValueError):
        audit.validate_progress("\n".join(text.splitlines()[:-1]), 1, inv)
    events[0]["global_index"] = 1
    with pytest.raises(ValueError):
        audit.validate_progress("\n".join(json.dumps(x) for x in events), 1, inv)


def tensors():
    result = {audit.LIFT: torch.arange(192).reshape(24, 8, 1, 1).float()}
    result.update({f"parameter_{i}": torch.ones(2) for i in range(27)})
    result.update({name: torch.ones(24 if i == 0 else 48) for i, name in enumerate(audit.FROZEN)})
    result["device_buffer"] = torch.empty(0)
    return result


def test_k4_mapping_and_frozen_bias_integrity():
    parent = tensors()
    saved = copy.deepcopy(parent)
    old = parent[audit.LIFT]
    new = old.new_zeros((24,20,1,1))
    new[:,9:13], new[:,16:] = old[:,:4], old[:,4:]
    new[:,0] = .001
    saved[audit.LIFT] = new
    initial, changed = audit.validate_tensor_pair(parent, saved, 4)
    assert torch.equal(initial[audit.LIFT][:,9:13], old[:,:4])
    assert torch.equal(initial[audit.LIFT][:,16:], old[:,4:])
    assert torch.count_nonzero(initial[audit.LIFT][:,:9]) == 0
    assert changed == [audit.LIFT]
    saved[audit.FROZEN[0]][0] += 1
    with pytest.raises(ValueError):
        audit.validate_tensor_pair(parent, saved, 4)


def optimizer_state():
    parameters = [v for n,v in tensors().items() if n not in audit.FROZEN and v.numel()]
    return dict(epoch=1, optimizer_state_dict=dict(param_groups=[dict(params=list(range(28)),
        lr=1.5625e-7, betas=(.9,.999), eps=1e-8, weight_decay=1e-4)],
        state={i: dict(step=torch.tensor(171.), exp_avg=torch.zeros_like(v), exp_avg_sq=torch.ones_like(v))
               for i,v in enumerate(parameters)}))


def test_actual_adam_28_steps_and_parameter_shapes():
    state = optimizer_state()
    audit.validate_optimizer(state, tensors())
    state["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(170.)
    with pytest.raises(ValueError):
        audit.validate_optimizer(state, tensors())


@pytest.mark.parametrize("mutation", ["shape", "lr", "nan", "negative", "bool_id"])
def test_optimizer_corruption(mutation):
    state = optimizer_state()
    opt = state["optimizer_state_dict"]
    if mutation == "shape":
        opt["state"][0]["exp_avg"] = torch.zeros(1)
        opt["state"][0]["exp_avg_sq"] = torch.ones(1)
    elif mutation == "lr":
        opt["param_groups"][0]["lr"] = 1e-5
    elif mutation == "nan":
        opt["state"][0]["exp_avg"].fill_(float("nan"))
    elif mutation == "negative":
        opt["state"][0]["exp_avg_sq"].fill_(-1)
    else:
        opt["param_groups"][0]["params"][0] = False
    with pytest.raises(ValueError):
        audit.validate_optimizer(state, tensors())


def test_candidate_map_exact_seven_confined_files(tmp_path):
    for name in audit.FILES:
        path = tmp_path/"candidate"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("software fixture")
    mapping = audit.candidate_files(tmp_path)
    assert set(mapping) == {"candidate/"+name for name in audit.FILES}
    (tmp_path/"candidate/result.json").unlink()
    (tmp_path/"outside.json").write_text("outside")
    (tmp_path/"candidate/result.json").symlink_to(tmp_path/"outside.json")
    with pytest.raises(ValueError):
        audit.candidate_files(tmp_path)


def test_panel_aggregate_and_warm_padding_recomputed():
    import train_fcp026_history as trainer
    from types import SimpleNamespace
    records, inv = fixture_records()
    flat = [row for record in records for row in record["records"]]
    indices = [160, 816, 923, 975, 1077, 1233]
    rows = []
    for i in indices:
        row = copy.deepcopy(flat[i])
        row["global_index"] = i
        domain = dict(bias_mse=.25, signed_mean_error=.5, rms_error_mse=.25,
                      absolute_rms_error=.5, centered_residual_mse=.3,
                      predicted_tail_rms=1.5, truth_tail_rms=1.)
        row["panel"] = dict(objective=copy.deepcopy(flat[i]), domains=dict(h1=domain, ar=domain))
        rows.append(row)
    def aggregate(rows):
        return {d: dict(six_window_original_objective=sum(r["panel"]["objective"][d+"_balanced"] for r in rows)/6,
                       five_nonzero={key: sum(r["panel"]["domains"][d][key] for r in rows[1:])/5
                                     for key in ("bias_mse", "rms_error_mse", "absolute_rms_error", "centered_residual_mse")})
                for d in ("h1", "ar")}
    p020 = SimpleNamespace(aggregate=aggregate)
    panel = dict(rows=rows, aggregate=aggregate(rows), history_subgroups=trainer.grouped_panel(rows))
    audit.validate_panel(panel, trainer, p020, inv["ordered_identities"], 1, indices)
    panel["aggregate"]["h1"]["five_nonzero"]["bias_mse"] += .001
    with pytest.raises(ValueError):
        audit.validate_panel(panel, trainer, p020, inv["ordered_identities"], 1, indices)
