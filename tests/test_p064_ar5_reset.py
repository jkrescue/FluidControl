import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import hashlib
import json
import sys

REPO = Path('/workspace/fluid_control')
FROZEN = REPO / 'artifacts/fcp064_training_source_20261006_immutable'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

SPEC = importlib.util.spec_from_file_location("ar5", Path(__file__).parents[1] / "scripts/p064_ar5_reset_states.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    state = torch.zeros(1, 3, 2, 2)
    targets = torch.arange(1, 101).float()[None, :, None, None, None].expand(1, 100, 3, 2, 2).clone() * 10
    mask = torch.ones(1, 1, 2, 2)
    omega = torch.arange(101).float()[None]
    flow = torch.nn.Linear(1, 1).requires_grad_(False)
    original = SimpleNamespace(
        true_state_inputs=lambda q, t: torch.cat((q[:, None], t[:, :-1]), dim=1),
        make_inputs=lambda q, mask, now, nxt: torch.cat((q, mask, now[:, None, None, None].expand(-1, 1, 2, 2), nxt[:, None, None, None].expand(-1, 1, 2, 2)), dim=1),
    )
    calls = []
    def predict(model, inputs, supplied_mask):
        assert model is flow and not torch.is_grad_enabled()
        calls.append(inputs.clone())
        return torch.ones_like(inputs[:, :3]), None
    return flow, state, targets, mask, omega, predict, original, calls


def test_exact_100_calls_resets_and_action_clock():
    *args, calls = fixture()
    states, audit = m.training_states(*args)
    assert len(calls) == 100 and audit["flow_forward_calls"] == 100
    assert audit["reset_indices"] == list(range(0, 100, 5))
    for s in range(100):
        assert torch.equal(states[:, s], torch.full_like(states[:, s], (s // 5) * 50 + s % 5))
        assert torch.all(calls[s][:, -2] == s)
        assert torch.all(calls[s][:, -1] == s + 1)
    assert not states.requires_grad


def test_future_target_does_not_enter_current_block():
    *args, _ = fixture()
    baseline, _ = m.training_states(*args)
    args[2][:, 0:4] += 10000  # q1..q4 are not restart inputs.
    args[2][:, 99] += 10000  # q100 never enters any supervised input.
    changed, _ = m.training_states(*args)
    assert torch.equal(baseline, changed)


def test_restart_uses_current_observed_q5_not_q6():
    *args, _ = fixture()
    baseline, _ = m.training_states(*args)
    args[2][:, 4] += 3
    changed, _ = m.training_states(*args)
    assert torch.equal(changed[:, :5], baseline[:, :5])
    assert torch.equal(changed[:, 5:10], baseline[:, 5:10] + 3)
    assert torch.equal(changed[:, 10:], baseline[:, 10:])


def test_frozen_flow_and_exact_shape_fail_closed():
    *args, _ = fixture()
    args[0].requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        m.training_states(*args)
    args[0].requires_grad_(False)
    args[4] = args[4][:, :-1]
    with pytest.raises(ValueError, match="H100"):
        m.training_states(*args)


def test_actual_original_chunk_mixed_loss_gradient_and_eight_average():
    sys.path.insert(0, str(FROZEN / 'scripts'))
    path = FROZEN / 'scripts/p026_history_objective.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '4d27fb53e05df73ba94d84bf42ba8205d78ebe6f91de832a68659870ea7d77c0'
    chunk = load('g_original_chunk', path)
    *args, flow_calls = fixture()
    ar, _ = m.training_states(*args)
    h1 = args[-1].true_state_inputs(args[1], args[2])
    class Objective:
        @staticmethod
        def balanced_force_objective(pred, target):
            cm = ((pred-target)**2).mean((0, 1))
            return dict(balanced=.5*cm.mean()+.5*cm[3], channel_mse=cm)
    w = torch.ones(4, requires_grad=True)
    aero_calls = []
    def predict(model, inputs, masks):
        aero_calls.append(inputs.detach().clone())
        return None, inputs[:, 0, 0, 0, None] * w[None]
    for _ in range(8):
        result = chunk.chunk_force_objective(None, ar, h1, args[3], args[4][:, :, None],
            torch.zeros(1, 100, 4), torch.empty(1, 0, 3, 2, 2), torch.empty(1, 0, 1),
            predict, Objective, backward=True)
    w.grad.div_(8)
    expected = w.detach().clone().requires_grad_()
    hl = Objective.balanced_force_objective(h1[:, :, 0, 0, 0, None]*expected, torch.zeros(1, 100, 4))
    al = Objective.balanced_force_objective(ar[:, :, 0, 0, 0, None]*expected, torch.zeros(1, 100, 4))
    (.5*hl['balanced']+.5*al['balanced']).backward()
    torch.testing.assert_close(w.grad, expected.grad)
    assert len(flow_calls) == 100 and len(aero_calls) == 80
    assert all(x.shape == (20, 6, 2, 2) for x in aero_calls)
    assert result['normalized_predictions']['ar'].shape == (1, 100, 4)
    opt = torch.optim.AdamW([w], lr=1.5625e-7)
    torch.nn.utils.clip_grad_norm_([w], 1.)
    opt.step()
    assert opt.state[w]['step'] == 1


def test_consumer_actual_producer_protocol_and_wrong_profile_rejection(tmp_path):
    root = Path(__file__).parents[1]
    consumer = root / 'dual_fno.py'
    if not consumer.is_file():
        consumer = root / 'src/fluid_control/dual_fno_ar5_reset.py'
    loader = load('g_consumer', consumer)
    runner = load('g_runner', root / 'scripts/train_p064_ar5_reset.py')
    parent = REPO / 'artifacts/fcp064_controlled_aero_arm_b_20261006'
    payload = json.loads((parent / 'dual_model_manifest.json').read_text())
    oldp = json.loads((parent / 'training_protocol.json').read_text())
    # Schedule/order separately have immutable audits; protocol consumes its exact digest.
    mock = SimpleNamespace(compile_schedule=lambda order, arm: None,
        schedule_sha256=lambda _: oldp['schedule_sha256'])
    protocol = runner.protocol('B', mock, [])
    contract = loader._experiment_contract('FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO')
    def write():
        target = tmp_path / 'training_protocol.json'
        target.write_text(json.dumps(protocol))
        payload.update(training_semantics=protocol, training_protocol_file=target.name,
            training_protocol_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    write()
    loader._validate_p026_protocol(tmp_path, payload, contract)
    with pytest.raises(ValueError):
        loader._validate_p026_protocol(tmp_path, payload, loader._experiment_contract(loader.P064_SYSTEM_KIND['B']))
    protocol['diagnostic_ar_reset_every'] = 5
    write()
    with pytest.raises(ValueError):
        loader._validate_p026_protocol(tmp_path, payload, contract)


def test_runner_diagnostic_branch_is_original_continuous_generator():
    import ast
    root = Path(__file__).parents[1]
    tree = ast.parse((root / 'scripts/train_p064_ar5_reset.py').read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run')
    branch = next(n for n in run.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'backward')
    assert 'reset.training_states' in ast.unparse(branch.body[0])
    assert 'objective.frozen_flow_states' in ast.unparse(branch.orelse[0])
    assert 'reset.training_states' not in ast.unparse(ast.Module(body=branch.orelse, type_ignores=[]))
    assert 'chunk.chunk_force_objective' in ast.unparse(run)
