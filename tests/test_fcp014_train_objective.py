import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import diagnose_fcp014_train_objective as d

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    objective = d.load_objective(ROOT / 'artifacts/fcp013_training_source_1634c05_immutable')
    model = torch.nn.Linear(1, 1).eval()
    states = torch.arange(100, dtype=torch.float32).view(1, 100, 1, 1, 1).expand(1, 100, 3, 1, 1)
    batch = {'state': states[:, 0], 'mask': torch.ones(1, 1, 1, 1),
             'omega': torch.zeros(1, 101), 'target_force': torch.zeros(1, 100, 4)}
    seen = []
    def predict(network, inputs, mask):
        seen.append((network.training, torch.is_grad_enabled(), inputs.shape[0]))
        return inputs[:, :3] * 0, inputs[:, 0, 0, 0, None].expand(-1, 4)
    return objective, model, states, states + 100, batch, torch.tensor([1., 2., 3., 4.]), predict, seen


def test_exact_objective_schedule_mode_and_residual(fixture):
    objective, model, ar, h1, batch, std, predict, seen = fixture
    before = copy.deepcopy(model.state_dict())
    panel = d.capture_force_panel(*fixture[:7])
    assert seen == [(True, False, 20)] * 10
    assert not model.training
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    direct = objective.chunk_force_objective(model, ar, h1, batch['mask'], batch['omega'],
                                             batch['target_force'], predict, chunk_size=10, backward=False)
    assert panel['objective'] == direct
    for domain, offset in [('h1', 100), ('ar', 0)]:
        residual = np.arange(100) + offset
        row = panel['domains'][domain]
        np.testing.assert_array_equal(row['rear_cl_normalized_residual'], residual)
        np.testing.assert_array_equal(row['rear_cl_physical_residual'], residual * 4)
        assert row['tail62']['samples'] == 62
        assert row['analytic_normalized_rear_cl_bias_derivative'] == pytest.approx(1.25 * residual.mean())


@pytest.mark.parametrize('n', [62, 100])
def test_bias_centered_identity(n):
    values = np.arange(n) - (n-1)/2 + 3
    result = d.residual_statistics(values)
    assert result['signed_residual_mean'] == 3
    assert result['bias_mse'] == 9
    assert result['mse'] == pytest.approx(9 + result['centered_residual_mse'])
    assert abs(result['decomposition_residual']) < 1e-10


@pytest.mark.parametrize('values', [[0]*61, [float('nan')]*100, [[0]*100]])
def test_invalid_residual(values):
    with pytest.raises(ValueError): d.residual_statistics(values)


def test_analytic_bias_matches_actual_loss_gradient(fixture):
    objective = fixture[0]
    bias = torch.tensor(0.3, requires_grad=True)
    forces = torch.arange(400, dtype=torch.float32).reshape(1, 100, 4)/400
    addition = torch.stack((bias*0, bias*0, bias*0, bias))
    predicted = forces + addition
    objective.balanced_force_objective(predicted, torch.zeros_like(predicted))['balanced'].backward()
    assert bias.grad.item() == pytest.approx(1.25 * predicted[..., 3].mean().item())


@pytest.mark.parametrize('bad', ['mask', 'std', 'dropout', 'nan'])
def test_reject_invalid_capture(fixture, bad):
    args = list(fixture[:7])
    if bad == 'mask': args[4]['mask'].zero_()
    if bad == 'std': args[5][3] = 0
    if bad == 'dropout': args[1] = torch.nn.Dropout()
    if bad == 'nan':
        args[6] = lambda model, x, mask: (x[:, :3], torch.full((20,4), float('nan')))
    with pytest.raises(ValueError): d.capture_force_panel(*args)


def test_modes_restored_on_failure(fixture):
    args = list(fixture[:7])
    def fail(*args): raise RuntimeError('sentinel')
    args[6] = fail
    with pytest.raises(RuntimeError): d.capture_force_panel(*args)
    assert not args[1].training


def test_fixed_train_windows():
    items = [{'global_index': i, 'family': f, 'identity': {'case': c, 'start': s,
              'split': 'train', 'rollout_steps': 100}} for i, f, c, s in d.WINDOWS]
    d.validate_windows(items)
    with pytest.raises(ValueError): d.validate_windows(items[::-1])
    items[0]['identity']['split'] = 'validation'
    with pytest.raises(ValueError): d.validate_windows(items)


def test_compare_and_exclusive_output(fixture, tmp_path):
    panel = d.capture_force_panel(*fixture[:7])
    diff = d.compare_panels(panel, panel)
    assert diff['objective_delta'] == {'h1_balanced': 0, 'ar_balanced': 0, 'total': 0}
    path = tmp_path/'result.json'
    d.write_exclusive(path, diff)
    assert json.loads(path.read_text()) == diff
    with pytest.raises(FileExistsError): d.write_exclusive(path, {})
    with pytest.raises(ValueError): d.write_exclusive(tmp_path/'nan.json', {'x': float('nan')})
    assert not (tmp_path/'nan.json').exists()


@pytest.mark.parametrize('free,available,okay', [(20971520,20971520,True),(20971519,30000000,False),(30000000,20971519,False)])
def test_resource_floor(tmp_path, free, available, okay):
    path = tmp_path/'meminfo'
    path.write_text(f'MemFree: {free} kB\nMemAvailable: {available} kB\n')
    if okay: assert d.check_memory(path)['MemFree'] == free
    else:
        with pytest.raises(RuntimeError): d.check_memory(path)


def test_reject_changed_objective(tmp_path):
    folder = tmp_path/'scripts'
    folder.mkdir()
    (folder/'train_fcp013_independent_force_fno.py').write_text('raise AssertionError("not executed")')
    with pytest.raises(ValueError, match='differs'): d.load_objective(tmp_path)
