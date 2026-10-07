"""Synthetic CPU integration tests; no CFD data or scientific FNO execution."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts/p064_fixed_small_fit_core.py'
spec = importlib.util.spec_from_file_location('reviewed_bounded_lbfgs', PATH)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def rosenbrock(*, outer=6, budget=300, fail_at=None):
    p = torch.nn.Parameter(torch.tensor([-1.2, 1.], dtype=torch.float64))
    count = 0
    def loss():
        return (1-p[0]).square() + 100*(p[1]-p[0].square()).square()
    def backward():
        nonlocal count
        count += 1
        if count == fail_at:
            raise RuntimeError('synthetic resource failure at trial')
        value = loss()
        value.backward()
        return value
    def evaluate():
        return {'loss': float(loss()), 'normalized_rmse': [1.]*4}
    result, opt = core.fit_fixed_panel([p], backward, evaluate,
                                      max_outer=outer, max_closures=budget)
    return p, result, opt


def test_actual_persistent_history_and_global_closures():
    _, r, _ = rosenbrock()
    assert [x['optimizer_n_iter'] for x in r['records']] == list(range(1, 7))
    assert max(x['history_length'] for x in r['records']) >= 3
    assert max(x['history_length'] for x in r['records']) <= 5
    assert r['closures'] == sum(x['closures_this_call'] for x in r['records'])
    assert [x['closure'] for x in r['trials']] == list(range(1, r['closures']+1))
    assert all(x['trial_not_accepted'] for x in r['trials'])


def quadratic(budget):
    p = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    def backward():
        loss = (p-1).square().sum()
        loss.backward()
        return loss
    def evaluate():
        return {'loss': float((p-1).square().sum()),
                'normalized_rmse': [float(abs(p.item()-1))]*4}
    r, opt = core.fit_fixed_panel([p], backward, evaluate, max_closures=budget)
    return p, r, opt


def test_actual_accepted_metric_not_returned_old_loss():
    p, r, _ = quadratic(10)
    assert p.item() == 1
    assert r['records'][0]['returned_initial_loss'] == 1
    assert r['final']['loss'] == 0
    assert r['status'] == 'TRAIN_PANEL_FITTED_NOT_ADMISSION'


def test_budget_trial_optimum_not_success_and_empty_state_restored():
    p, r, opt = quadratic(1)
    assert p.item() == 2
    assert p.grad is None
    assert not opt.state
    assert r['closures'] == 1
    assert r['status'] == 'BUDGET_STOP_NOT_FITTED'
    assert r['final']['loss'] == 1
    assert r['records'][-1]['restored_previous_accepted']


@pytest.mark.parametrize('failure', ['budget', 'runtime'])
def test_nonempty_state_parameter_and_grad_restore(failure):
    p0, base, opt0 = rosenbrock(outer=4)
    count = base['closures']
    p, r, opt = rosenbrock(outer=5, budget=count+1 if failure=='budget' else 300,
                           fail_at=count+2 if failure=='runtime' else None)
    assert r['records'][-1]['restored_previous_accepted']
    assert r['records'][-1]['error_type'] == ('BudgetStop' if failure=='budget' else 'RuntimeError')
    assert r['closures'] == count+(1 if failure=='budget' else 2)
    equal(p, p0)
    equal(p.grad, p0.grad)
    equal(opt.state_dict(), opt0.state_dict())
    assert r['final'] == base['final']
    assert all('measurement' not in x for x in r['records'][4:])
