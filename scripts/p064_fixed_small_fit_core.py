"""Project lifecycle wrapper around installed torch.optim.LBFGS; no model API."""
from __future__ import annotations

import copy
import time
import torch


class BudgetStop(RuntimeError):
    pass


def fit_fixed_panel(parameters, objective_backward, evaluate, *, max_outer=200,
                    max_closures=300, seconds=1200, clock=time.monotonic,
                    progress=None):
    """Callbacks must use the same fixed panel, partition, precision and mean.

    objective_backward() returns a scalar and accumulates its exact gradient.
    evaluate() returns loss and four normalized_rmse values without gradients.
    No clipping/decay. A trial callback can NEVER declare fitting success.
    """
    params = list(parameters)
    if not params or any(not p.requires_grad for p in params):
        raise ValueError("explicit nonempty trainable parameter list required")
    opt = torch.optim.LBFGS(params, lr=1., max_iter=1, max_eval=300,
                           history_size=5, line_search_fn="strong_wolfe",
                           tolerance_grad=1e-7, tolerance_change=1e-9)
    start = clock()
    closures = 0
    records = []
    trials = []

    def measured():
        with torch.no_grad():
            row = evaluate()
        values = torch.as_tensor([row['loss'], *row['normalized_rmse']], dtype=torch.float64)
        if len(row['normalized_rmse']) != 4 or not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite/invalid accepted-point measurement")
        return row

    initial = measured()
    status = "BUDGET_STOP_NOT_FITTED"
    final = initial
    for outer in range(max_outer):
        if max(final['normalized_rmse']) <= .01:
            status = "TRAIN_PANEL_FITTED_NOT_ADMISSION"
            break
        if closures >= max_closures or clock() - start >= seconds:
            break
        before = [p.detach().clone() for p in params]
        gradients_before = [None if p.grad is None else p.grad.detach().clone() for p in params]
        state_before = copy.deepcopy(opt.state_dict())
        calls_before = closures

        def closure():
            nonlocal closures
            if closures >= max_closures or clock() - start >= seconds:
                raise BudgetStop("global closure/time budget")
            closures += 1
            opt.zero_grad(set_to_none=True)
            loss = objective_backward()
            if loss.ndim or not torch.isfinite(loss):
                raise FloatingPointError("nonfinite/non-scalar trial loss")
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("missing/nonfinite trial gradient")
            entry = {'event': 'lbfgs_trial_closure', 'trial_not_accepted': True,
                     'outer': outer + 1, 'closure': closures, 'loss': float(loss.detach()),
                     'gradient_l2': sum(float(p.grad.detach().abs().double().square().sum())
                                        for p in params) ** .5,
                     'gradient_inf': max(float(p.grad.detach().abs().max()) for p in params),
                     'elapsed_seconds': clock() - start}
            trials.append(entry)
            if progress is not None:
                progress(entry)
            return loss

        try:
            returned_initial_loss = float(opt.step(closure))
            final = measured()  # Not step's old loss and not the last trial.
        except Exception as error:
            try:
                with torch.no_grad():
                    for p, old in zip(params, before, strict=True):
                        p.copy_(old)
                opt.load_state_dict(state_before)
                for p, grad in zip(params, gradients_before, strict=True):
                    p.grad = grad
            except Exception as restore_error:
                records.append({'outer': outer + 1, 'restored_previous_accepted': False,
                                'error': repr(error), 'restore_error': repr(restore_error)})
                return {'status': 'RESTORE_FAILURE', 'initial': initial, 'final': None,
                        'last_recorded_measurement_not_current_tensor_claim': final,
                        'records': records, 'trials': trials, 'closures': closures}, opt
            status = "BUDGET_STOP_NOT_FITTED" if isinstance(error, BudgetStop) else "OPTIMIZATION_FAILURE"
            records.append({'outer': outer + 1, 'restored_previous_accepted': True,
                            'error': repr(error), 'error_type': type(error).__name__,
                            'closures': closures})
            break
        displacement = sum(float((p.detach() - old).abs().square().sum())
                           for p, old in zip(params, before, strict=True)) ** .5
        state = opt.state[params[0]]
        records.append({'outer': outer + 1, 'measurement': final,
                        'returned_initial_loss': returned_initial_loss,
                        'parameter_displacement_l2': displacement,
                        'closures': closures, 'closures_this_call': closures - calls_before,
                        'optimizer_n_iter': state.get('n_iter', 0),
                        'history_length': len(state.get('old_dirs', [])),
                        'step_size': float(state.get('t', 0.)),
                        'elapsed_seconds': clock() - start})
        if progress is not None:
            progress({'event': 'lbfgs_returned_point', **records[-1]})
        if displacement == 0:
            status = "LOCAL_STAGNATION_NOT_FITTED"
            break
    if status != "OPTIMIZATION_FAILURE" and max(final['normalized_rmse']) <= .01:
        status = "TRAIN_PANEL_FITTED_NOT_ADMISSION"
    return {'status': status, 'initial': initial, 'final': final, 'records': records,
            'trials': trials, 'closures': closures, 'elapsed_seconds': clock() - start}, opt
