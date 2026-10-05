"""Small CPU software fixtures; not CFD or model-quality evidence."""
import copy
import json

import pytest
import torch

from audit_fcp013_dual_candidate import validate_guard, validate_optimizer, validate_state_pair
from train_fcp013_independent_force_fno import OFFICIAL_FROZEN_PARAMETER_NAMES


def test_tensor_change_requires_frozen_bias_preservation_and_finite_values():
    parent = {key: torch.zeros(2) for key in OFFICIAL_FROZEN_PARAMETER_NAMES}
    parent['weight'] = torch.ones(2)
    aero = copy.deepcopy(parent)
    aero['weight'][0] = 2
    assert validate_state_pair(parent, aero) == ['weight']
    aero[OFFICIAL_FROZEN_PARAMETER_NAMES[0]][0] = 1
    with pytest.raises(ValueError):
        validate_state_pair(parent, aero)
    aero = copy.deepcopy(parent)
    with pytest.raises(ValueError):
        validate_state_pair(parent, aero)
    aero['weight'][0] = float('nan')
    with pytest.raises(ValueError):
        validate_state_pair(parent, aero)


def optimizer():
    return {'epoch': 1, 'optimizer_state_dict': {
        'param_groups': [{'lr': 1e-5, 'weight_decay': 1e-4, 'betas': (0.9, 0.999),
                          'eps': 1e-8, 'params': list(range(28))}],
        'state': {i: {'step': torch.tensor(1368.), 'exp_avg': torch.zeros(2),
                      'exp_avg_sq': torch.zeros(2)} for i in range(28)}}}


def test_persisted_optimizer_steps_not_self_reported_count():
    value = optimizer()
    validate_optimizer(value)
    value['optimizer_state_dict']['state'][0]['step'] = torch.tensor(1367.)
    with pytest.raises(ValueError):
        validate_optimizer(value)


def test_optimizer_frozen_count_and_protocol():
    for change in ('params', 'lr', 'weight_decay'):
        value = optimizer()
        value['optimizer_state_dict']['param_groups'][0][change] = [] if change == 'params' else 0.5
        with pytest.raises(ValueError):
            validate_optimizer(value)


def guard():
    return {'event': 'gpu_guard_complete', 'exit_code': 0,
            'min_required_mem_available_gib': 20, 'min_observed_mem_available_gib': 100,
            'memory_samples': 100}


def test_completed_guard_and_no_hidden_violation():
    value = guard()
    validate_guard(json.dumps(value))
    with pytest.raises(ValueError):
        validate_guard('still running')
    with pytest.raises(ValueError):
        validate_guard(json.dumps({'event': 'gpu_floor_violation'}) + '\n' + json.dumps(value))
    for key, bad in [('exit_code', 1), ('min_observed_mem_available_gib', 19),
                     ('min_observed_mem_available_gib', float('nan'))]:
        value = guard()
        value[key] = bad
        with pytest.raises(ValueError):
            validate_guard(json.dumps(value))
