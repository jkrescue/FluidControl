import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('audit64', Path(__file__).resolve().parents[1] / 'scripts/check_absolute64_terminal.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    protocol = dict(training_experiment='FC-P064-ABSOLUTE64', arm='B',
                    training_windows=512, optimizer_steps=64, accumulation_windows=8,
                    schedule_epochs=2, diagnostic_counts=[0, 512], b00_windows=128,
                    b00_windows_per_schedule_epoch=64, b00_weight=.25,
                    replacement_within_each_update=[0, 4],
                    objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',
                    allocator_bytes=16 * 1024**3, update32_reference_tensor_sha256='abc')
    comparison = dict(expected='abc', actual='abc', exact=True)
    result = dict(update32_comparison=comparison,
                  records=[dict(schedule_epoch=i // 32 + 1) for i in range(64)],
                  resources=[dict(MemAvailable=100)])
    manifest = dict(kind='FC_P064_ABSOLUTE64_ARM_B_CONTROLLED_AERO_FORCE_FNO',
                    aerodynamic=dict(checkpoint_epoch=2, model_file='FNO.0.2.mdlus', state_file='checkpoint.0.2.pt'))
    return result, protocol, manifest, copy.deepcopy(comparison), dict(aerodynamic_terminal_tensor_sha256='abc')


def test_two_identical_schedules_continuous_counter():
    rows = audit.repeated_schedule(list(range(1368)), 'B')
    assert len(rows) == 512
    assert [x['consumed'] for x in rows] == list(range(1, 513))
    assert [{k: v for k, v in x.items() if k != 'consumed'} for x in rows[:256]] == [
        {k: v for k, v in x.items() if k != 'consumed'} for x in rows[256:]]
    assert sum(x['source'] == 'controlled_b00' for x in rows) == 128


def test_extension_valid():
    assert audit.check_extension(*fixture())['update32_exact']


@pytest.mark.parametrize('change', ['mismatch', 'epoch', 'panel', 'reserve'])
def test_extension_rejects(change):
    args = fixture()
    if change == 'mismatch':
        args[3]['actual'] = 'different'
    elif change == 'epoch':
        args[2]['aerodynamic']['checkpoint_epoch'] = 1
    elif change == 'panel':
        args[1]['diagnostic_counts'] = [0, 256, 512]
    else:
        args[0]['resources'][0]['MemAvailable'] = 21
    with pytest.raises(ValueError):
        audit.check_extension(*args)


def test_live_unit_rejected_before_candidate():
    with pytest.raises(ValueError, match='still running'):
        audit.base.terminal(dict(InvocationID='inv', MainPID='42'), 'inv')
