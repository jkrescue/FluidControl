import json
import pytest
from test_dashboard_registered import m, state, REG

@pytest.mark.parametrize('kind,mode', [('p029_scales','scales'),('p029_train','train')])
def test_p029_counts_actual_events(kind, mode):
    reg = {**REG, 'progress_kind': kind, 'planned_updates': {'FNO': 171}}
    rows = [dict(event='window_complete', mode=mode, case='case', start=i, dataset_index=0, split='train', rollout_steps=100) for i in range(8)]
    rows.append(dict(event='group_complete', mode=mode, group=1))
    def parse(value):
        return m._parse_registered_progress(state(), '\n'.join(map(json.dumps, value)), True, reg)
    result = parse(rows)
    assert result['verified'] and result['running'] and not result['admission']
    assert result['updates'] == {'FNO': 1} and result['windows'] == {'FNO': 8}
    for bad in (rows + [rows[0]], rows[1:], rows + [rows[-1]], [{**rows[0], 'mode':'wrong'}], [{**rows[0], 'split':'validation'}]):
        assert not parse(bad)['verified']

def test_p029_probe_is_one_window_not_training():
    reg = {**REG, 'progress_kind':'p029_probe', 'planned_updates':{'FNO':1}}
    row = dict(event='window_complete', mode='resource-probe', case='case', start=0, dataset_index=0, split='train', rollout_steps=100)
    result = m._parse_registered_progress(state(), json.dumps(row), True, reg)
    assert result['verified'] and result['updates'] == {'FNO':1} and '不更新参数' in result['progress_unit']
    assert not m._parse_registered_progress(state(), '', True, {**reg, 'invocation':'wrong'})['verified']
