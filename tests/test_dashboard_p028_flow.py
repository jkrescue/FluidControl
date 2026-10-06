import json
from test_dashboard_registered import m, state, REG

def test_actual_flow_log_schema_counts_windows_and_updates():
    reg = {**REG, 'progress_kind': 'flow_training', 'planned_updates': {'FNO': 171}}
    rows = [dict(event='training_window_complete', case='real_case', start=i,
                 dataset_index=0, split='train', rollout_steps=100) for i in range(8)]
    rows.append(dict(event='accumulation_update_complete', update=1))
    parsed = m._parse_registered_progress(state(), '\n'.join(map(json.dumps, rows)), True, reg)
    assert parsed['verified'] and parsed['running']
    assert parsed['updates'] == {'FNO': 1} and parsed['windows'] == {'FNO': 8}
    assert parsed['admission'] is False
    for bad in (rows + [rows[0]], rows[1:], rows + [rows[-1]],
                rows[:-1] + [dict(event='accumulation_update_complete', update=2)]):
        assert not m._parse_registered_progress(state(), '\n'.join(map(json.dumps, bad)), True, reg)['verified']

def test_flow_rejects_validation_and_invalid_plan():
    reg = {**REG, 'progress_kind': 'flow_training', 'planned_updates': {'FNO': 171}}
    row = dict(event='training_window_complete', case='case', start=0, dataset_index=0, split='validation')
    assert not m._parse_registered_progress(state(), json.dumps(row), True, reg)['verified']
    assert not m._parse_registered_progress(state(), '', True, {**reg, 'planned_updates': {'FNO': 170}})['verified']
