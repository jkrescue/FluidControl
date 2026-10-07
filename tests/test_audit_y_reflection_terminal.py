import copy
import importlib.util
from pathlib import Path

import pytest

CHECKER = Path(__file__).with_name('audit_y_reflection_terminal.py')
if not CHECKER.is_file():
    CHECKER = Path(__file__).resolve().parents[1] / 'scripts/audit_y_reflection_terminal.py'
spec = importlib.util.spec_from_file_location('audit_reflection', CHECKER)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture():
    result = {'records': []}
    events = []
    for group in range(32):
        rows = []
        for window in range(8):
            i = group*8+window+1
            branches = {}
            for offset, label in enumerate(('original', 'reflected'), 1):
                branch = dict(identity={'split': 'train', 'start': i}, history={'k': 1},
                    reflection_branch=label, flow_calls=100, aerodynamic_calls=10,
                    flow_history_sha256='b'*64,
                    branch_input_sha256={key:'a'*64 for key in ('state','target_state','omega','target_force','mask')},
                    h1_balanced=float(offset), ar_balanced=3.*offset, total=2.*offset)
                branches[label] = branch
                events.append(dict(event='training_reflection_branch_complete', branch=label,
                    consumed_original_windows=i, completed_transformed_branches=2*(i-1)+offset,
                    flow_calls=100, aerodynamic_calls=10, flow_history_sha256=branch['flow_history_sha256'],
                    branch_input_sha256=copy.deepcopy(branch['branch_input_sha256'])))
            rows.append(dict(branches=branches,h1_balanced=1.5,ar_balanced=4.5,total=3.))
            events.append(dict(event='training_window_complete',completed_transformed_branches=2*i))
        result['records'].append({'records':rows})
        events.append({'event':'accumulation_update_complete'})
    return result, events


def test_full_512_pair_binding_and_no_mutation():
    result, events = fixture()
    projected = a.paired_records(result, events)
    assert projected['records'][0]['records'][0]['identity']['start'] == 1
    assert 'identity' not in result['records'][0]['records'][0]


@pytest.mark.parametrize('failure', ['weight','calls','mask','order','missing','sha'])
def test_pair_fail_closed(failure):
    result, events = fixture()
    row = result['records'][0]['records'][0]
    if failure == 'weight': row['total'] += .1
    if failure == 'calls': row['branches']['reflected']['flow_calls'] = 99
    if failure == 'mask': row['branches']['reflected']['branch_input_sha256']['mask'] = 'c'*64
    if failure == 'order': events[0], events[1] = events[1], events[0]
    if failure == 'missing': events.pop(0)
    if failure == 'sha': events[0]['flow_history_sha256'] = 'd'*64
    with pytest.raises(ValueError): a.paired_records(result, events)


def test_live_unit_rejected_before_candidate():
    with pytest.raises(ValueError, match='still running'):
        a.b.terminal({'InvocationID':'fixed','MainPID':'12'}, 'fixed')


@pytest.mark.parametrize('failure', [None, 'live', 'receipt', 'journal', 'manager'])
def test_collected_transient_requires_all_evidence(failure):
    props = {'LoadState':'not-found','MainPID':'0'}
    receipt = {'status':'P064_Y_REFLECTION_PAIRED_SUPERVISOR_COMPLETE','invocation_id':'fixed','unit':'test.service'}
    journal = [{'_SYSTEMD_INVOCATION_ID':'fixed','__REALTIME_TIMESTAMP':'10'},
               {'USER_UNIT':'test.service','MESSAGE_ID':'ae8f7b866b0347b9af31fe1c80b127c0',
                'MESSAGE':'test.service: Consumed 1s CPU time.', '__REALTIME_TIMESTAMP':'11'}]
    if failure == 'live': props['MainPID'] = '9'
    if failure == 'receipt': receipt['invocation_id'] = 'other'
    if failure == 'journal': journal.pop(0)
    if failure == 'manager': journal.pop()
    if failure is None:
        evidence = a.reclaimed_terminal(props,'fixed','test.service',receipt,journal)
        assert not evidence['retained_unit_limits_available']
        assert 'InvocationID' not in evidence['observed_systemctl']
    else:
        with pytest.raises(ValueError):
            a.reclaimed_terminal(props,'fixed','test.service',receipt,journal)
