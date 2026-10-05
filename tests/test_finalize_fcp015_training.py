import json
import pytest
from finalize_fcp015_training import INVOCATION, persist_or_verify, terminal_state


def terminal():
    return dict(LoadState='loaded', InvocationID=INVOCATION, ActiveState='active',
                SubState='exited', Result='success', ExecMainStatus='0',
                ExecMainCode='1', MainPID='0')


def test_retained_success():
    assert terminal_state(terminal()) is True


def test_running_is_not_completion():
    fields = terminal(); fields.update(SubState='running', MainPID='123')
    assert terminal_state(fields) is False


@pytest.mark.parametrize('key,value', [
    ('LoadState', 'not-found'), ('InvocationID', 'other'),
    ('ActiveState', 'inactive'), ('SubState', 'failed'), ('Result', 'timeout'),
    ('ExecMainStatus', '1'), ('ExecMainCode', '2'), ('MainPID', '123'),
])
def test_reject_wrong_terminal(key, value):
    fields = terminal(); fields[key] = value
    with pytest.raises(ValueError): terminal_state(fields)


def test_reject_running_no_process():
    fields = terminal(); fields['SubState'] = 'running'
    with pytest.raises(ValueError): terminal_state(fields)


def test_receipt_never_overwrites(tmp_path):
    path = tmp_path / 'receipt.json'
    persist_or_verify(path, {'status': 'test'})
    persist_or_verify(path, {'status': 'test'})
    with pytest.raises(ValueError): persist_or_verify(path, {'status': 'other'})
    assert json.loads(path.read_text()) == {'status': 'test'}
