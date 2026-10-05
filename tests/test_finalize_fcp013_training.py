"""Terminal-unit and immutable receipt software fixtures only."""
import pytest
from finalize_fcp013_training import INVOCATION, persist_or_verify, terminal_state


def unit(**changes):
    value = dict(LoadState='loaded', InvocationID=INVOCATION, ActiveState='inactive',
                 SubState='dead', Result='success', ExecMainStatus='0', ExecMainCode='1')
    value.update(changes)
    return value


def test_active_success_fields_do_not_mean_completed():
    assert not terminal_state(unit(ActiveState='active', SubState='running'))
    assert terminal_state(unit())


@pytest.mark.parametrize('changes', [dict(LoadState='not-found'), dict(InvocationID='old'),
    dict(ActiveState='failed'), dict(Result='timeout'), dict(ExecMainStatus='137'),
    dict(ExecMainCode='2'), dict(SubState='running')])
def test_terminal_fault_rejected(changes):
    with pytest.raises(ValueError):
        terminal_state(unit(**changes))


def test_receipt_is_idempotent_but_never_overwritten(tmp_path):
    path = tmp_path / 'receipt.json'
    persist_or_verify(path, {'value': 1})
    persist_or_verify(path, {'value': 1})
    with pytest.raises(ValueError):
        persist_or_verify(path, {'value': 2})
