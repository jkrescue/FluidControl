import pytest
import finalize_fcp018_training as f


def fields(**changes):
    value=dict(LoadState='loaded',InvocationID=f.INVOCATION,ActiveState='active',SubState='exited',
               Result='success',ExecMainStatus='0',ExecMainCode='1',MainPID='0')
    value.update(changes);return value


def test_retained_success_only():
    assert f.terminal_state(fields())
    assert not f.terminal_state(fields(SubState='running',MainPID='123'))


@pytest.mark.parametrize('change',[dict(InvocationID='7842742926284d0c94b0383163d5dc0b'),dict(LoadState='not-found'),
    dict(ActiveState='inactive',SubState='dead'),dict(Result='exit-code'),dict(ExecMainStatus='1'),
    dict(ExecMainCode='2'),dict(MainPID='123'),dict(SubState='running')])
def test_wrong_or_failed_state_rejected(change):
    with pytest.raises(ValueError):f.terminal_state(fields(**change))


def test_receipt_exclusive(tmp_path):
    path=tmp_path/'receipt.json'
    f.persist_or_verify(path,{'status':'FC_P018_TRAINING_COMPLETE_NOT_ADMISSION'})
    f.persist_or_verify(path,{'status':'FC_P018_TRAINING_COMPLETE_NOT_ADMISSION'})
    with pytest.raises(ValueError):f.persist_or_verify(path,{'status':'wrong'})
