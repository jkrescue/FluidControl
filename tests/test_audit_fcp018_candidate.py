import copy

import pytest
import torch

import audit_fcp018_candidate as a
import audit_fcp015_candidate as old
from test_audit_fcp015_candidate import optimizer_state,records


def state():
    value=optimizer_state();value['optimizer_state_dict']['param_groups'][0]['lr']=a.LR
    return value


def groups():
    value=records()
    for row in value:row.update(actual_learning_rate=a.LR,training_protocol_sha256=a.PROTOCOL_SHA)
    return value


def test_correct_optimizer_and_legacy_unchanged():
    a.validate_optimizer(state())
    old.validate_optimizer(optimizer_state())
    with pytest.raises(ValueError):a.validate_optimizer(optimizer_state())
    with pytest.raises(ValueError):old.validate_optimizer(state())
    assert old.EXPERIMENT['training_experiment']=='FC-P015'


@pytest.mark.parametrize('mutation',['step','missing','nan','negative','oldrate','count'])
def test_reject_saved_optimizer_mismatch(mutation):
    value=state();opt=value['optimizer_state_dict']
    if mutation=='step':opt['state'][0]['step']=torch.tensor(1368.)
    if mutation=='missing':del opt['state'][0]
    if mutation=='nan':opt['state'][0]['exp_avg'][0]=float('nan')
    if mutation=='negative':opt['state'][0]['exp_avg_sq'][0]=-1
    if mutation=='oldrate':opt['param_groups'][0]['lr']=1e-5
    if mutation=='count':opt['param_groups'][0]['params'].pop()
    with pytest.raises(ValueError):a.validate_optimizer(value)


def test_records_lr_protocol_and_neutral_order():
    value=groups()
    assert len(a.validate_records(value))==1368
    assert len(old.validate_records(records()))==1368
    value[100]['actual_learning_rate']=1e-5
    with pytest.raises(ValueError):a.validate_records(value)
    value=groups();value[170]['training_protocol_sha256']='0'*64
    with pytest.raises(ValueError):a.validate_records(value)


def test_execution_external_pin_before_read(tmp_path):
    with pytest.raises(ValueError,match='execution pin'):
        a.validate_execution(tmp_path,'a'*64,a.OBSERVATION_SHA)


def test_missing_candidate_no_cross_profile(tmp_path):
    with pytest.raises(ValueError,match='root differs'):
        a.validate_candidate(tmp_path,tmp_path/'artifacts/fcp015_window_accumulation_training_20261005',
                             approval_sha=a.APPROVAL_SHA,observation_sha=a.OBSERVATION_SHA)


def test_result_rejects_p015_identity():
    with pytest.raises(ValueError):
        a.validate_training_result({'status':'FC_P015_WINDOW_ACCUMULATION_TRAINING_COMPLETE_NOT_ADMISSION'},'0'*64)


def test_optimizer_boolean_epoch_or_parameter_id_rejected():
    value=state();value['epoch']=True
    with pytest.raises(ValueError):a.validate_optimizer(value)
    value=state();value['optimizer_state_dict']['param_groups'][0]['params'][1]=True
    with pytest.raises(ValueError):a.validate_optimizer(value)
