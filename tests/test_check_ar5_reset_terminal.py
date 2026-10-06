import copy
import importlib.util
from pathlib import Path
import pytest

checker_path=Path(__file__).with_name('check_ar5_reset_terminal.py')
if not checker_path.exists():
    checker_path=Path(__file__).resolve().parents[1]/'scripts'/'check_ar5_reset_terminal.py'
spec=importlib.util.spec_from_file_location('checker',checker_path)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def fixture():
    p=dict(training_experiment='FC-P064-AR5-RESET',arm='B',candidate_profile='FC_P064_AR5_RESET_K1_FRESH',
        objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',training_ar_reset_every=5,
        training_ar_reset_indices=list(range(0,100,5)),training_flow_forward_calls_per_window=100,
        training_ar_supervised_points=100,diagnostic_ar_reset_every=None,diagnostic_ar_horizon=100,
        training_windows=256,optimizer_steps=32,accumulation_windows=8,b00_windows=64,b00_weight=.25,
        replacement_within_each_update=[0,4],validation_accessed=False,frozen_test_accessed=False,selection_performed=False)
    profile=dict(profile='training_ar_reset_every_5_true_current_states',reset_indices=list(range(0,100,5)),
        flow_forward_calls=100,supervised_ar_points=100,discarded_block_end_updates=20,
        future_state_inputs=False,optimizer_steps=0)
    records=[];audit=[]
    for group in range(32):
        rows=[]
        for j in range(8):
            ident=dict(index=group*8+j)
            rows.append(dict(identity=ident,flow_history_sha256='a'*64,ar_state_profile=copy.deepcopy(profile),
                h1_balanced=2.,ar_balanced=6.,total=4.))
            audit.append(dict(identity=ident,flow_history_sha256='a'*64,**copy.deepcopy(profile)))
        records.append(dict(records=rows))
    diagnostic=dict(profile='diagnostic_continuous_ar100',reset_indices=[0],flow_forward_calls=100,supervised_ar_points=100)
    panels=[dict(rows=[dict(panel=dict(objective=dict(ar_state_profile=copy.deepcopy(diagnostic)))) for _ in range(6)]) for _ in range(2)]
    return dict(records=records,training_state_audit=audit,fixed_train_panels=panels),p,dict(kind='FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO',training_experiment='FC-P064-AR5-RESET')


def test_original_mixed_objective_and_continuous_diagnostic():
    e=c.check_reset_objective(*fixture())
    assert e['checked_windows']==256 and e['mean_saved_training_objective']==4
    assert e['reset_indices_per_window']==20 and e['flow_calls_per_training_window']==100
    assert not e['independent_training_loss_recomputed']


@pytest.mark.parametrize('key,value',[('reset_indices',[0]),('flow_forward_calls',80),('future_state_inputs',True),('optimizer_steps',1)])
def test_reset_contract_drift(key,value):
    r,p,m=fixture();r['training_state_audit'][0][key]=value
    with pytest.raises(ValueError):c.check_reset_objective(r,p,m)


def test_record_pairing():
    r,p,m=fixture();r['training_state_audit'][0]['identity']={'index':3}
    with pytest.raises(ValueError):c.check_reset_objective(r,p,m)


def test_diagnostic_cannot_use_training_resets():
    r,p,m=fixture();r['fixed_train_panels'][1]['rows'][0]['panel']['objective']['ar_state_profile']['reset_indices']=list(range(0,100,5))
    with pytest.raises(ValueError):c.check_reset_objective(r,p,m)


def test_h1_only_objective_rejected():
    r,p,m=fixture();r['records'][0]['records'][0]['total']=2.
    with pytest.raises(ValueError):c.check_reset_objective(r,p,m)


def test_full_path_and_live_gate_preserved():
    capture={};values={'/approval.json':{'planned_output':'/candidate'},'/candidate/dual_model_manifest.json':{'candidate':True},'/parent/dual_model_manifest.json':{}}
    reader=c.capture_read(lambda p:values[p],'/approval.json',capture)
    for path in values:reader(path)
    assert capture['dual_model_manifest.json']['candidate']
    with pytest.raises(ValueError):c.base.terminal({'InvocationID':'a','MainPID':'1'},'a')


def test_actual_approval_omits_redundant_protocol_keys():
    r,p,m=fixture()
    approval={'protocol':{k:p[k] for k in ('training_experiment','objective','training_ar_reset_every',
        'training_ar_reset_indices','training_flow_forward_calls_per_window','diagnostic_ar_reset_every','diagnostic_ar_horizon')}}
    c.check_approval_protocol(p,approval)
    p['training_ar_supervised_points']=99
    with pytest.raises(ValueError):c.check_reset_objective(r,p,m)
    p['training_ar_supervised_points']=100
    approval['protocol']['training_flow_forward_calls_per_window']=99
    with pytest.raises(ValueError):c.check_approval_protocol(p,approval)
