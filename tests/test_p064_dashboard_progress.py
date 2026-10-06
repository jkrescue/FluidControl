import hashlib
import json
import pytest
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE if (HERE/'p064_dashboard_progress.py').exists() else HERE.parent/'scripts'))
from p064_dashboard_progress import parse_journal,status,PROFILES

def events():
    return '\n'.join(json.dumps({'MESSAGE':json.dumps(m),'__REALTIME_TIMESTAMP':'1791281947000000'}) for m in [
        {'event':'training_window_complete','history_k':1,'consumed':8},
        {'event':'accumulation_update_complete','history_k':1,'update':1}])

def test_parser():
    result=parse_journal(events());assert result['windows']==8 and result['updates']==1
    assert result['last_update_utc'].endswith('+00:00')

@pytest.mark.parametrize('field,value',[('history_k',True),('consumed',257),('consumed',False)])
def test_bad_event(field,value):
    row={'event':'training_window_complete','history_k':1,'consumed':8};row[field]=value
    with pytest.raises(ValueError):parse_journal(json.dumps({'MESSAGE':json.dumps(row)}))

def test_pending_never_queries(tmp_path):
    assert status(tmp_path,'A',run=lambda *a,**k:1/0)['status']=='尚未启动'

def test_actual_identity_and_zero_exit_not_success(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'source')
    item={'path':'source','sha256':hashlib.sha256(b'source').hexdigest()}
    reg={**PROFILES['A'],'invocation':'a'*32,'approval':item,'driver':item}
    def running(cmd,**kw):
        return 'InvocationID='+('a'*32)+'\nActiveState=active\nSubState=running\nMainPID=10\nExecMainStatus=0' if cmd[0]=='systemctl' else events()
    assert status(tmp_path,'A',reg,running)['status']=='训练中'
    def ended(cmd,**kw):return running(cmd,**kw).replace('SubState=running','SubState=exited').replace('MainPID=10','MainPID=0')
    out=status(tmp_path,'A',reg,ended);assert '等待独立' in out['status'] and not out['terminal_verified']
    def failed(cmd,**kw):return ended(cmd,**kw).replace('ExecMainStatus=0','ExecMainStatus=1')
    assert status(tmp_path,'A',reg,failed)['status']=='失败/停止（未自动重试）'
    reg['invocation']='b'*32
    assert status(tmp_path,'A',reg,running)['status']=='身份或状态未验证'
def test_development_missing_evidence_fails_closed(tmp_path):
    from p064_dashboard_progress import development_summary
    assert development_summary(tmp_path)['verified'] is False


def test_actual_small_reviewed_development_json():
    from pathlib import Path
    from p064_dashboard_progress import development_summary
    root=Path('/workspace/fluid_control')
    data=development_summary(root)
    assert data['verified'] is True
    assert [row['label'] for row in data['rows']]==['K1','A','B']
    assert data['rows'][2]['h1_cl'] < data['rows'][1]['h1_cl']
    assert data['rows'][2]['h5_cd'] > data['rows'][0]['h5_cd']
    assert '尚未训练' in data['current_stage']
def test_json_tail_ignores_partial_write(tmp_path):
    from p064_dashboard_progress import last_json_row
    p=tmp_path/'events.jsonl'
    p.write_bytes(b'{"num_timesteps":4}\n{"num_timesteps":8')
    assert last_json_row(p)=={'num_timesteps':4}


def test_ppo_missing_approval_is_not_running(tmp_path):
    from p064_dashboard_progress import candidate_ppo_status
    row=candidate_ppo_status(tmp_path)
    assert row['invocation'] is None and row['status']=='身份或状态未验证'
def test_ppo_terminal_producer_not_independent_review():
    import pytest
    from p064_dashboard_progress import producer_terminal_counts
    spec={'candidate_arm':'B','candidate_manifest_sha256':'b'*64}
    result={**spec,'status':'P064_CANDIDATE_DIVERSE_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION',
        'timesteps':32768,'ppo_n_updates':256,'fno_tensors_unchanged':True,
        'optimizer_steps':[{'optimizer_step':i} for i in range(1,513)]}
    row=producer_terminal_counts(result,spec)
    assert row['optimizer_steps']==512 and row['terminal_verified'] is False
    result['optimizer_steps'].pop()
    with pytest.raises(ValueError):producer_terminal_counts(result,spec)
def test_cfd_count_is_not_physical_pass():
    import pytest
    from p064_dashboard_progress import cfd_progress_counts
    row=cfd_progress_counts({'completed_cycles':1,'rows':[{'step':1,'end_time':148.1,'applied_omega':.1}]})
    assert row['cycles']==1 and 'physical_pass' not in row
    with pytest.raises(ValueError):cfd_progress_counts({'completed_cycles':800,'rows':[]})


def test_cfd_unbound_not_running(tmp_path):
    from p064_dashboard_progress import candidate_cfd_status
    row=candidate_cfd_status(tmp_path)
    assert row['invocation'] is None and row['physical_pass'] is None
def test_b01_clock_is_explicit():
    from p064_dashboard_progress import cfd_progress_counts
    row={'completed_cycles':1,'rows':[{'step':1,'end_time':130.1,'applied_omega':.1}]}
    assert cfd_progress_counts(row,start=130.)['current_time']==130.1

def test_formal_missing_identity_does_not_claim_running(tmp_path):
    from p064_dashboard_progress import formal_evaluation_status
    result=formal_evaluation_status(tmp_path)
    assert result['invocation'] is None
    assert result['training'] is False
    assert result['scientific_pass'] is None

def test_verified_formal_terminal_is_scientific_failure(tmp_path,monkeypatch):
    import hashlib,json
    import p064_dashboard_progress as module
    receipt=tmp_path/'artifacts/fcp064_arm_b_formal_resume_r3_20261006/receipt.json'
    review=tmp_path/'docs/P064_B_FORMAL_TERMINAL_REVIEW_20261006.md'
    receipt.parent.mkdir(parents=True);review.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'scientific_admission':False}));review.write_text('independent review')
    monkeypatch.setattr(module,'FORMAL_TERMINAL_BINDINGS',{'receipt':hashlib.sha256(receipt.read_bytes()).hexdigest(),'review':hashlib.sha256(review.read_bytes()).hexdigest()})
    result=module.verified_formal_terminal(tmp_path)
    assert result['terminal_verified'] and result['scientific_pass'] is False
    assert result['running'] is False and '2/6' in result['note']
    assert '未运行' in result['note']
    review.write_text('changed')
    import pytest
    with pytest.raises(ValueError):module.verified_formal_terminal(tmp_path)

def test_b07_exact_progress_clock():
    import pytest
    from p064_dashboard_progress import cfd_progress_counts
    p={'completed_cycles':1,'rows':[{'step':1,'end_time':110.1,'applied_omega':.01}]}
    assert cfd_progress_counts(p,start=110.)['cycles']==1
    with pytest.raises(ValueError):cfd_progress_counts(p,start=130.)

def test_signed_h1_unbound_not_busy_or_training(tmp_path):
    from p064_dashboard_progress import signed_h1_status
    x=signed_h1_status(tmp_path)
    assert not x['running'] and not x['training']
    assert x['invocation'] is None and x['completed_endpoints'] is None

def test_scale_counts_only_actual_scales_train_events():
    import json,pytest
    from p064_dashboard_progress import scale_window_count
    row={'event':'window_complete','mode':'scales','split':'train'}
    assert scale_window_count('loading\n'+json.dumps(row)+'\n'+json.dumps({'event':'group_complete','mode':'scales'}))==1
    assert scale_window_count(json.dumps(dict(row,mode='train')))==0
    with pytest.raises(ValueError):scale_window_count(json.dumps(dict(row,split='validation')))
    with pytest.raises(ValueError):scale_window_count('\n'.join([json.dumps(row)]*1369))

def test_unbound_scales_never_reports_training(tmp_path):
    from p064_dashboard_progress import h25_scales_status
    x=h25_scales_status(tmp_path)
    assert not x['running'] and not x['training'] and x['optimizer_steps']==0

def test_scales_terminal_requires_exact_review_and_result(tmp_path,monkeypatch):
    import hashlib,json,pytest,p064_dashboard_progress as module
    result=tmp_path/'artifacts/fcp064_b_h25_scales_20261006_r2/payload/result.json'
    review=tmp_path/'docs/P064_B_H25_SCALES_R2_TERMINAL_REVIEW_20261006.md'
    result.parent.mkdir(parents=True);review.parent.mkdir(parents=True)
    result.write_text(json.dumps({'training_windows':1368,'optimizer_steps':0,'model_saved':False,'fixed_scales':{'field':.1,'force':.2}}));review.write_text('reviewed')
    monkeypatch.setattr(module,'SCALES_TERMINAL_BINDINGS',{'result':hashlib.sha256(result.read_bytes()).hexdigest(),'review':hashlib.sha256(review.read_bytes()).hexdigest()})
    assert module.verified_scales_terminal(tmp_path)['terminal_verified']
    review.write_text('changed')
    with pytest.raises(ValueError):module.verified_scales_terminal(tmp_path)

def test_h25_training_counts_do_not_count_failed_first_window_as_update():
    import json,pytest
    from p064_dashboard_progress import h25_training_counts
    row={'event':'window_complete','mode':'train','split':'train'}
    assert h25_training_counts(json.dumps(row)+'\nFloatingPointError')==(1,0)
    lines=[json.dumps(row)]*8+[json.dumps({'event':'group_complete','mode':'train','group':1})]
    assert h25_training_counts('\n'.join(lines))==(8,1)
    with pytest.raises(ValueError):h25_training_counts(json.dumps({'event':'group_complete','mode':'train','group':1}))
    with pytest.raises(ValueError):h25_training_counts('\n'.join(lines+[lines[-1]]))

def test_unbound_h25_training_is_not_running(tmp_path):
    from p064_dashboard_progress import h25_training_status
    x=h25_training_status(tmp_path)
    assert not x['running'] and not x['training'] and x['updates']==0 and x['invocation'] is None

def test_unbound_initial_control_not_training_or_physical_pass(tmp_path):
    from p064_dashboard_progress import candidate_cfd_status
    x=candidate_cfd_status(tmp_path)
    assert not x['running'] and not x['training'] and x['physical_pass'] is None
    assert x['invocation'] is None and 'trained_reference' in x
