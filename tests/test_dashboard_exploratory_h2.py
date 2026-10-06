import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).with_name('serve_live_research_dashboard.py')
if not path.exists():path=Path(__file__).parents[1]/'scripts/serve_live_research_dashboard.py'
spec=importlib.util.spec_from_file_location('mpc_dashboard',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def fixture(tmp_path):
    reg={'unit':'fluid-control-exploratory-short-h2-real-cfd-20261006.service','invocation':'e3b9eb7b58724a1c9ec4e64d63ac7bbe'}
    state={'InvocationID':reg['invocation'],'ActiveState':'active','SubState':'running','MainPID':'42'}
    force={'front_cd':1.4,'rear_cd':1.0,'front_cl':.2,'rear_cl':.3}
    row={'step':1,'start_time':148.,'end_time':148.1,'selected_omega':.05,'previous_omega':0.,'selected_predicted_next_forces':force,'actual_endpoint_forces':{'mpc':force,'zero':force}}
    doc={'status':'EXPLORATORY_REAL_CFD_SHORT_H2_FEEDBACK_RUNNING_NOT_ADMISSION','completed_cycles':1,'identity':{'k1_manifest_sha256':'7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7'},'rows':[row]}
    path=tmp_path/'artifacts/exploratory_paired_h2_real_cfd_20261006/progress.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(doc))
    return tmp_path,reg,state,path,doc

def test_live_actual_metrics(fixture):
    root,reg,state,_,_=fixture;r=m._exploratory_mpc_progress(root,reg,state,True)
    assert r['running'] and r['completed_cycles']==1 and r['latest']['actual_cd']==2.4
    assert not r['scientific_admission']


def test_causal_history_profile_reuses_parser_without_overwriting_old_trial(tmp_path):
    reg={
        'progress_kind':'exploratory_causal_history_h2_feedback',
        'unit':'fluid-control-exploratory-causal-h2-real-cfd-20261006.service',
        'invocation':'6f554f10e87e4b9f9d6b6ed8b555c548',
    }
    state={'InvocationID':reg['invocation'],'ActiveState':'active','SubState':'running','MainPID':'42'}
    force={'front_cd':1.4,'rear_cd':1.0,'front_cl':.2,'rear_cl':.3}
    row={'step':1,'start_time':148.,'end_time':148.1,'selected_omega':0.,'previous_omega':0.,'selected_predicted_next_forces':force,'actual_endpoint_forces':{'mpc':force,'zero':force}}
    doc={'status':'EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H2_RUNNING_NOT_ADMISSION','completed_cycles':1,'identity':{'k1_manifest_sha256':'7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7'},'rows':[row]}
    progress=tmp_path/'artifacts/exploratory_causal_history_h2_real_cfd_20261006/progress.json'
    progress.parent.mkdir(parents=True);progress.write_text(json.dumps(doc))
    result=m._exploratory_mpc_progress(tmp_path,reg,state,True)
    assert result['verified'] and result['running'] and result['completed_cycles']==1
    assert result['progress_kind']=='exploratory_causal_history_h2_feedback'
    assert not result['terminal_review_verified'] and not result['control_success_verified']


def test_causal_profile_rejects_old_unit_and_status(tmp_path):
    reg={'progress_kind':'exploratory_causal_history_h2_feedback','unit':'fluid-control-exploratory-short-h2-real-cfd-20261006.service','invocation':'6f554f10e87e4b9f9d6b6ed8b555c548'}
    state={'InvocationID':reg['invocation'],'ActiveState':'active','SubState':'running','MainPID':'42'}
    assert m._exploratory_mpc_progress(tmp_path,reg,state,True)=={'verified':False}


def test_h5_live_profile_uses_separate_output_and_never_claims_success(tmp_path):
    reg={'progress_kind':'exploratory_causal_history_h5_feedback','unit':'fluid-control-exploratory-causal-h5-real-cfd-20261006.service','invocation':'6adc59fae65344d2b49b57cbe5b30f70'}
    state={'InvocationID':reg['invocation'],'ActiveState':'active','SubState':'running','MainPID':'42'}
    force={'front_cd':1.4,'rear_cd':1.0,'front_cl':.2,'rear_cl':.3}
    row={'step':1,'start_time':148.,'end_time':148.1,'selected_omega':0.,'previous_omega':0.,'selected_predicted_next_forces':force,'actual_endpoint_forces':{'mpc':force,'zero':force}}
    doc={'status':'EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H5_RUNNING_NOT_ADMISSION','completed_cycles':1,'identity':{'k1_manifest_sha256':'7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7'},'rows':[row]}
    progress=tmp_path/'artifacts/exploratory_causal_history_h5_real_cfd_20261006/progress.json'
    progress.parent.mkdir(parents=True);progress.write_text(json.dumps(doc))
    result=m._exploratory_mpc_progress(tmp_path,reg,state,True)
    assert result['verified'] and result['running'] and result['completed_cycles']==1
    assert result['progress_kind']=='exploratory_causal_history_h5_feedback'
    assert not result['terminal_review_verified'] and not result['control_success_verified']
    assert '真实CFD因果历史H5短时控制试验' in m.PAGE


def test_causal_terminal_binding_is_distinct_and_nonadmitting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    base=tmp_path/'artifacts/exploratory_causal_history_h2_real_cfd_20261006';base.mkdir(parents=True)
    docs=tmp_path/'docs';docs.mkdir()
    result={'status':'EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H2_COMPLETE_NOT_ADMISSION','selector_mode':'canonical_causal_history_h2_v1','cycles':10,'physical_duration_D_over_U':1.0,'scientific_admission':False,'ppo_executed':False,'hydrogym_solver_used':False,'original_long_ar_gate_passed':False,'source_restart_unchanged':True,'rows':[{'selected_omega':0.0} for _ in range(10)]}
    rows=[(docs/'EXPLORATORY_CAUSAL_HISTORY_H2_TERMINAL_REVIEW_20261006.md','review','9ceb4d58a66b549faa86834d15d0444d57c8fdcc2f2635f18859440a679d2098'),(base/'result.json',json.dumps(result),'74a28d45dce9b84ec5044700fe470390cde899a2fcf40a0b893c1b28817d99ca'),(base/'container_terminal_e59efa04750c.json','terminal1','6ab3d0319b4e4f04bb3498b0e302533f6628aec5603d4b2bb3fe010b5831ccd5'),(base/'container_terminal_e610fafa3753.json','terminal2','b0a8bb934ef1c222bb318e4a92ac8ff50b39e96ab6648b3ca0f67d148cc12b97')]
    hashes={}
    for path,text,digest in rows:path.write_text(text);hashes[text.encode()]=digest
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'invalid')))
    assert m._exploratory_mpc_terminal_review(tmp_path,'causal_history_h2') is True
    assert m._exploratory_mpc_terminal_review(tmp_path,'unknown') is False
    assert '10/10因果历史H2真实闭环完成；全部HOLD、配对收益为零' in m.PAGE

@pytest.mark.parametrize('bad',['invocation','pid','count','action','nan'])
def test_bad_evidence(fixture,bad):
    root,reg,state,path,doc=fixture;matches=True
    if bad=='invocation':state['InvocationID']='wrong'
    elif bad=='pid':matches=False
    elif bad=='count':doc['completed_cycles']=2
    elif bad=='action':doc['rows'][0]['selected_omega']=.2
    else:doc['rows'][0]['selected_omega']=float('nan')
    path.write_text(json.dumps(doc))
    assert m._exploratory_mpc_progress(root,reg,state,matches)=={'verified':False}

def test_exit_does_not_claim_completed_control(fixture):
    root,reg,state,_,_=fixture
    state.update(MainPID='0',SubState='exited',Result='success',ExecMainCode='1',ExecMainStatus='0')
    r=m._exploratory_mpc_progress(root,reg,state,False)
    assert r['terminal_review_pending'] and not r['control_success_verified'] and not r['running']


def test_missing_independent_report_rejected(tmp_path):
    assert m._exploratory_mpc_terminal_review(tmp_path) is False


def test_terminal_binding_requires_all_pinned_files(tmp_path,monkeypatch):
    from types import SimpleNamespace
    base=tmp_path/'artifacts/exploratory_paired_h2_real_cfd_20261006';base.mkdir(parents=True)
    docs=tmp_path/'docs';docs.mkdir()
    result={'status':'EXPLORATORY_REAL_CFD_SHORT_H2_FEEDBACK_COMPLETE_NOT_ADMISSION','cycles':10,'physical_duration_D_over_U':1.0,'scientific_admission':False,'ppo_executed':False,'hydrogym_solver_used':False,'original_long_ar_gate_passed':False,'source_restart_unchanged':True}
    rows=[(docs/'EXPLORATORY_PAIRED_H2_TERMINAL_REVIEW_20261006.md','review','8c600368d836e8c34c09e4ac3be1129ffcfb67fd3587e48e4e4feba33d711dcd'),(base/'result.json',json.dumps(result),'45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb'),(base/'container_terminal_257e7829da44.json','terminal1','d0036b3f7fd0fbc66b28ee8c7111a0951f25d045be41e842fb894e90e7f5fc52'),(base/'container_terminal_e49a8a5a3542.json','terminal2','cfe5e053c8bd8fac787607844d18ded16d4b823b5ead3106a6197080fab64819')]
    hashes={}
    for path,text,digest in rows:path.write_text(text);hashes[text.encode()]=digest
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'invalid')))
    assert m._exploratory_mpc_terminal_review(tmp_path) is True
    rows[-1][0].write_text('changed')
    assert m._exploratory_mpc_terminal_review(tmp_path) is False
