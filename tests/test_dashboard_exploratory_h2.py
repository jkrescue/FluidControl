import importlib.util
import hashlib
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


def test_accelerated_long_h5_profile_reports_124_gpu_inference_cycles(tmp_path):
    reg={'progress_kind':'exploratory_accelerated_long_h5_feedback',
         'unit':'fluid-control-accelerated-long-h5-20261006.service',
         'invocation':'a601eec2da7649b4af6f9354a4deb470'}
    state={'InvocationID':reg['invocation'],'ActiveState':'active',
           'SubState':'running','MainPID':'42'}
    force={'front_cd':1.4,'rear_cd':1.0,'front_cl':.2,'rear_cl':.3}
    row={'step':1,'start_time':148.,'end_time':148.1,'selected_omega':.1,
         'previous_omega':0.,'selected_predicted_next_forces':force,
         'actual_endpoint_forces':{'mpc':force,'zero':force}}
    doc={'status':'EXPLORATORY_ACCELERATED_LONG_H5_RUNNING_NOT_ADMISSION',
         'completed_cycles':1,
         'identity':{'k1_manifest_sha256':'7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7'},
         'rows':[row]}
    progress=tmp_path/'artifacts/exploratory_accelerated_long_h5_real_cfd_20261006/progress.json'
    progress.parent.mkdir(parents=True);progress.write_text(json.dumps(doc))
    result=m._exploratory_mpc_progress(tmp_path,reg,state,True)
    assert result['verified'] and result['running']
    assert result['completed_cycles']==1 and result['planned_cycles']==124
    assert result['planned_duration_D_over_U']==12.4
    assert not result['scientific_admission'] and not result['control_success_verified']
    assert 'GPU运行官方FNO推理选动作' in m.PAGE


def test_current_trial_field_is_sha_time_shape_bound_and_actual_only(tmp_path):
    np = pytest.importorskip('numpy')
    profile=m._EXPLORATORY_MPC_PROFILES['exploratory_accelerated_long_h5_feedback']
    base=tmp_path/profile['base'];base.mkdir(parents=True)
    path=base/'current_mpc_148.0.npz'
    state=np.zeros((3,128,256),dtype=np.float32);state[0]=1.;state[2]=.25
    mask=np.ones((1,128,256),dtype=np.uint8);mask[:,:,0]=0
    np.savez(path,state=state,mask=mask,time=np.array([148.],dtype=np.float64),
             x=np.linspace(8,25,256,dtype=np.float32),
             y=np.linspace(4,11,128,dtype=np.float32))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[{'step':1,'start_time':148.,'end_time':148.1,
           'current_sample_sha256':{'mpc':digest}}]
    evidence=m._current_trial_field_evidence(tmp_path,profile,rows)
    assert evidence['verified'] and evidence['sha256']==digest
    assert evidence['field_time']==148. and evidence['force_time']==148.1
    assert evidence['quantity_semantics']['pressure'].startswith('CFD pressure with ROI mean removed')
    png=m._current_trial_field_png(tmp_path,profile,rows,digest)
    assert png.startswith(b'\x89PNG\r\n\x1a\n')
    assert '不是FNO预测场' in m.PAGE


def test_current_trial_field_rejects_hash_time_and_non_long_profile(tmp_path):
    np = pytest.importorskip('numpy')
    profile=m._EXPLORATORY_MPC_PROFILES['exploratory_accelerated_long_h5_feedback']
    base=tmp_path/profile['base'];base.mkdir(parents=True)
    path=base/'current_mpc_148.0.npz'
    np.savez(path,state=np.zeros((3,128,256),dtype=np.float32),
             mask=np.ones((1,128,256),dtype=np.uint8),
             time=np.array([148.1]),x=np.linspace(8,25,256,dtype=np.float32),
             y=np.linspace(4,11,128,dtype=np.float32))
    rows=[{'step':1,'start_time':148.,'end_time':148.1,
           'current_sample_sha256':{'mpc':hashlib.sha256(path.read_bytes()).hexdigest()}}]
    assert m._current_trial_field_evidence(tmp_path,profile,rows) is None
    rows[0]['current_sample_sha256']['mpc']='0'*64
    assert m._current_trial_field_evidence(tmp_path,profile,rows) is None
    assert m._current_trial_field_evidence(tmp_path,
        m._EXPLORATORY_MPC_PROFILES['exploratory_causal_history_h5_feedback'],rows) is None


def test_long_timeseries_labels_start_field_and_end_force_separately(tmp_path):
    assert '流场样本在周期起点 t，受力在终点 t+0.1' in m.PAGE
    assert '无预测值' in m.PAGE
    assert '不是完整求解域' in m.PAGE


def test_requested_field_sha_resolves_completed_row_not_latest(tmp_path):
    np = pytest.importorskip('numpy')
    profile=m._EXPLORATORY_MPC_PROFILES['exploratory_accelerated_long_h5_feedback']
    base=tmp_path/profile['base'];base.mkdir(parents=True)
    rows=[]
    for step,start in ((1,148.0),(2,148.1)):
        path=base/f'current_mpc_{start:.1f}.npz'
        state=np.full((3,128,256),step,dtype=np.float32)
        np.savez(path,state=state,mask=np.ones((1,128,256),dtype=np.uint8),
                 time=np.array([start],dtype=np.float64),
                 x=np.linspace(8,25,256,dtype=np.float32),
                 y=np.linspace(4,11,128,dtype=np.float32))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'step':step,'start_time':start,'end_time':start+.1,
                     'current_sample_sha256':{'mpc':digest}})
    first_sha=rows[0]['current_sample_sha256']['mpc']
    evidence=m._current_trial_field_evidence(tmp_path,profile,rows,first_sha)
    assert evidence['field_time']==148.0 and evidence['step']==1
    assert m._current_trial_field_png(tmp_path,profile,rows,first_sha).startswith(b'\x89PNG')
    assert m._current_trial_field_evidence(tmp_path,profile,rows,'f'*64) is None
    assert m._current_trial_field_png(tmp_path,profile,rows,'f'*64) is None
    assert m._current_trial_field_evidence(tmp_path,profile,rows*63,first_sha) is None


def test_current_field_accepts_existing_float32_time_grid_representation(tmp_path):
    np = pytest.importorskip('numpy')
    profile=m._EXPLORATORY_MPC_PROFILES['exploratory_accelerated_long_h5_feedback']
    base=tmp_path/profile['base'];base.mkdir(parents=True)
    path=base/'current_mpc_160.3.npz'
    np.savez(path,state=np.zeros((3,128,256),dtype=np.float32),
             mask=np.ones((1,128,256),dtype=np.uint8),
             time=np.array([np.float32(160.3)],dtype=np.float64),
             x=np.linspace(8,25,256,dtype=np.float32),
             y=np.linspace(4,11,128,dtype=np.float32))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[{'step':124,'start_time':160.3,'end_time':160.4,
           'current_sample_sha256':{'mpc':digest}}]
    evidence=m._current_trial_field_evidence(tmp_path,profile,rows,digest)
    assert evidence['field_time']==160.3
    assert evidence['stored_sample_time']==float(np.float32(160.3))
    assert evidence['time_tolerance'] >= abs(evidence['stored_sample_time']-160.3)


def test_long_solver_cycles_complete_postprocessing_failure_is_not_success(tmp_path):
    reg={'progress_kind':'exploratory_accelerated_long_h5_feedback',
         'unit':'fluid-control-accelerated-long-h5-20261006.service',
         'invocation':'a601eec2da7649b4af6f9354a4deb470'}
    state={'InvocationID':reg['invocation'],'ActiveState':'failed','SubState':'failed',
           'MainPID':'0','Result':'exit-code','ExecMainCode':'1','ExecMainStatus':'1'}
    force={'front_cd':1.4,'rear_cd':1.0,'front_cl':.2,'rear_cl':.3}
    rows=[]
    for step in range(1,125):
        start=148.+(step-1)*.1
        rows.append({'step':step,'start_time':start,'end_time':148.+step*.1,
                     'selected_omega':0.,'previous_omega':0.,
                     'selected_predicted_next_forces':force,
                     'actual_endpoint_forces':{'mpc':force,'zero':force}})
    doc={'status':'EXPLORATORY_ACCELERATED_LONG_H5_RUNNING_NOT_ADMISSION',
         'completed_cycles':124,
         'identity':{'k1_manifest_sha256':'7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7'},
         'rows':rows}
    progress=tmp_path/'artifacts/exploratory_accelerated_long_h5_real_cfd_20261006/progress.json'
    progress.parent.mkdir(parents=True);progress.write_text(json.dumps(doc))
    result=m._exploratory_mpc_progress(tmp_path,reg,state,False)
    assert result['verified'] and result['solver_cycles_complete']
    assert result['postprocessing_failed'] and not result['exited_success']
    assert not result['terminal_review_verified'] and not result['control_success_verified']
    assert '统计后处理失败，等待恢复复核' in m.PAGE


def test_recovered_metrics_preserve_original_exit1_and_percent_units(tmp_path,monkeypatch):
    from types import SimpleNamespace
    base=tmp_path/'artifacts/exploratory_accelerated_long_h5_real_cfd_20261006';base.mkdir(parents=True)
    docs=tmp_path/'docs';docs.mkdir()
    def window(interval,samples,drag,ratio,bias,mean):
        return {'interval_open_left_closed_right':interval,'branches':{
            'mpc':{'samples':samples,'total_cd_mean':2.31,'rear_cl_mean':mean,
                   'rear_cl_fluctuation_rms':.98},
            'zero':{'samples':samples,'total_cd_mean':2.30,'rear_cl_mean':0.,
                    'rear_cl_fluctuation_rms':1.18}},
            'paired_drag_reduction':drag,
            'paired_rear_cl_fluctuation_rms_ratio':ratio,
            'absolute_mean_rear_cl_over_paired_zero_rms':bias}
    payload={'status':'OFFLINE_METRICS_RECOVERED_FROM_FAILED_POSTPROCESSING_NOT_ADMISSION',
             'cycles':124,'scientific_admission':False,'new_cfd_or_model_execution':False,
             'original_result_written':False,'original_restart_rehashed_unchanged':True,
             'original_unit':{'InvocationID':'a601eec2da7649b4af6f9354a4deb470',
                              'Result':'exit-code','ExecMainStatus':'1'},
             'windows':{'full':window([148.,160.4],2480,-.0065,.832,.0007,-.0009),
                        'first_6p2':window([148.,154.2],1240,.041,.914,.173,.203),
                        'trailing_6p2':window([154.2,160.4],1240,-.054,.700,.174,-.205)}}
    metrics=base/'recovered_metrics.json';metrics.write_text(json.dumps(payload))
    review=docs/'EXPLORATORY_ACCELERATED_LONG_H5_TERMINAL_REVIEW_20261006.md';review.write_text('review')
    hashes={metrics.read_bytes():m._LONG_H5_RECOVERED_SHA,review.read_bytes():m._LONG_H5_REVIEW_SHA}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    recovered=m._long_h5_recovered_metrics(tmp_path)
    assert recovered['verified'] and recovered['original_unit_exit_status']==1
    assert recovered['scientific_admission'] is False and recovered['original_result_written'] is False
    assert recovered['windows'][0]['drag_reduction_percent']==pytest.approx(-.65)
    assert recovered['windows'][2]['rear_cl_rms_change_percent']==pytest.approx(-30.)
    assert '124周期CFD完成；后处理已恢复，整体未减阻' in m.PAGE


def test_exploratory_h5_ppo_terminal_card_binds_complete_jsonl_and_frozen_fno(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    payload=tmp_path/'artifacts/exploratory_h5_ppo_training_20261006/payload'
    payload.mkdir(parents=True);docs=tmp_path/'docs';docs.mkdir()
    approval=docs/'EXPLORATORY_H5_PPO_APPROVAL_20261006.json';approval.write_text('approval')
    artifact_text={'ppo_final.zip':'policy','vecnormalize.pkl':'vec','source_spec.json':'approval'}
    for name,text in artifact_text.items():(payload/name).write_text(text)
    progress=[]
    for i in range(8):
        progress.append({'time/total_timesteps':512*(i+1),'rollout/ep_rew_mean':-.54})
    progress.append({'train/value_loss':.021034008590504527,
                     'train/approx_kl':.010616972111165524})
    progress_bytes=b''.join(json.dumps(row).encode()+b'\n' for row in progress)
    (payload/'progress.json').write_bytes(progress_bytes)
    transitions=[]
    for step in range(4096):
        transitions.append(json.dumps({'env_index':step%4,'scientific_admission':False,
                                       'TimeLimit.truncated':step<816}).encode()+b'\n')
    transitions_bytes=b''.join(transitions)+b'{incomplete'
    (payload/'transitions.jsonl').write_bytes(transitions_bytes)
    artifact_hashes={'ppo_final.zip':'policy-sha','vecnormalize.pkl':'vec-sha',
                     'source_spec.json':m._H5_PPO_APPROVAL_SHA,
                     'progress.json':'progress-sha','transitions.jsonl':'transitions-sha'}
    result={'status':'EXPLORATORY_H5_PPO_TRAINING_COMPLETE_NOT_ADMISSION',
            'timesteps':4096,'scientific_admission':False,'cfd_executed':False,
            'fno_tensors_unchanged':True,'policy_tensor_sha256_before':'before',
            'policy_tensor_sha256_after':'after','ppo_n_updates':32,
            'optimizer_steps':[{} for _ in range(64)],
            'diagnostics':{'episodes_completed':816,
                           'episode_return':{'mean':-.5444709350490196}},
            'artifacts':artifact_hashes}
    result_path=payload/'result.json';result_path.write_text(json.dumps(result))
    supervisor={'returncode':0,'result_sha256':m._H5_PPO_RESULT_SHA,
                'scientific_admission':False,'minimum_available_bytes':119542509568}
    supervisor_path=payload.parent/'supervisor_result.json'
    supervisor_path.write_text(json.dumps(supervisor))
    hashes={approval.read_bytes():m._H5_PPO_APPROVAL_SHA,
            result_path.read_bytes():m._H5_PPO_RESULT_SHA,
            supervisor_path.read_bytes():m._H5_PPO_SUPERVISOR_SHA,
            b'policy':'policy-sha',b'vec':'vec-sha',b'approval':m._H5_PPO_APPROVAL_SHA,
            progress_bytes:'progress-sha',transitions_bytes:'transitions-sha'}
    monkeypatch.setattr(m.hashlib,'sha256',
        lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=21cb82da66214924b38f120eb30723e5',
        'MainPID=0','ActiveState=active','SubState=exited','Result=success',
        'ExecMainCode=1','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    card=m._exploratory_h5_ppo_training(tmp_path)
    assert card['verified'] and card['training_complete'] and not card['running']
    assert card['timesteps']==4096 and card['environment_counts']=={0:1024,1:1024,2:1024,3:1024}
    assert card['optimizer_steps']==64 and card['ppo_updates']==32
    assert card['value_loss']==pytest.approx(.021034008590504527)
    assert card['approx_kl']==pytest.approx(.010616972111165524)
    assert card['episode_reward_mean']==pytest.approx(-.5444709350490196)
    assert card['minimum_available_gib']==pytest.approx(111.33263778686523)
    assert card['policy_changed'] and card['fno_tensors_unchanged']
    assert not card['cfd_executed'] and not card['scientific_admission']
    assert '训练完成，等待真实 CFD 配对验证' in m.PAGE
    assert '没有执行真实CFD' in m.PAGE


def test_final_ppo_real_cfd_live_card_uses_authentic_force_channels_and_no_mpc_field(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    docs=tmp_path/'docs';docs.mkdir()
    approval={'status':'EXPLORATORY_FINAL_PPO_REAL_CFD_EXECUTION_APPROVED',
              'execution_authorized':True,'steps':124,'inference_device':'cpu',
              'scientific_admission':False,'driver_sha256':m._FINAL_PPO_CFD_DRIVER_SHA}
    approval_path=docs/'EXPLORATORY_FINAL_PPO_CFD_APPROVAL_20261006.json'
    approval_path.write_text(json.dumps(approval))
    driver=tmp_path/'artifacts/exploratory_final_ppo_cfd_source_20261006_immutable/run_exploratory_ppo_real_cfd.py'
    driver.parent.mkdir(parents=True);driver.write_text('driver')
    base=tmp_path/'artifacts/exploratory_final_ppo_real_cfd_20261006';base.mkdir()
    def obs(front_cd,rear_cd,rear_cl,omega):
        row=[0.0]*69;row[64]=front_cd;row[65]=.2;row[66]=rear_cd;row[67]=rear_cl;row[68]=omega
        return row
    rows=[]
    for step in (1,2):
        rows.append({'step':step,'start_time':148.+(step-1)*.1,
                     'end_time':148.+step*.1,'requested_omega':.75,
                     'applied_omega':step*.1,'applied_delta_omega':.1,
                     'output_observation':obs(1.4,1.0,.3,step*.1),
                     'zero_observation':obs(1.41,1.01,.31,0.),
                     'solver_health':{'ppo':{'steps':20,'solver_ended_cleanly':True},
                                      'zero':{'steps':20,'solver_ended_cleanly':True}}})
    (base/'progress.json').write_text(json.dumps({'completed_cycles':2,'rows':rows}))
    (base/'resources.jsonl').write_text(json.dumps({'MemAvailable':120*2**30})+'\n'+json.dumps({'MemAvailable':119*2**30})+'\n')
    hashes={approval_path.read_bytes():m._FINAL_PPO_CFD_APPROVAL_SHA,
            driver.read_bytes():m._FINAL_PPO_CFD_DRIVER_SHA}
    monkeypatch.setattr(m.hashlib,'sha256',
        lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=fd6d92f7ea9946b49c91c07e21f1d74b',
                    'MainPID=2346139','ActiveState=active','SubState=running',
                    'Result=success','ExecMainCode=0','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    run=m._exploratory_final_ppo_real_cfd(tmp_path)
    assert run['verified'] and run['running'] and run['completed_cycles']==2
    assert run['latest']['ppo_total_cd']==pytest.approx(2.4)
    assert run['latest']['zero_total_cd']==pytest.approx(2.42)
    assert run['latest']['ppo_rear_cl']==pytest.approx(.3)
    assert run['minimum_available_gib']==pytest.approx(119.)
    assert run['policy_training_complete'] and not run['gpu_training']
    assert not run['online_fno'] and not run['mpc']
    assert not run['scientific_admission'] and not run['control_success_verified']
    assert '冻结最终 PPO · 真实 CFD 配对运行' in m.PAGE
    assert '不使用上方旧 MPC 流场图' in m.PAGE


def test_diverse_ppo_terminal_card_is_separate_and_fno_frozen(tmp_path, monkeypatch):
    from types import SimpleNamespace
    docs = tmp_path / 'docs'; docs.mkdir()
    approval = docs / 'EXPLORATORY_DIVERSE_H5_PPO_APPROVAL_20261006.json'
    approval.write_text(json.dumps({'status': 'EXPLORATORY_DIVERSE_H5_PPO_EXECUTION_APPROVED',
                                    'execution_authorized': True}))
    source = (tmp_path/'artifacts/exploratory_diverse_h5_ppo_source_20261006_immutable'
              /'scripts/train_exploratory_diverse_h5_ppo.py')
    source.parent.mkdir(parents=True); source.write_text('driver')
    base = tmp_path/'artifacts/exploratory_diverse_h5_ppo_training_20261006'
    payload = base/'payload'; payload.mkdir(parents=True)
    artifact_names = ('ppo_final.zip','vecnormalize.pkl','transitions.jsonl',
                      'source_spec.json','progress.json','reset_packets.json')
    artifact_shas = {}
    hashes = {approval.read_bytes(): m._DIVERSE_H5_PPO_APPROVAL_SHA,
              source.read_bytes(): m._DIVERSE_H5_PPO_DRIVER_SHA}
    for i, name in enumerate(artifact_names):
        path = payload/name; path.write_bytes(f'artifact-{i}'.encode())
        artifact_shas[name] = f'sha-{i}'; hashes[path.read_bytes()] = f'sha-{i}'
    result = {'status':'EXPLORATORY_DIVERSE_H5_PPO_TRAINING_COMPLETE_NOT_ADMISSION',
      'timesteps':4096,'protocol':{'reset_count':24,'cfd_execution':False},
      'scientific_admission':False,'fno_tensors_unchanged':True,
      'policy_tensor_sha256_before':'before','policy_tensor_sha256_after':'after',
      'ppo_n_updates':32,'optimizer_steps':[{}]*64,
      'reset_counts_by_phase':{p:[35,34,34,34,34,34] for p in ('00','02','04','06')},
      'diagnostics':{'episode_return':{'mean':-3.693961018382353},
                     'applied_omega':{'rms':.3977217033938288}},
      'artifacts':artifact_shas}
    result_path=payload/'result.json';result_path.write_text(json.dumps(result))
    supervisor={'returncode':0,'result_sha256':m._DIVERSE_H5_PPO_RESULT_SHA,
                'scientific_admission':False,'minimum_available_bytes':119470489600}
    supervisor_path=base/'supervisor_result.json';supervisor_path.write_text(json.dumps(supervisor))
    hashes[result_path.read_bytes()]=m._DIVERSE_H5_PPO_RESULT_SHA
    hashes[supervisor_path.read_bytes()]=m._DIVERSE_H5_PPO_SUPERVISOR_SHA
    monkeypatch.setattr(m.hashlib,'sha256',
        lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=3a34c4d621244e4bbacdf1816b5b1374','MainPID=0',
        'ActiveState=active','SubState=exited','Result=success','ExecMainCode=1','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    card=m._exploratory_diverse_h5_ppo_training(tmp_path)
    assert card['verified'] and card['training_complete'] and not card['running']
    assert card['reset_count']==24 and card['phase_reset_counts']==[205]*4
    assert card['optimizer_steps']==64 and card['ppo_updates']==32
    assert card['policy_changed'] and card['fno_tensors_unchanged']
    assert not card['cfd_executed'] and not card['scientific_admission']
    assert '24 个固定真实重置态 · H5 PPO 训练' in m.PAGE
    assert '这里没有执行真实CFD' in m.PAGE


def test_32768_r2_card_requires_actual_progress_and_preserves_r1_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    docs=tmp_path/'docs';docs.mkdir()
    approval=docs/'EXPLORATORY_DIVERSE_H5_32768_PPO_R2_APPROVAL_20261006.json'
    approval.write_text(json.dumps({'status':'EXPLORATORY_DIVERSE_H5_32768_PPO_EXECUTION_APPROVED',
      'execution_authorized':True,'reviewed_by_lead':True,
      'protocol':{'timesteps':32768,'reset_count':24,'device':'cuda:0',
                  'cfd_execution':False,'scientific_admission':False}}))
    driver=tmp_path/'artifacts/exploratory_diverse_h5_32768_source_20261006_immutable/train_exploratory_diverse_h5_32768_ppo.py'
    driver.parent.mkdir(parents=True);driver.write_text('driver')
    r1=tmp_path/'artifacts/exploratory_diverse_h5_32768_ppo_training_20261006/supervisor_result.json'
    r1.parent.mkdir(parents=True);r1.write_text(json.dumps({'returncode':1,'result_sha256':None,
      'scientific_admission':False,'error':'worker exited 1'}))
    base=tmp_path/'artifacts/exploratory_diverse_h5_32768_ppo_training_20261006_r2'
    payload=base/'payload';payload.mkdir(parents=True)
    (payload/'progress.json').write_text(json.dumps({'time/total_timesteps':1024,
      'train/n_updates':4,'rollout/ep_rew_mean':-3.61})+'\n')
    (base/'memory.jsonl').write_text(json.dumps({'MemAvailable':110*2**30})+'\n')
    hashes={approval.read_bytes():m._DIVERSE_32768_PPO_R2_APPROVAL_SHA,
            driver.read_bytes():m._DIVERSE_32768_PPO_DRIVER_SHA,
            r1.read_bytes():m._DIVERSE_32768_PPO_R1_FAILURE_SHA}
    monkeypatch.setattr(m.hashlib,'sha256',
        lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=a19900b2bfa64d8d8372b67bc0564139','MainPID=2560906',
      'ActiveState=active','SubState=running','Result=success','ExecMainCode=0','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    card=m._exploratory_diverse_h5_32768_ppo_training(tmp_path)
    assert card['verified'] and card['running'] and card['timesteps']==1024
    assert card['target_timesteps']==32768 and card['ppo_updates']==4
    assert card['gpu_policy_training'] and card['fno_tensors_frozen']
    assert card['r1_pretraining_failure_preserved'] and not card['cfd_executed']
    assert '当前阶段 · 24-reset PPO 延长训练' in m.PAGE
    assert 'R1因审批JSON数值类型不一致' in m.PAGE


def test_diverse_ppo_real_cfd_live_card_is_separate_from_old_result(tmp_path, monkeypatch):
    from types import SimpleNamespace
    docs=tmp_path/'docs';docs.mkdir()
    approval=docs/'EXPLORATORY_DIVERSE_PPO_CFD_APPROVAL_20261006.json'
    approval.write_text(json.dumps({'status':'EXPLORATORY_DIVERSE_PPO_REAL_CFD_EXECUTION_APPROVED',
      'execution_authorized':True,'steps':124,'inference_device':'cpu',
      'scientific_admission':False,'driver_sha256':m._DIVERSE_PPO_CFD_DRIVER_SHA}))
    driver=tmp_path/'artifacts/exploratory_diverse_ppo_cfd_source_20261006_immutable/run_exploratory_diverse_ppo_real_cfd.py'
    driver.parent.mkdir(parents=True);driver.write_text('driver')
    base=tmp_path/'artifacts/exploratory_diverse_ppo_real_cfd_20261006';base.mkdir()
    def obs(front,rear,cl,omega):
        x=[0.0]*69;x[64]=front;x[66]=rear;x[67]=cl;x[68]=omega;return x
    row={'step':1,'start_time':148.,'end_time':148.1,'requested_omega':-.12,
         'applied_omega':-.1,'output_observation':obs(1.3,1.0,.4,-.1),
         'zero_observation':obs(1.31,1.01,.42,0.),
         'solver_health':{'ppo':{'steps':20,'solver_ended_cleanly':True},
                          'zero':{'steps':20,'solver_ended_cleanly':True}}}
    (base/'progress.json').write_text(json.dumps({'completed_cycles':1,'rows':[row]}))
    (base/'resources.jsonl').write_text(json.dumps({'MemAvailable':110*2**30})+'\n')
    hashes={approval.read_bytes():m._DIVERSE_PPO_CFD_APPROVAL_SHA,
            driver.read_bytes():m._DIVERSE_PPO_CFD_DRIVER_SHA}
    monkeypatch.setattr(m.hashlib,'sha256',
        lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=464de68ee1114eea8e8ae214d18dc045','MainPID=2474346',
      'ActiveState=active','SubState=running','Result=success','ExecMainCode=0','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    run=m._exploratory_diverse_ppo_real_cfd(tmp_path)
    assert run['verified'] and run['running'] and run['completed_cycles']==1
    assert run['latest']['requested_omega']==pytest.approx(-.12)
    assert run['latest']['ppo_total_cd']==pytest.approx(2.3)
    assert run['latest']['zero_total_cd']==pytest.approx(2.32)
    assert not run['fno_training'] and not run['scientific_admission']
    assert '24-reset PPO · 新一轮真实 CFD 配对运行' in m.PAGE
    assert '不复用旧 PPO/MPC 场图' in m.PAGE


def test_final_ppo_terminal_review_reports_all_windows_and_keeps_constraints(tmp_path,monkeypatch):
    from types import SimpleNamespace
    docs=tmp_path/'docs';docs.mkdir()
    approval={'status':'EXPLORATORY_FINAL_PPO_REAL_CFD_EXECUTION_APPROVED',
              'execution_authorized':True,'steps':124,'inference_device':'cpu',
              'scientific_admission':False,'driver_sha256':m._FINAL_PPO_CFD_DRIVER_SHA}
    approval_path=docs/'EXPLORATORY_FINAL_PPO_CFD_APPROVAL_20261006.json';approval_path.write_text(json.dumps(approval))
    review=docs/'EXPLORATORY_FINAL_PPO_CFD_TERMINAL_REVIEW_20261006.md';review.write_text('review')
    driver=tmp_path/'artifacts/exploratory_final_ppo_cfd_source_20261006_immutable/run_exploratory_ppo_real_cfd.py'
    driver.parent.mkdir(parents=True);driver.write_text('driver')
    base=tmp_path/'artifacts/exploratory_final_ppo_real_cfd_20261006';base.mkdir()
    def obs(front,rear,cl,omega):
        x=[0.0]*69;x[64]=front;x[66]=rear;x[67]=cl;x[68]=omega;return x
    rows=[]
    for step in range(1,125):
        rows.append({'step':step,'start_time':148.+(step-1)*.1,'end_time':148.+step*.1,
          'requested_omega':.75,'applied_omega':min(.75,step*.1),'applied_delta_omega':.1 if step<=7 else 0.,
          'output_observation':obs(1.4,.9,-.62,min(.75,step*.1)),
          'zero_observation':obs(1.4,.9,0.,0.),
          'solver_health':{'ppo':{'steps':20,'solver_ended_cleanly':True},'zero':{'steps':20,'solver_ended_cleanly':True}}})
    (base/'progress.json').write_text(json.dumps({'completed_cycles':124,'rows':rows}))
    (base/'resources.jsonl').write_text(json.dumps({'MemAvailable':119*2**30})+'\n')
    def window(interval,samples,drag,ratio,bias,mean):
        return {'interval_open_left_closed_right':interval,
          'branches':{'ppo':{'samples':samples,'total_cd_mean':2.29,'rear_cl_mean':mean,'rear_cl_fluctuation_rms':1.30},
                      'zero':{'samples':samples,'total_cd_mean':2.30,'rear_cl_mean':0.,'rear_cl_fluctuation_rms':1.18}},
          'paired_drag_reduction':drag,'paired_rear_cl_fluctuation_rms_ratio':ratio,
          'absolute_mean_rear_cl_over_paired_zero_rms':bias}
    result={'status':'EXPLORATORY_FINAL_PPO_REAL_CFD_COMPLETE_NOT_ADMISSION','cycles':124,
      'scientific_admission':False,'approval_sha256':m._FINAL_PPO_CFD_APPROVAL_SHA,
      'source_restart_unchanged':True,'owned_containers_cleaned':True,'fno_inference':False,'mpc_action_selection':False,
      'rows':[{'requested_omega':.75} for _ in range(124)],
      'action_summary':{'saturated_endpoints':117,'rate_limited_endpoints':7},
      'windows':{'full':window([148.,160.4],2480,.004117553,1.1058626,.5273107,-.62114),
                 'first_6p2':window([148.,154.2],1240,-.01635856,1.199832,.417266,-.49151),
                 'trailing_6p2':window([154.2,160.4],1240,.02459428,.990986,.637353,-.75077)}}
    result_path=base/'result.json';result_path.write_text(json.dumps(result))
    hashes={approval_path.read_bytes():m._FINAL_PPO_CFD_APPROVAL_SHA,driver.read_bytes():m._FINAL_PPO_CFD_DRIVER_SHA,
            result_path.read_bytes():m._FINAL_PPO_CFD_RESULT_SHA,review.read_bytes():m._FINAL_PPO_CFD_REVIEW_SHA}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=fd6d92f7ea9946b49c91c07e21f1d74b','MainPID=0','ActiveState=active',
                    'SubState=exited','Result=success','ExecMainCode=1','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    terminal=m._exploratory_final_ppo_real_cfd(tmp_path)['terminal_review']
    assert terminal['verified'] and not terminal['physical_success']
    assert terminal['requested_positive_limit_count']==124 and terminal['saturated_endpoints']==117
    assert len(terminal['windows'])==3
    assert all(not row['passes_original_10_percent_mean_bias'] for row in terminal['windows'])
    assert all(not row['passes_sensitivity_20_percent_mean_bias'] for row in terminal['windows'])
    assert terminal['windows'][0]['drag_reduction_percent']==pytest.approx(.4117553)
    assert terminal['windows'][0]['rear_cl_rms_change_percent']==pytest.approx(10.58626)
    assert '20%只作敏感性参考，不改变原10%均值偏置标准' in m.PAGE


def test_final_ppo_actual_cfd_field_is_sha_bound_and_not_old_mpc(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    base = tmp_path / 'artifacts/exploratory_final_ppo_field_preview_20261006'
    base.mkdir(parents=True)
    npz = base / 'final_ppo_actual_cfd_160.4.npz'
    x = np.linspace(8., 25., 256, dtype=np.float32)
    y = np.linspace(4., 11., 128, dtype=np.float32)
    state = np.zeros((3, 128, 256), dtype=np.float32)
    state[0] = 1.; state[2] = np.linspace(-1., 1., 256, dtype=np.float32)
    mask = np.ones((1, 128, 256), dtype=np.uint8)
    np.savez(npz, state=state, mask=mask,
             time=np.array([160.39999389648438], dtype=np.float64), x=x, y=y)
    result = {'status': 'FINAL_PPO_ACTUAL_CFD_FIELD_PREVIEW_COMPLETE_NOT_ADMISSION',
              'actual_cfd': True, 'model_prediction': False, 'cfd_rerun': False,
              'scientific_admission': False, 'npz_sha256': m._FINAL_PPO_FIELD_NPZ_SHA,
              'result_sha256': m._FINAL_PPO_CFD_RESULT_SHA,
              'review_sha256': m._FINAL_PPO_CFD_REVIEW_SHA,
              'evidence': {'intended_time': 160.4,
                           'stored_time': 160.39999389648438,
                           'time_tolerance': 3.0517578125e-05,
                           'valid_fraction': 1.0, 'state_shape': [3, 128, 256]}}
    unit_state = {'MainPID': '0', 'ActiveState': 'active', 'SubState': 'exited',
                  'Result': 'success', 'ExecMainCode': '1', 'ExecMainStatus': '0',
                  'MemoryMax': '4294967296', 'MemorySwapMax': '0'}
    unit = {'export_unit': unit_state, 'curator_unit': unit_state,
            'owned_container_ids_after_cleanup': []}
    files = {'result.json': result, 'unit_evidence.json': unit,
             'manifest.sha256.json': {'status': 'FINAL_PPO_FIELD_PREVIEW_EVIDENCE'}}
    for name, payload in files.items():
        (base / name).write_text(json.dumps(payload))
    expected = {result and (base/'result.json').read_bytes(): m._FINAL_PPO_FIELD_RESULT_SHA,
                npz.read_bytes(): m._FINAL_PPO_FIELD_NPZ_SHA,
                (base/'unit_evidence.json').read_bytes(): m._FINAL_PPO_FIELD_UNIT_EVIDENCE_SHA,
                (base/'manifest.sha256.json').read_bytes(): m._FINAL_PPO_FIELD_MANIFEST_SHA}
    monkeypatch.setattr(m.hashlib, 'sha256',
        lambda raw: SimpleNamespace(hexdigest=lambda: expected.get(raw, 'wrong')))
    evidence = m._final_ppo_field_evidence(tmp_path)
    assert evidence['verified'] and evidence['actual_cfd']
    assert not evidence['model_prediction'] and not evidence['cfd_rerun']
    assert evidence['intended_time'] == 160.4
    assert m._final_ppo_field_png(tmp_path, '0' * 64) is None
    png = m._final_ppo_field_png(tmp_path, m._FINAL_PPO_FIELD_NPZ_SHA)
    assert png.startswith(b'\x89PNG\r\n\x1a\n')
    assert '/final-ppo-field.png?v=${field.sha256}' in m.PAGE
    assert '不使用上方旧 MPC 流场图' in m.PAGE
    assert '/current-trial-field.png?v=${field.sha256}' in m.PAGE


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
