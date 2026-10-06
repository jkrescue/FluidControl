import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).parents[1] / 'scripts/serve_live_research_dashboard.py'
if not path.exists():
    path = Path(__file__).with_name('serve_live_research_dashboard.py')
spec = importlib.util.spec_from_file_location('comparison_dashboard', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_missing_diagnostic_fails_closed(tmp_path):
    assert not m._policy_h5_comparison(tmp_path)['verified']

def test_missing_or_changed_field_fails_closed(tmp_path):
    assert not m._long_ppo_field(tmp_path)['verified']
    p=tmp_path/'artifacts/exploratory_diverse_32768_long_field_preview_20261006'
    p.mkdir(parents=True)
    (p/'result.json').write_text('{}')
    (p/'paired_actual_cfd_228.png').write_bytes(b'not a verified image')
    assert not m._long_ppo_field(tmp_path)['verified']
    assert '非绝对压力差' in m.PAGE and '未另作无量纲缩放' in m.PAGE

def test_projected_field_is_separate_sha_bound_actual_cfd(tmp_path,monkeypatch):
    p=tmp_path/'artifacts/exploratory_projected_32768_long_field_preview_20261006';p.mkdir(parents=True)
    result={'status':'PAIRED_PROJECTED_PPO_ACTUAL_CFD_PREVIEW_COMPLETE_NOT_ADMISSION',
      'source_result_sha256':'199127979c6cb43e6304c60fc3373a2b1a8465476ffdd265d30c108dfffd0ca6',
      'source_unchanged':True,'owned_containers_cleaned':True,'actual_cfd':True,
      'model_prediction':False,'cfd_rerun':False,'scientific_admission':False,'time':228}
    (p/'result.json').write_text(json.dumps(result));(p/'paired_actual_cfd_228.png').write_bytes(b'projected png')
    hashes={(p/'result.json').read_bytes():'198aea20a908125ba33781873f6fd6f9b22f04db09c35f04aae87d14d698875a',
            (p/'paired_actual_cfd_228.png').read_bytes():'cea4dc2d48ec1919a555704fc7654f08e958738fcd17408a57a7600eaf3c1acf'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    evidence=m._projected_long_ppo_field(tmp_path)
    assert evidence['verified'] and evidence['actual_cfd'] and not evidence['model_prediction']
    assert evidence['source_result_sha256']==result['source_result_sha256']
    assert 'FC-E058 终点 t=228' in m.PAGE and 'FC-E055场图仍是未投影策略的历史证据' in m.PAGE

def test_projected_field_missing_or_changed_fails_closed(tmp_path):
    assert not m._projected_long_ppo_field(tmp_path)['verified']

def test_short_horizon_confirmation_is_inference_only_without_fake_progress(tmp_path,monkeypatch):
    docs=tmp_path/'docs';docs.mkdir()
    out=tmp_path/'artifacts/short_horizon_frozen_confirmation_20261006';out.mkdir(parents=True)
    approval=docs/'SHORT_HORIZON_FROZEN_CONFIRMATION_APPROVAL_20261006.json'
    approval.write_text(json.dumps({'status':'SHORT_HORIZON_FROZEN_CONFIRMATION_EXECUTION_APPROVED',
      'execution_authorized':True,'heldout_access_authorized':True,'optimizer_steps':0,
      'scientific_admission':False,'horizons':[1,2,3,4,5],
      'cases':[f'case{i}' for i in range(10)],'output':str(out/'payload')}))
    (out/'memory.jsonl').write_text(json.dumps({'MemAvailable':110*2**30})+'\n'+json.dumps({'MemAvailable':108*2**30})+'\n')
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:'1ae960dcee7c9aa81e438a990dbf88397f5593decd1cdbb3ba1336ea4888c162'))
    unit='\n'.join(['InvocationID=7a887ed792ec4d60a429f4a7a3660b3a','MainPID=3832775',
      'ActiveState=active','SubState=running','Result=success','ExecMainStatus=0',
      f'MemoryCurrent={4*2**30}',f'MemoryPeak={5*2**30}'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    run=m._short_horizon_frozen_confirmation(tmp_path)
    assert run['verified'] and run['running'] and run['starts_per_case']==32
    assert run['optimizer_steps']==0 and not run['cfd_executed'] and run['gpu_inference']
    assert run['progress_available'] is False and run['minimum_available_gib']==108
    assert 'FNO预留工况预测评估（推理，非训练）' in m.PAGE
    assert '不显示或估算百分比' in m.PAGE and '旧K1 H100正式FAIL' in m.PAGE
    p=tmp_path/'artifacts/exploratory_projected_32768_long_field_preview_20261006';p.mkdir(parents=True)
    (p/'result.json').write_text('{}');(p/'paired_actual_cfd_228.png').write_bytes(b'wrong')
    assert not m._projected_long_ppo_field(tmp_path)['verified']

def test_short_horizon_terminal_binds_review_and_compact_h1_h5_metrics(tmp_path,monkeypatch):
    docs=tmp_path/'docs';docs.mkdir()
    base=tmp_path/'artifacts/short_horizon_frozen_confirmation_20261006';payload=base/'payload';payload.mkdir(parents=True)
    approval=docs/'SHORT_HORIZON_FROZEN_CONFIRMATION_APPROVAL_20261006.json'
    approval.write_text(json.dumps({'status':'SHORT_HORIZON_FROZEN_CONFIRMATION_EXECUTION_APPROVED',
      'execution_authorized':True,'heldout_access_authorized':True,'optimizer_steps':0,
      'scientific_admission':False,'horizons':[1,2,3,4,5],
      'cases':[f'case{i}' for i in range(10)],'output':str(payload)}))
    evaluation={'cases':[{}]*10,'summary':{}}
    for h in range(1,6):
      evaluation['summary'][str(h)]={'segments':320,'failed_segments':0,'stable':True,
        'velocity_relative_l2':.001*h,'field_relative_l2_u_v_p':[.001,.002,.003*h],
        'total_drag_mae':.004*h,'rear_cl_mae':.005*h}
    (payload/'evaluation.json').write_text(json.dumps(evaluation))
    (payload/'segments.json').write_text(json.dumps({'segments':[{}]*1600}))
    result={'status':'SHORT_HORIZON_CONFIRMATION_COMPLETE_NOT_ADMISSION',
      'approval_sha256':'1ae960dcee7c9aa81e438a990dbf88397f5593decd1cdbb3ba1336ea4888c162',
      'evaluation_sha256':'cab65822d1c8ce5588cbcd6e1ce105d245c62d9214067a1befa4c14993fa95ec',
      'segments_sha256':'fb39a1efe690424c92267c7f01f6bf4c2ab54df0d9615e95c35a783ae6a8250d',
      'optimizer_steps':0,'scientific_admission':False}
    (payload/'result.json').write_text(json.dumps(result));(base/'memory.jsonl').write_text(json.dumps({'MemAvailable':108*2**30})+'\n')
    review=docs/'SHORT_HORIZON_FROZEN_CONFIRMATION_TERMINAL_REVIEW_20261006.md';review.write_text('review')
    hashes={approval.read_bytes():'1ae960dcee7c9aa81e438a990dbf88397f5593decd1cdbb3ba1336ea4888c162',
      (payload/'result.json').read_bytes():'77ab4fb85f238d1e76b6e5a20c18f8992182d24a82b38d7b4b5a52483f9fce20',
      (payload/'evaluation.json').read_bytes():'cab65822d1c8ce5588cbcd6e1ce105d245c62d9214067a1befa4c14993fa95ec',
      (payload/'segments.json').read_bytes():'fb39a1efe690424c92267c7f01f6bf4c2ab54df0d9615e95c35a783ae6a8250d',
      review.read_bytes():'6c6c883396a142c875e0d50758be5a10646f8ec817ed7f9cc15dac522b0ea20c'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=7a887ed792ec4d60a429f4a7a3660b3a','MainPID=0',
      'ActiveState=active','SubState=exited','Result=success','ExecMainStatus=0',
      'MemoryCurrent=0',f'MemoryPeak={5*2**30}'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    run=m._short_horizon_frozen_confirmation(tmp_path)
    assert run['verified'] and not run['running'] and run['terminal']['endpoints']==1600
    assert run['terminal']['metrics']['h1']['velocity_relative_l2']==.001
    assert run['terminal']['metrics']['h5']['rear_cl_mae']==.025
    assert '已完成并独立复核' in m.PAGE and '不证明任意策略动作分布' in m.PAGE

def test_equal_weight_comparison(tmp_path, monkeypatch):
    p = tmp_path/'artifacts/diverse_policy_h5_comparison_20261006/payload/result.json'
    p.parent.mkdir(parents=True)
    d = {'status':'DIVERSE_POLICY_H5_COMPARISON_COMPLETE_NOT_ADMISSION',
         'optimizer_steps':0,'cfd_executed':False,'scientific_admission':False,
         'panels':{'4096':{'rows':[{'return':0.}]*24},
                   '32768':{'rows':[{'return':v} for v in [1.]*7+[-1.]*10+[0.]*7]}}}
    p.write_text(json.dumps(d))
    monkeypatch.setattr(m.hashlib,'sha256',lambda _:SimpleNamespace(hexdigest=lambda:'3f8c6f7e5b03877601a3b25b26409e9d6f943fcbaec62600fa51343995922b9f'))
    r=m._policy_h5_comparison(tmp_path)
    assert r['verified'] and (r['better'],r['worse'],r['equal'])==(7,10,7)
    assert r['mean_delta']==-3/24 and not r['scientific_admission']

def test_terminal_never_replaces_live_and_windows_are_explicit(tmp_path):
    assert m._long_cfd_reported_terminal(tmp_path,{'verified':True,'running':True,'completed_cycles':800}) is None
    assert '(168,228]' in m.PAGE and '[168,228]' in m.PAGE
    assert '20%仅敏感性参考' in m.PAGE

def test_projected_missing_evidence_never_claims_live(tmp_path):
    r=m._exploratory_diverse_32768_long_cfd(tmp_path,projected=True)
    assert not r['verified'] and not r['running']
    assert 'PPO策略＋镜像对称处理' in m.PAGE
    assert '不是新训练或新模型' in m.PAGE

def test_b01_projected_r2_requires_exact_phase_identity_and_progress(tmp_path,monkeypatch):
    docs=tmp_path/'docs';docs.mkdir()
    approval=docs/'EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_APPROVAL_20261006.json'
    approval.write_text(json.dumps({'status':'EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_EXECUTION_APPROVED',
      'execution_authorized':True,'steps':800,'inference_device':'cpu','scientific_admission':False,
      'driver_sha256':'8b653f43bd1ffc69b6285dd10523199898d88d74aebe4279f65a66fb4807c741'}))
    driver=tmp_path/'artifacts/exploratory_projected_32768_ppo_b01_long_cfd_source_20261006_immutable/run_exploratory_projected_32768_ppo_b01_long_cfd.py'
    driver.parent.mkdir(parents=True);driver.write_text('driver')
    base=tmp_path/'artifacts/exploratory_projected_32768_ppo_b01_long_cfd_20261006';base.mkdir()
    def obs(front,rear,cl,omega):
        x=[0.0]*69;x[64]=front;x[66]=rear;x[67]=cl;x[68]=omega;return x
    row={'step':1,'start_time':130.,'end_time':130.1,'requested_omega':0.,'applied_omega':0.,
         'output_observation':obs(1.3,1.0,.4,0.),'zero_observation':obs(1.31,1.01,.42,0.),
         'solver_health':{'ppo':{'steps':20,'solver_ended_cleanly':True},
                          'zero':{'steps':20,'solver_ended_cleanly':True}}}
    (base/'progress.json').write_text(json.dumps({'completed_cycles':1,'rows':[row]}))
    (base/'resources.jsonl').write_text(json.dumps({'MemAvailable':111*2**30})+'\n')
    hashes={approval.read_bytes():'790bb12fae2f5df729efda98ef59e5d99f75521d220cb9b39e8caf04808e4c59',
            driver.read_bytes():'8b653f43bd1ffc69b6285dd10523199898d88d74aebe4279f65a66fb4807c741'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join(['InvocationID=9ef43959e065431490bd4725fa8fb7fe','MainPID=3509955',
      'ActiveState=active','SubState=running','Result=success','ExecMainCode=0','ExecMainStatus=0'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:unit)
    run=m._exploratory_diverse_32768_long_cfd(tmp_path,projected_b01=True)
    assert run['verified'] and run['running'] and run['completed_cycles']==1
    assert run['phase']=='b01_validation' and run['primary_window']==[150.,210.]
    assert run['latest']['force_time']==130.1 and not run['scientific_admission']
    assert 'FC-E059 · b01固定相位复验' in m.PAGE

def test_projected_b00_terminal_is_sha_bound_and_reports_early_failure(tmp_path,monkeypatch):
    base=tmp_path/'artifacts/exploratory_projected_32768_ppo_long_cfd_20261006';base.mkdir(parents=True)
    result=base/'result.json';result.write_text(json.dumps({
      'status':'EXPLORATORY_PROJECTED_32768_PPO_LONG_CFD_COMPLETE_NOT_ADMISSION',
      'cycles':800,'scientific_admission':False,'owned_containers_cleaned':True,
      'source_restart_unchanged':True,'windows':{
       'primary_final_60':{'interval':[168.,228.],'left_endpoint_included':False,
         'paired_drag_reduction':.0389197994,'paired_rear_cl_fluctuation_rms_ratio':.815695748,
         'absolute_mean_rear_cl_over_paired_zero_rms':.011385207},
       'early_first_6p2':{'absolute_mean_rear_cl_over_paired_zero_rms':.135461748}}}))
    review=tmp_path/'docs/EXPLORATORY_PROJECTED_32768_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md';review.parent.mkdir();review.write_text('review')
    hashes={result.read_bytes():'199127979c6cb43e6304c60fc3373a2b1a8465476ffdd265d30c108dfffd0ca6',
            review.read_bytes():'44ef122bae110d22b8046b98ced195e9c4f7548441bc22f863015106bfbe7ff4'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    terminal=m._projected_b00_reported_terminal(tmp_path,{'verified':True,'running':False})
    assert terminal['paired_drag_reduction']==.0389197994

def test_projected_b01_terminal_is_sha_bound_and_reports_early_failure(tmp_path,monkeypatch):
    base=tmp_path/'artifacts/exploratory_projected_32768_ppo_b01_long_cfd_20261006';base.mkdir(parents=True)
    result=base/'result.json';result.write_text(json.dumps({
      'status':'EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_COMPLETE_NOT_ADMISSION',
      'cycles':800,'scientific_admission':False,'owned_containers_cleaned':True,
      'source_restart_unchanged':True,'windows':{
       'primary_final_60':{'interval':[150.,210.],'left_endpoint_included':False,
         'paired_drag_reduction':.039236372469,'paired_rear_cl_fluctuation_rms_ratio':.81578550745,
         'absolute_mean_rear_cl_over_paired_zero_rms':.02730022153},
       'early_first_6p2':{'absolute_mean_rear_cl_over_paired_zero_rms':.127807798}}}))
    review=tmp_path/'docs/EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_TERMINAL_REVIEW_20261006.md';review.parent.mkdir();review.write_text('review')
    hashes={result.read_bytes():'961e1bc3ccb7a9f9dae4b54e9f8233c906507c794cff9a497d806391e0fc5c37',
            review.read_bytes():'1b59fdfdb9d698fd2c0622085c19bf4d59a670070ea7434cb469ac19c72a297b'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    terminal=m._projected_b01_reported_terminal(tmp_path,{'verified':True,'running':False})
    assert terminal['paired_drag_reduction']==.039236372469
    assert terminal['early_mean_bias_ratio']>.1 and terminal['review_sha256'].startswith('1b59')
    assert '800周期已完成，等待独立原始数据复核' in m.PAGE
    assert terminal['early_mean_bias_ratio']>.1
    assert '三项原标准均通过' in m.PAGE and '不能写成全部窗口通过' in m.PAGE
