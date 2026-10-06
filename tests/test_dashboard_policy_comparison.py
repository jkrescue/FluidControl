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
    assert terminal['early_mean_bias_ratio']>.1
    assert '三项原标准均通过' in m.PAGE and '不能写成全部窗口通过' in m.PAGE
