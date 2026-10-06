import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).parents[1]/'scripts/serve_live_research_dashboard.py'
if not SOURCE.exists():
    SOURCE = Path(__file__).with_name('serve_live_research_dashboard.py')
spec = importlib.util.spec_from_file_location('dashboard_b03', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path, monkeypatch, invocation='47612677a9f64dfc968917fada5e9ba8', start=144.):
    docs=tmp_path/'docs';docs.mkdir()
    approval=docs/'EXPLORATORY_PROJECTED_32768_PPO_B03_LONG_CFD_APPROVAL_20261006.json'
    approval.write_text(json.dumps({'status':'EXPLORATORY_PROJECTED_32768_PPO_B03_LONG_CFD_EXECUTION_APPROVED',
        'execution_authorized':True,'steps':800,'inference_device':'cpu','scientific_admission':False,
        'driver_sha256':'6516456f07d765728055f58036bed97a5e37036f205c4d1ee2bf2456be8cc3b0'}))
    driver=tmp_path/'artifacts/exploratory_projected_32768_ppo_b03_long_cfd_source_20261006_immutable/run_exploratory_projected_32768_ppo_b03_long_cfd.py'
    driver.parent.mkdir(parents=True);driver.write_text('driver')
    base=tmp_path/'artifacts/exploratory_projected_32768_ppo_b03_long_cfd_20261006';base.mkdir()
    row={'step':1,'start_time':start,'end_time':start+.1,'requested_omega':0.,'applied_omega':0.,
         'output_observation':[0.]*69,'zero_observation':[0.]*69,
         'solver_health':{k:{'steps':20,'solver_ended_cleanly':True} for k in ('ppo','zero')}}
    (base/'progress.json').write_text(json.dumps({'completed_cycles':1,'rows':[row]}))
    (base/'resources.jsonl').write_text(json.dumps({'MemAvailable':110*2**30})+'\n')
    hashes={approval.read_bytes():'3ca5531815c48cff59fd1ca0d96e40e2305402435cb2e28d2adb26eeaf9328a6',
            driver.read_bytes():'6516456f07d765728055f58036bed97a5e37036f205c4d1ee2bf2456be8cc3b0'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    unit='\n'.join([f'InvocationID={invocation}','MainPID=4013554','ActiveState=active','SubState=running'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:unit)


def test_actual_b03_binding_and_no_premature_success(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    r=m._exploratory_diverse_32768_long_cfd(tmp_path,projected_b03=True)
    assert r['verified'] and r['running'] and r['completed_cycles']==1
    assert r['primary_window']==[164.,224.] and r['phase']=='b03_already_opened_fixed_action'
    assert not r['gpu_training'] and not r['control_success_verified'] and not r['scientific_admission']
    assert 'FC-E061 · b03固定相位物理确认' in m.PAGE
    assert '主窗尚待完整原始数据复核' in m.PAGE


def test_wrong_invocation_rejected(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,invocation='wrong')
    assert not m._exploratory_diverse_32768_long_cfd(tmp_path,projected_b03=True)['verified']


def test_old_phase_time_rejected(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,start=130.)
    assert not m._exploratory_diverse_32768_long_cfd(tmp_path,projected_b03=True)['verified']


def test_missing_evidence_no_fake_job(tmp_path):
    assert not m._exploratory_diverse_32768_long_cfd(tmp_path,projected_b03=True)['running']


def test_b03_terminal_requires_exact_result_and_review(tmp_path, monkeypatch):
    result_path = tmp_path/'artifacts/exploratory_projected_32768_ppo_b03_long_cfd_20261006/result.json'
    result_path.parent.mkdir(parents=True)
    primary = {'interval':[164.,224.], 'left_endpoint_included':False,
        'branches':{'ppo':{'samples':12000}, 'zero':{'samples':12000}},
        'paired_drag_reduction':.0389714021,
        'paired_rear_cl_fluctuation_rms_ratio':.814858779,
        'absolute_mean_rear_cl_over_paired_zero_rms':.0168885808}
    result_path.write_text(json.dumps({'status':'EXPLORATORY_PROJECTED_32768_PPO_B03_LONG_CFD_COMPLETE_NOT_ADMISSION',
        'cycles':800,'scientific_admission':False,'owned_containers_cleaned':True,
        'source_restart_unchanged':True,'windows':{'primary_final_60':primary}}))
    review = tmp_path/'docs/EXPLORATORY_PROJECTED_32768_PPO_B03_LONG_CFD_TERMINAL_REVIEW_20261006.md'
    review.parent.mkdir();review.write_text('review')
    hashes={result_path.read_bytes():'d4d755faf913d393ca1466ee74614c0fb11f33b8f662a76a1cb7e4de3dd17a2f',
            review.read_bytes():'0d48a7e914ec82ad682d374e6531f2aab6a305aa81dad2dde6cd5c23dca48a0f'}
    monkeypatch.setattr(m.hashlib,'sha256',lambda raw:SimpleNamespace(hexdigest=lambda:hashes.get(raw,'bad')))
    terminal=m._projected_b03_reported_terminal(tmp_path,{'verified':True,'running':False})
    assert terminal['paired_drag_reduction']==primary['paired_drag_reduction']
    assert terminal['review_sha256'].startswith('0d48a7e')
    result_path.write_text('{}')
    assert m._projected_b03_reported_terminal(tmp_path,{'verified':True,'running':False}) is None
