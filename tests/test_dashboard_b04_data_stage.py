import ast
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'

def fixture(tmp_path,live=True):
    digest='da6587c6baea40077d7482031c0edadedefe61a3663ca861110532ce22213ea4'
    approval=tmp_path/'docs/P064_B04_LONG_EXCITATION_CURATOR_APPROVAL_20261007.json';approval.parent.mkdir();approval.write_text('{}')
    state=f'InvocationID=777b61f9027244bbb069db9ee8acda90\nMainPID={123 if live else 0}\nActiveState=active\nSubState={"running" if live else "exited"}\nResult=success\nExecMainStatus=0\nExecStart={approval} {digest}\n'
    ns=dict(re=re,hashlib=SimpleNamespace(sha256=lambda b:SimpleNamespace(hexdigest=lambda:digest)),subprocess=SimpleNamespace(check_output=lambda argv,**k:state if argv[0]=='systemctl' else 'Progress: 0/1\np064_b04_long_excitation_vtk_view_20261007: sampled 51/801 VTK frames',SubprocessError=subprocess.SubprocessError))
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_b04_data_stage')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_b04_data_stage']

def test_all_files_do_not_imply_unit_finished(tmp_path):
    f=fixture(tmp_path)
    for i in range(801):
        p=tmp_path/f'artifacts/p064_b04_long_excitation_vtk_view_20261007/VTK_curator/{i}/internal.vtu';p.parent.mkdir(parents=True);p.write_text('fixture')
    r=f(tmp_path);assert r['vtk_frames']==801 and r['running'] and not r['process_completed']
    assert not r['curator_output_exists'] and not r['gpu_training'] and r['sampled_frames']==51

def test_terminal_is_not_data_qc_pass(tmp_path):
    r=fixture(tmp_path,False)(tmp_path);assert not r['running'] and r['process_completed']
    assert not r.get('terminal_verified',False)

def test_observation_failure_unknown(tmp_path):
    f=fixture(tmp_path)
    def timeout(*a,**k):raise subprocess.TimeoutExpired('systemctl',3)
    f.__globals__['subprocess'].check_output=timeout
    assert f(tmp_path)==dict(verified=False,observation_state='unavailable')

def test_curator_directory_is_not_execution_evidence(tmp_path):
    f=fixture(tmp_path);(tmp_path/'artifacts/p064_b04_long_excitation_curated_20261007').mkdir(parents=True)
    r=f(tmp_path);assert r['curator_output_exists'] and r['stage']=='curator_hdf_only'
    assert 'curator_running' not in r

def test_top_delivers_latest_verified_physics_before_research():
    s=SOURCE.read_text();assert '基本闭环已验证：E114新增800次真实CFD反馈' in s
    assert '100*(1-m.paired_rear_cl_fluctuation_rms_ratio)' in s
    assert '完整模型预测精度仍未达标，是独立研究阶段' in s
    assert 'renderBSelectedCurves(d)' in s and '无在线FNO/MPC' in s

def test_missing_sample_is_unknown_and_over_budget_rejected(tmp_path):
    f=fixture(tmp_path);old=f.__globals__['subprocess'].check_output
    f.__globals__['subprocess'].check_output=lambda argv,**k:old(argv,**k) if argv[0]=='systemctl' else 'Progress: 0/1'
    assert f(tmp_path)['sampled_frames'] is None
    f.__globals__['subprocess'].check_output=lambda argv,**k:old(argv,**k) if argv[0]=='systemctl' else 'p064_b04_long_excitation_vtk_view_20261007: sampled 802/801 VTK frames'
    assert not f(tmp_path)['verified']
