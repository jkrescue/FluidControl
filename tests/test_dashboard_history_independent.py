import ast
import json
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/serve_live_research_dashboard.py'


def page():
    tree = ast.parse(SOURCE.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAGE' for t in n.targets))


def test_javascript_syntax():
    js = page().split('<script>', 1)[1].split('</script>', 1)[0]
    result = subprocess.run(['node', '--check'], input=js, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_terminal_summary_does_not_skip_historical_curves():
    html = page()
    active = html.split('function renderActiveExperiment(d){', 1)[1].split('function renderHistoricalClosedLoopEvidence(d){', 1)[0]
    history = html.split('function renderHistoricalClosedLoopEvidence(d){', 1)[1].split('function renderLegacyActiveSummary(d,active){', 1)[0]
    import re
    renderers = sorted(set(re.findall(r'\b(render\w+)\(', history)))
    program = 'const calls=[]; const elements={}; const $=id=>(elements[id]??={});\n'
    program += ''.join(f'function {name}(){{calls.push({json.dumps(name)});}}\n' for name in renderers)
    program += 'function renderHistoricalClosedLoopEvidence(d){' + history
    program += 'function renderCurrentGClosedLoop(){}\n'
    program += 'function renderAbsolute64ClosedLoop(){}\n'
    program += 'function renderActiveExperiment(d){' + active
    program += '''renderActiveExperiment({p064_coverage_d:{invocation:'old-terminal',training:false,windows:256,updates:32}});
if (!calls.includes('renderProjectedCFD') || !calls.includes('renderFinalPPORealCFD') || !calls.includes('renderExploratoryDiverse32768LongCFD')) throw Error('historical rendering skipped');
if (!elements['lead-monitor'].textContent.includes('已结束')) throw Error('terminal shown as running');
'''
    result = subprocess.run(['node'], input=program, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_old_projected_is_not_labeled_canonical():
    assert '历史 FC-E058 · PPO策略＋镜像对称处理（旧projected，非当前canonical策略）' in page()


def test_current_g_curves_map_real_rows_without_b():
    html=page()
    mapper=html.split('function currentGSeries(rows){',1)[1].split('function renderCurrentGClosedLoop(c){',1)[0]
    program='function currentGSeries(rows){'+mapper+'''
const o=Array(69).fill(0),z=Array(69).fill(0);
o[64]=1;o[65]=99;o[66]=2;o[67]=-3;z[64]=4;z[66]=5;z[67]=6;
const rows=Array.from({length:800},(_,i)=>({end_time:130+(i+1)*.1,applied_omega:.2,output_observation:o,zero_observation:z}));
const a=currentGSeries(rows);
if(a.length!==800||a[0].end_time!==130.1||a[799].end_time!==210||a[0].omega!==.2||a[0].ppo_total_cd!==3||a[0].zero_total_cd!==9||a[0].ppo_rear_cl!==-3||a[0].zero_rear_cl!==6)throw Error('wrong G mapping/clock');
let rejected=false;try{currentGSeries([{...rows[0],output_observation:[1]}]);}catch(e){rejected=true;}if(!rejected)throw Error('bad observation accepted');
'''
    p=subprocess.run(['node'],input=program,text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    render=html.split('function renderCurrentGClosedLoop(c){',1)[1].split('function renderAbsolute64ClosedLoop(d){',1)[0]
    assert 'currentGSeries(c.rows)' in render and 'reproduction' not in render
    assert '原B b01：130→210（E095）' in html

def test_terminal_and_aux_are_separate():
    html = page()
    assert '六窗原标准通过' in html
    assert '不是PPO或新CFD，flow冻结' in html
    assert '尚无已完成窗口' in html
    assert '两项H1误差与保留指标均略差，未采用' in html
    assert 'H1 Cl/Cd改善，但H5与AR保留性退化，未采用，保留B' in html
    assert 'f.development_verified&&f.next_step&&!d.bf_true_state?.verified' in html


def test_true_state_diagnostic_only_accepts_exact_terminal():
    import hashlib
    from unittest.mock import patch
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_bf_true_state')
    scope=dict(hashlib=hashlib,json=json,subprocess=subprocess,Path=Path)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)
    fn=scope[node.name];root=SOURCE.parents[1]
    state='InvocationID=c9ea8b5a18d54bc1bb5f3e20631c17f0\nMainPID=0\nSubState=exited\nResult=success\nExecMainStatus=0\n'
    with patch.object(subprocess,'check_output',return_value=state):
        r=fn(root)
        assert r['verified'] and not r['running'] and r['next_training_authorized']
        assert r['review_verified']
        assert r['actual_calls']==dict(aero=160,flow=0,optimizer=0)
    with patch.object(subprocess,'check_output',return_value=state.replace('MainPID=0','MainPID=9')):
        assert fn(root)=={'verified':False}


def test_f_uses_actual_events_and_stops_on_exited():
    import hashlib
    from unittest.mock import patch
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_h1_only_f')
    scope = dict(hashlib=hashlib, json=json, subprocess=subprocess, Path=Path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    fn = scope[node.name]
    root = SOURCE.parents[1]
    state = 'InvocationID=31f1692d9f984b91a17e21a133426e99\nMainPID=123\nSubState=running\n'
    log = '\n'.join([json.dumps(dict(event='training_window_complete',consumed=9)),json.dumps(dict(event='accumulation_update_complete',update=1))])
    with patch.object(subprocess,'check_output',side_effect=[state,log]):
        result=fn(root)
        assert result['running'] and result['windows']==9 and result['updates']==1
        assert not result['terminal_verified']
    with patch.object(subprocess,'check_output',side_effect=[state.replace('123','0').replace('running','exited'),log]):
        assert not fn(root)['running']
    complete=state.replace('123','0').replace('running','exited')+'Result=success\nExecMainStatus=0\n'
    with patch.object(subprocess,'check_output',side_effect=[complete,log]):
        result=fn(root)
        assert result['terminal_verified'] and result['development_verified'] and not result['promoted']
        assert result['next_step_running'] is False
        assert 'G状态见当前训练条目' in result['next_step']


def test_g_initialization_uses_actual_journal_not_gpu():
    import hashlib
    from unittest.mock import patch
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_ar5_reset_g')
    scope=dict(hashlib=hashlib,json=json,subprocess=subprocess,Path=Path)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)
    fn=scope[node.name];root=SOURCE.parents[1]
    state='InvocationID=a8e2f0a18136461c99e8154df953aa7e\nMainPID=1940841\nSubState=running\n'
    with patch.object(subprocess,'check_output',side_effect=[state,'initializing']):
        r=fn(root);assert r['running'] and r['windows']==0 and r['updates']==0
    log=json.dumps(dict(event='training_window_complete',consumed=17))+'\n'+json.dumps(dict(event='accumulation_update_complete',update=2))
    with patch.object(subprocess,'check_output',side_effect=[state,log]):
        r=fn(root);assert r['windows']==17 and r['updates']==2 and not r['terminal_verified']
    with patch.object(subprocess,'check_output',side_effect=[state.replace('1940841','0').replace('running','exited'),log]):
        assert not fn(root)['running']
    with patch.object(subprocess,'check_output',return_value=state.replace('a8e2f0a18136461c99e8154df953aa7e','other')):
        assert fn(root)=={'verified':False}
    terminal=state.replace('1940841','0').replace('running','exited')+'Result=success\nExecMainStatus=0\n'
    with patch.object(subprocess,'check_output',side_effect=[terminal,log]):
        r=fn(root)
        assert r['terminal_verified'] and r['development_verified']
        assert r['selection']=='FAIL_RETENTION' and not r['promoted']
    with patch.object(subprocess,'check_output',side_effect=[terminal.replace('ExecMainStatus=0','ExecMainStatus=1'),log]):
        assert not fn(root)['terminal_verified']


def test_g_exploratory_ppo_actual_identity_and_terminal_not_automatic():
    import hashlib
    from unittest.mock import patch
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_g_exploratory_ppo')
    scope=dict(hashlib=hashlib,json=json,subprocess=subprocess,Path=Path)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)
    fn=scope[node.name];root=SOURCE.parents[1]
    state='InvocationID=6aa96fbfeeb34269b1f49e04380cd417\nMainPID=2067872\nSubState=running\n'
    with patch.object(subprocess,'check_output',return_value=state):
        r=fn(root);assert r['running'] and r['timesteps']>0
        assert not r['selection_passed'] and not r['replaces_B'] and r['cfd_authorized']
    with patch.object(subprocess,'check_output',return_value=state.replace('2067872','0').replace('running','exited')):
        r=fn(root);assert not r['running'] and not r['terminal_verified']
    with patch.object(subprocess,'check_output',return_value=state.replace('2067872','0').replace('running','exited')+'Result=success\nExecMainStatus=0\n'):
        r=fn(root);assert r['terminal_verified'] and not r['selection_passed']
    with patch.object(subprocess,'check_output',return_value=state.replace('6aa96fbfeeb34269b1f49e04380cd417','wrong')):
        assert fn(root)=={'verified':False}


def test_g_exploratory_cfd_is_actual_cpu_not_training():
    import hashlib
    from unittest.mock import patch
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_g_exploratory_cfd')
    scope=dict(hashlib=hashlib,json=json,subprocess=subprocess,Path=Path)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)
    fn=scope[node.name];root=SOURCE.parents[1]
    state='InvocationID=c14a66a1464a4919a3904c1f4d2efb2d\nMainPID=2110235\nSubState=running\n'
    with patch.object(subprocess,'check_output',return_value=state):
        r=fn(root);assert r['running'] and 0<r['cycles']<=800 and not r['training']
        assert r['scientific_pass'] is None and not r['surrogate_selection_passed']
    with patch.object(subprocess,'check_output',return_value=state.replace('2110235','0').replace('running','exited')):
        r=fn(root);assert not r['running'] and not r['terminal_verified']
    with patch.object(subprocess,'check_output',return_value=state.replace('2110235','0').replace('running','exited')+'Result=success\nExecMainStatus=0\n'):
        r=fn(root);assert r['terminal_verified'] and r['physical_windows_passed']==6
        assert r['scientific_pass'] and not r['surrogate_selection_passed'] and not r['replaces_B']
    with patch.object(subprocess,'check_output',return_value=state.replace('c14a66a1464a4919a3904c1f4d2efb2d','wrong')):
        assert fn(root)=={'verified':False}


def test_reproduction_uses_canvas_and_images_before_return():
    text = page().split('function renderActiveExperiment(d){', 1)[1]
    assert text.index('canonical_seeds_real_cfd_t228') < text.index('if(reproduction?.verified)')
    for name in ['action', 'drag', 'lift']:
        assert f'<canvas width="1000" height="220" id="canonical-reproduction-{name}"></canvas>' in page()
    assert 'renderBSelectedCurves(d)' in text


def test_live_reproduction_requires_exact_invocation_and_active_pid(tmp_path):
    import hashlib
    from unittest.mock import patch
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_canonical_b01_reproduction')
    scope = dict(hashlib=hashlib, json=json, subprocess=subprocess, Path=Path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    fn = scope[node.name]
    assert fn(tmp_path) == {'verified': False}
    root = SOURCE.parents[1]
    def state(inv='f6c3fc3464074493b19ea5418ddfada5', pid=123, sub='running'):
        return f'InvocationID={inv}\nMainPID={pid}\nActiveState=active\nSubState={sub}\nMemoryCurrent=1024\nResult=success\nExecMainStatus=0\n'
    with patch.object(subprocess, 'check_output', return_value=state()):
        x = fn(root)
        assert x['verified'] and x['running'] and x['cycles'] == len(x['rows'])
    with patch.object(subprocess, 'check_output', return_value=state(pid=0, sub='exited')):
        terminal = fn(root)
        assert terminal['running'] is False
        assert terminal['terminal_verified'] is True
        assert terminal['receipt_sha256'] == '67c145d755ac1b1146b5e9cb228813ae5bf957c816acce6a5bef9a2cd09e600a'
    with patch.object(subprocess, 'check_output', return_value=state(inv='different')):
        assert fn(root) == {'verified': False}
