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
    program += 'function renderActiveExperiment(d){' + active
    program += '''renderActiveExperiment({p064_coverage_d:{invocation:'old-terminal',training:false,windows:256,updates:32}});
if (!calls.includes('renderProjectedCFD') || !calls.includes('renderFinalPPORealCFD') || !calls.includes('renderExploratoryDiverse32768LongCFD')) throw Error('historical rendering skipped');
if (!elements['lead-monitor'].textContent.includes('已结束')) throw Error('terminal shown as running');
'''
    result = subprocess.run(['node'], input=program, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_old_projected_is_not_labeled_canonical():
    assert '历史 FC-E058 · PPO策略＋镜像对称处理（旧projected，非当前canonical策略）' in page()

def test_terminal_and_aux_are_separate():
    html = page()
    assert '六窗原标准通过' in html
    assert '不是PPO或新CFD，flow冻结' in html
    assert '尚无已完成窗口' in html
    assert '两项H1误差与保留指标均略差，未采用' in html


def test_reproduction_uses_canvas_and_images_before_return():
    text = page().split('function renderActiveExperiment(d){', 1)[1]
    assert text.index('canonical_seeds_real_cfd_t228') < text.index('if(reproduction?.verified)')
    for name in ['action', 'drag', 'lift']:
        assert f'<canvas width="1000" height="220" id="canonical-reproduction-{name}"></canvas>' in text


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
