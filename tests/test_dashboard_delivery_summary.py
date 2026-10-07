import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'scripts/serve_live_research_dashboard.py'

def helper(output=None):
    def call(*args,**kwargs):
        if output is None:raise subprocess.TimeoutExpired(args[0],3)
        return output
    ns={'subprocess':SimpleNamespace(check_output=call,SubprocessError=subprocess.SubprocessError)}
    n=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_delivery_activity')
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['_delivery_activity']()

def test_monitors_are_not_science():
    assert helper('fluid-control-dashboard-20261002.service loaded active running web\nfluid-control-dual-node-watchdog-20261003.service loaded active running watch\n')=={'verified':True,'active_units':[]}

def test_new_job_not_hidden():
    assert helper('fluid-control-new-training.service loaded active running job')['active_units']==['fluid-control-new-training.service']

def test_timeout_is_unknown():
    assert helper()=={'verified':False}

def test_navigation_and_history_scope():
    s=SOURCE.read_text()
    for anchor in ['canonical-seeds-real-cfd-t228','canonical-reproduction-action','canonical-reproduction-drag','canonical-reproduction-lift','absolute64-closed-loop']:
        assert 'href="#'+anchor+'"' in s and ('id="'+anchor+'"' in s or "card.id='"+anchor+"'" in s)
    assert '历史模型开发记录（展开查看，不代表当前运行）' in s
    assert '完整预测精度尚未通过' in s and '不把旧流场冒充新候选预测' in s
    assert "a?.verified?(a.active_units.length?" in s
