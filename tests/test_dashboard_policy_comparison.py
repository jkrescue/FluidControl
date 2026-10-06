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
