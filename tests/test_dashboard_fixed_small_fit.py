import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/serve_live_research_dashboard.py'
APPROVAL_SHA = 'f3f5701dca46f10e31647b843cd1e4840d2f90d4f1fc2b12645e30d70b91a6ba'
DRIVER_SHA = '2f43366dde682bfc8e7fadf76140ddb2c5dc5ce7d4b4d0d5077ad17f9bf85c9c'
CORE_SHA = 'e0623157281a041b05257d9843ca46e8e1557ffca83c5872d9280fbacdd5a6c0'
INV = 'a36a5f0f9d21455c84e7cf0c85e67d9f'


def load_helper(root, monkeypatch, *, failed=False, progress=()):
    driver = root / 'artifacts/p064_fixed_small_fit_r2_source_20261007_immutable/run_fixed_small_fit.py'
    core = root / 'artifacts/p064_fixed_small_fit_r2_source_20261007_immutable/bounded_lbfgs.py'
    driver.parent.mkdir(parents=True);driver.write_text('driver');core.write_text('core')
    approval = root / 'docs/P064_FIXED_SMALL_FIT_R2_APPROVAL_20261007.json';approval.parent.mkdir()
    output = root / 'artifacts/p064_fixed_small_fit_r2_20261007'
    approval.write_text(json.dumps(dict(
        status='P064_FIXED_SMALL_FIT_R2_RECOVERY_EXECUTION_APPROVED', execution_authorized=True,
        unit='fluid-control-p064-fixed-small-fit-r2-20261007.service', output=str(output),
        driver={'path':str(driver.relative_to(root)), 'sha256':DRIVER_SHA},
        core={'path':str(core.relative_to(root)), 'sha256':CORE_SHA})))
    if progress or failed:
        output.mkdir(parents=True)
    if progress:
        (output / 'progress.jsonl').write_text('\n'.join(json.dumps(row) for row in progress))
    if failed:
        (output / 'supervisor_receipt.json').write_text(json.dumps(dict(
            approval_sha256=APPROVAL_SHA, returncode=1, error="RuntimeError('worker failed')")))
    def fake_sha(blob):
        values = {b'driver':DRIVER_SHA, b'core':CORE_SHA}
        return SimpleNamespace(hexdigest=lambda:values.get(blob, APPROVAL_SHA))
    monkeypatch.setattr(hashlib, 'sha256', fake_sha)
    state = '\n'.join([
        f'InvocationID={INV}', f'MainPID={0 if failed else 1234}',
        f'ActiveState={"failed" if failed else "active"}',
        f'SubState={"failed" if failed else "running"}',
        f'Result={"exit-code" if failed else "success"}', f'ExecMainStatus={1 if failed else 0}',
        f'ExecStart=python {driver} --spec {approval} --sha256 {APPROVAL_SHA} --execute',
        'CPUQuotaPerSecUSec=4s', 'MemoryCurrent=1', 'MemoryPeak=2',
        f'MemoryMax={24*2**30}', 'MemorySwapMax=0'])
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k:state)
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_fixed_small_fit')
    ns = dict(Path=Path, hashlib=hashlib, json=json, math=math, subprocess=subprocess)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['_fixed_small_fit']


def test_only_returned_points_are_displayed(monkeypatch, tmp_path):
    rows = [
        {'event':'lbfgs_trial_closure', 'trial_not_accepted':True, 'outer':1,
         'closure':1, 'loss':1e-12},
        {'event':'lbfgs_returned_point', 'outer':1, 'closures':2,
         'measurement':{'loss':.4, 'normalized_rmse':[.1,.2,.3,.4],
                        'physical_rmse':[.01,.02,.03,.04],
                        'physical_mae':[.008,.018,.028,.038],
                        'total_cd_physical':{'rmse':.031, 'mae':.029}}},
    ]
    value = load_helper(tmp_path, monkeypatch, progress=rows)(tmp_path)
    assert value['verified'] and value['running'] and value['accepted_steps'] == 1
    assert value['closures'] == 2 and value['latest_accepted']['loss'] == .4
    assert value['latest_accepted']['normalized_rmse'] == [.1,.2,.3,.4]
    assert value['latest_accepted']['total_cd_mae'] == .029
    assert value['latest_accepted']['rear_cl_mae'] == .038
    assert not value['trial_values_displayed']


def test_actual_r1_failure_is_not_running_or_scientific_failure(monkeypatch, tmp_path):
    value = load_helper(tmp_path, monkeypatch, failed=True)(tmp_path)
    assert value['verified'] and value['failed'] and not value['running']
    assert value['accepted_steps'] == 0 and value['latest_accepted'] is None
    assert not value['scientific_admission']


def test_duplicate_returned_outer_fails_closed(monkeypatch, tmp_path):
    row = {'event':'lbfgs_returned_point', 'outer':1, 'closures':2,
           'measurement':{'loss':.4, 'normalized_rmse':[.1,.2,.3,.4],
                          'physical_rmse':[.01,.02,.03,.04],
                          'physical_mae':[.008,.018,.028,.038],
                          'total_cd_physical':{'rmse':.031, 'mae':.029}}}
    assert load_helper(tmp_path, monkeypatch, progress=[row,row])(tmp_path)['verified'] is False


def test_copy_keeps_delivery_layers_and_hides_trials():
    source = SOURCE.read_text()
    assert '只显示优化器返回后重新测量的接受点；线搜索trial值不展示、不当作最佳结果' in source
    assert 'E114基本闭环仍按原标准通过' in source
    assert '完整代理精度仍未达标' in source
    assert '物理total-Cd MAE' in source and 'rear-Cl MAE' in source
    assert 'R1在模型加载精度校验处工程失败、0前向/0优化器' in source
    assert "data['fixed_small_fit'] = _fixed_small_fit(self.root)" in source


def test_actual_terminal_gc_fallback_when_evidence_is_present():
    root = SOURCE.parents[1]
    if not (root / 'docs/P064_FIXED_SMALL_FIT_R2_INDEPENDENT_REVIEW_20261007.md').is_file():
        import pytest
        pytest.skip('actual terminal evidence absent')
    namespace = dict(Path=Path, hashlib=__import__('hashlib'), json=json,
                     math=math, subprocess=subprocess)
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_fixed_small_fit')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    value = namespace['_fixed_small_fit'](root)
    assert value['verified'] and value['terminal_verified'] and not value['running']
    assert value['accepted_steps'] == 140 and value['closures'] == 300
    assert value['latest_accepted']['total_cd_mae'] > 0
