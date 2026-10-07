import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/serve_live_research_dashboard.py'
UNIT = 'fluid-control-p064-b-causal-history-h5-real-cfd-20261007.service'
INVOCATION = '3dd6413fd91d4ba9bb72c61e49920792'
DIGEST = '70c2e91fdab53731d43cf0c561262c385dad2396a2291ca4f6fcaea6465aeef9'


def load_helper(tmp_path, *, running=True):
    approval = tmp_path / 'docs/P064_B_CAUSAL_HISTORY_H5_CFD_APPROVAL_20261007.json'
    approval.parent.mkdir()
    approval.write_text(json.dumps({
        'status': 'P064_B_PAIRED_CANONICAL_HISTORY_H5_REAL_CFD_EXECUTION_APPROVED',
        'execution_authorized': True,
    }))
    output = tmp_path / 'artifacts/p064_b_causal_history_h5_real_cfd_20261007'
    state = '\n'.join([
        f'InvocationID={INVOCATION}', f'MainPID={943542 if running else 0}',
        f'ActiveState={"active" if running else "inactive"}',
        f'SubState={"running" if running else "exited"}', 'Result=success',
        'ExecMainStatus=0', f'ExecStart=python {approval} --spec-sha256 {DIGEST} --output {output}',
    ])
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_b_h5_mpc_cfd')
    ns = dict(
        hashlib=SimpleNamespace(sha256=lambda _: SimpleNamespace(hexdigest=lambda: DIGEST)),
        json=json, math=__import__('math'),
        subprocess=SimpleNamespace(check_output=lambda *a, **k: state,
                                   SubprocessError=subprocess.SubprocessError),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['_b_h5_mpc_cfd'], output


def write_progress(output, cycles=2):
    output.mkdir(parents=True)
    rows = [dict(step=i, start_time=148 + (i - 1) * .1,
                 end_time=148 + i * .1, selected_omega=(-.05 + i * .05))
            for i in range(1, cycles + 1)]
    (output / 'progress.json').write_text(json.dumps({
        'status': 'EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H5_RUNNING_NOT_ADMISSION',
        'completed_cycles': cycles, 'rows': rows,
    }))


def test_live_cpu_mpc_progress_is_exact_and_not_gpu_training(tmp_path):
    helper, output = load_helper(tmp_path)
    write_progress(output, 2)
    value = helper(tmp_path)
    assert value['verified'] and value['running'] and not value['process_completed']
    assert value['unit'] == UNIT and value['invocation'] == INVOCATION
    assert value['cycles'] == 2 and value['target_cycles'] == 10
    assert value['last_time'] == 148.2 and value['selected_omega'] == .05
    assert value['cpu_online_mpc'] and not value['gpu_training']
    assert value['original_b_ppo_unchanged']


def test_terminal_process_is_not_called_scientifically_verified(tmp_path):
    helper, output = load_helper(tmp_path, running=False)
    write_progress(output, 10)
    value = helper(tmp_path)
    assert value['process_completed'] and not value['running']
    assert 'terminal_verified' not in value


def test_bad_clock_or_identity_fails_closed(tmp_path):
    helper, output = load_helper(tmp_path)
    write_progress(output, 2)
    progress = json.loads((output / 'progress.json').read_text())
    progress['rows'][-1]['end_time'] = 149
    (output / 'progress.json').write_text(json.dumps(progress))
    assert helper(tmp_path) == {'verified': False, 'observation_state': 'unavailable'}
    (tmp_path / 'docs/P064_B_CAUSAL_HISTORY_H5_CFD_APPROVAL_20261007.json').write_text('{}')
    assert helper(tmp_path) == {'verified': False, 'observation_state': 'unavailable'}


def test_dashboard_copy_distinguishes_mpc_from_delivered_ppo():
    source = SOURCE.read_text()
    assert 'CPU 在线选择动作＋配对真实 OpenFOAM' in source
    assert '不是 GPU 训练，也不是冻结 B-PPO 的重复运行' in source
    assert '10周期只作工程接线验证' in source
    assert '基本真实闭环已完成' in source
    assert '高精度FNO/泛化/MPC仍未达标' in source
    assert '无需降低已通过的10%物理偏置门限' in source
    assert "data['b_h5_mpc_cfd'] = _b_h5_mpc_cfd(self.root)" in source


def test_actual_terminal_evidence_when_available():
    root = SOURCE.parents[1]
    pins = [
        root / 'docs/P064_B_CAUSAL_HISTORY_H5_TERMINAL_REVIEW_20261007.md',
        root / 'artifacts/p064_b_h5_independent_audit_20261007/receipt.json',
        root / 'artifacts/p064_b_causal_history_h5_real_cfd_20261007/result.json',
    ]
    if not all(path.is_file() for path in pins):
        pytest.skip('actual Spark terminal evidence is not present')
    namespace = {}
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_b_h5_mpc_cfd')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'),
         dict(hashlib=hashlib, json=json, math=__import__('math'), subprocess=subprocess), namespace)
    value = namespace['_b_h5_mpc_cfd'](root)
    assert value['terminal_verified'] and value['cycles'] == 10
    assert not value['running'] and not value['admitted']
    assert abs(value['drag_reduction'] - (-.000078862246)) < 1e-12
    assert abs(value['rear_cl_rms_ratio'] - .9836105589) < 1e-10
