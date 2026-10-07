import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/serve_live_research_dashboard.py'
DIGEST = '98f5638ae454e532b9ddb22b90937c95468214400b0cfe31b3031f7e3dc565f2'
RUNNER_DIGEST = '19e0799b73f218edf4a1e59f1695532154c3dec9fcf4641b07108c7da82c16c0'
INV = '0d2508de79b646f08c87d7c0f0c1d53c'


def helper(tmp_path, *, bad_inv=False):
    runner = tmp_path / 'frozen/train.py';runner.parent.mkdir();runner.write_text('runner')
    output = tmp_path / 'artifacts/p064_temporal_increment_aux_20261007'
    approval = tmp_path / 'docs/P064_TEMPORAL_INCREMENT_AUX_TRAINING_APPROVAL_20261007.json';approval.parent.mkdir()
    approval.write_text(json.dumps(dict(
        status='FC_P064_TEMPORAL_INCREMENT_AUX_TRAINING_EXECUTION_APPROVED',
        execution_authorized=True, planned_unit='fluid-control-p064-temporal-increment-aux-20261007.service',
        planned_output=str(output), argv=['python', '-u', str(runner)], runner_sha256=RUNNER_DIGEST)))
    state = '\n'.join([
        f'InvocationID={"bad" if bad_inv else INV}', 'MainPID=1028856', 'ActiveState=active',
        'SubState=running', 'Result=success', 'ExecMainStatus=0',
        f'ExecStart=python -u {runner} --output {output} --execute',
        'CPUQuotaPerSecUSec=8s', 'MemoryCurrent=2147483648', 'MemoryPeak=3221225472',
        'MemoryMax=12884901888', 'MemorySwapMax=0'])
    lines = [
        dict(event='training_window_complete', consumed=1, original_total_loss=.007,
             temporal_increment_loss=.0002, temporal_increment_weighted_loss=.0002,
             training_objective=.0072),
        dict(event='accumulation_update_complete', update=1, consumed_windows=8,
             original_total_loss=.004, temporal_increment_loss=.00015,
             training_objective=.00415, preclip_mean_gradient_norm=3.75,
             applied_clip_scale=.2667),
    ]
    log = '\n'.join(json.dumps(row) for row in lines)
    def check_output(argv, **kwargs):
        return log if argv[0] == 'journalctl' else state
    def fake_sha(blob):
        value = RUNNER_DIGEST if blob == b'runner' else DIGEST
        return SimpleNamespace(hexdigest=lambda: value)
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_temporal_increment_training')
    ns = dict(Path=Path, hashlib=SimpleNamespace(sha256=fake_sha), json=json, math=math,
              subprocess=SimpleNamespace(check_output=check_output,
                                         SubprocessError=subprocess.SubprocessError))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['_temporal_increment_training']


def test_actual_events_and_resources_are_separate(tmp_path):
    value = helper(tmp_path)(tmp_path)
    assert value['verified'] and value['running'] and value['windows'] == 1 and value['updates'] == 1
    assert value['latest_window']['original_total_loss'] == .007
    assert value['latest_window']['temporal_increment_loss'] == .0002
    assert value['latest_window']['temporal_increment_weighted_loss'] == .0002
    assert value['latest_window']['training_objective'] == .0072
    assert value['latest_update']['preclip_mean_gradient_norm'] == 3.75
    assert value['latest_update']['applied_clip_scale'] == .2667
    assert value['cpu_quota_percent'] == 800 and value['memory_max'] == 12884901888
    assert value['swap_max'] == 0 and value['gpu_device'] == 'CUDA:0'
    assert not value['improvement_claimed']


def test_wrong_invocation_fails_closed(tmp_path):
    assert helper(tmp_path, bad_inv=True)(tmp_path) == {
        'verified': False, 'observation_state': 'unavailable'}


def test_copy_does_not_claim_improvement():
    source = SOURCE.read_text()
    assert '原受力误差' in source and '相邻时刻变化误差' in source and '训练总目标' in source
    assert '训练日志，不代表精度改善或准入' in source
    assert 'H1微降、AR微升' in source
    assert '未运行dev、PPO或新CFD' in source
    assert "data['temporal_increment_training'] = _temporal_increment_training(self.root)" in source


def test_actual_terminal_evidence_when_available():
    root = SOURCE.parents[1]
    pins = [
        root / 'docs/P064_TEMPORAL_INCREMENT_AUX_TERMINAL_REVIEW_20261007.md',
        root / 'artifacts/p064_temporal_increment_terminal_audit_r2_20261007/receipt.json',
        root / 'artifacts/p064_temporal_increment_aux_20261007/result.json',
        root / 'artifacts/p064_temporal_increment_aux_20261007/dual_model_manifest.json',
    ]
    if not all(path.is_file() for path in pins):
        pytest.skip('actual Spark terminal evidence is not present')
    namespace = dict(Path=Path, hashlib=__import__('hashlib'), json=json,
                     math=math, subprocess=subprocess)
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_temporal_increment_training')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    value = namespace['_temporal_increment_training'](root)
    assert value['terminal_verified'] and value['windows'] == 256 and value['updates'] == 32
    assert not value['promoted'] and value['development_evaluation'] == 'not_run'
    assert value['retention']['h1']['nondegrading']
    assert not value['retention']['ar']['nondegrading']
