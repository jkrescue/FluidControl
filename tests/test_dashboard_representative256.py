import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/serve_live_research_dashboard.py'


def load_helper():
    node = next(node for node in ast.parse(SOURCE.read_text()).body
                if isinstance(node, ast.FunctionDef)
                and node.name == '_representative256_training')
    namespace = dict(Path=Path, hashlib=hashlib, json=json, math=math,
                     subprocess=subprocess)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['_representative256_training']


def test_actual_terminal_training_is_bound_and_rejected():
    value = load_helper()(SOURCE.parents[1])
    assert value['verified'] and value['terminal_verified'] and not value['running']
    assert value['invocation'] == '6bc6aca4e1bd48d5938b273d319fc716'
    assert value['accepted_steps'] == 144
    assert value['accepted_steps'] <= value['closures'] <= 300
    assert value['latest_accepted']['outer'] == value['accepted_steps']
    assert len(value['latest_accepted']['normalized_rmse']) == 4
    assert value['trial_values_displayed'] is False
    assert value['scientific_admission'] is False
    assert value['closures'] == 300 and value['selection_passed'] is False


def _prepared_root(tmp_path):
    real = SOURCE.parents[1]
    (tmp_path / 'docs').mkdir()
    approval = real / 'docs/P064_REPRESENTATIVE256_TRAINING_APPROVAL_20261007.json'
    (tmp_path / 'docs/P064_REPRESENTATIVE256_TRAINING_APPROVAL_20261007.json').write_bytes(approval.read_bytes())
    spec = json.loads(approval.read_text())
    for key in ('driver', 'core', 'consumer', 'panel_adapter', 'plan'):
        source = Path(spec[key]['path'])
        if source.is_absolute():
            continue
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(real / source)
    (tmp_path / 'artifacts/p064_representative256_training_20261007').mkdir(parents=True)
    return tmp_path


def _unit(invocation, running=True):
    root = SOURCE.parents[1]
    approval = root / 'docs/P064_REPRESENTATIVE256_TRAINING_APPROVAL_20261007.json'
    driver = root / 'artifacts/p064_representative256_source_20261007_immutable/train_representative256.py'
    return '\n'.join([
        f'InvocationID={invocation}', f'MainPID={123 if running else 0}',
        f'ActiveState={"active" if running else "inactive"}',
        f'SubState={"running" if running else "dead"}', 'Result=success', 'ExecMainStatus=0',
        f'ExecStart={{ path={driver} ; argv[]={driver} --spec {approval} --sha256 42e8c15d4835ea6696ecb12271246d9babba16b73f869b4802b57bbc8dabc38c --execute ; }}',
        'CPUQuotaPerSecUSec=4s', 'MemoryCurrent=1', 'MemoryPeak=1',
        f'MemoryMax={24 * 2**30}', 'MemorySwapMax=0', ''])


def test_isolated_running_and_mismatched_identity(monkeypatch, tmp_path):
    root = _prepared_root(tmp_path)
    real_approval = SOURCE.parents[1] / 'docs/P064_REPRESENTATIVE256_TRAINING_APPROVAL_20261007.json'
    temp_approval = root / 'docs/P064_REPRESENTATIVE256_TRAINING_APPROVAL_20261007.json'
    driver = SOURCE.parents[1] / 'artifacts/p064_representative256_source_20261007_immutable/train_representative256.py'
    text = _unit('6bc6aca4e1bd48d5938b273d319fc716').replace(str(real_approval), str(temp_approval))
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda argv, **kwargs: '96, 100, 200\n' if argv[0] == 'nvidia-smi' else text)
    value = load_helper()(root)
    assert value['verified'] and value['running'] and not value['terminal_verified']
    wrong = _unit('wrong').replace(str(real_approval), str(temp_approval))
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda argv, **kwargs: '96, 100, 200\n' if argv[0] == 'nvidia-smi' else wrong)
    assert load_helper()(root) == {'verified': False}


def test_api_registration_and_copy_are_scoped():
    source = SOURCE.read_text()
    assert "data['representative256_training'] = _representative256_training(self.root)" in source
    assert '固定256个真实H1点的代表性拟合' in source
    assert '不展开256点预测或把line-search trial当结果' in source
    assert '不是开发集改善、模型准入或新闭环达标' in source
    assert 'B-PPO/E114真实CFD闭环保持' in source
    assert 'Representative256已结束并拒绝' in source


def test_helper_never_returns_large_prediction_arrays():
    value = load_helper()(SOURCE.parents[1])
    assert 'prediction' not in value
    assert 'prediction' not in (value.get('latest_accepted') or {})
