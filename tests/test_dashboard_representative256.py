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


def test_actual_running_training_is_bound_and_uses_accepted_points():
    value = load_helper()(SOURCE.parents[1])
    assert value['verified'] and value['running']
    assert value['invocation'] == '6bc6aca4e1bd48d5938b273d319fc716'
    assert 1 <= value['accepted_steps'] <= 200
    assert value['accepted_steps'] <= value['closures'] <= 300
    assert value['latest_accepted']['outer'] == value['accepted_steps']
    assert len(value['latest_accepted']['normalized_rmse']) == 4
    assert value['trial_values_displayed'] is False
    assert value['scientific_admission'] is False
    assert value['cpu_quota_percent'] == 400
    assert value['memory_max'] == 24 * 2**30 and value['swap_max'] == 0


def test_api_registration_and_copy_are_scoped():
    source = SOURCE.read_text()
    assert "data['representative256_training'] = _representative256_training(self.root)" in source
    assert '固定256个真实H1点的代表性拟合' in source
    assert '不展开256点预测或把line-search trial当结果' in source
    assert '不是开发集改善、模型准入或新闭环达标' in source
    assert 'B-PPO/E114真实CFD闭环保持' in source


def test_helper_never_returns_large_prediction_arrays():
    value = load_helper()(SOURCE.parents[1])
    assert 'prediction' not in value
    assert 'prediction' not in (value.get('latest_accepted') or {})
