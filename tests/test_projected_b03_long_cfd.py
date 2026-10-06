"""CPU-only b03 adaptation tests; no model, CFD or dataset payload reads."""
import ast
import importlib.util
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).parent
REPO = Path(os.environ.get('PPO_REVIEW_REPO', '/workspace/fluid_control'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NEW_PATH = HERE / 'run_exploratory_projected_32768_ppo_b03_long_cfd.py'
if not NEW_PATH.exists():
    NEW_PATH = HERE.parent / 'scripts' / NEW_PATH.name
OLD_PATH = REPO / 'scripts/run_exploratory_projected_32768_ppo_b01_long_cfd.py'
NEW = load('b03_review', NEW_PATH)
OLD = load('b01_review', OLD_PATH)


def test_all_unmodified_functions_exact():
    changed = {'execute', 'build_pair_at'}
    for node in ast.parse(OLD_PATH.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name not in changed:
            assert inspect.getsource(getattr(NEW, node.name)) == inspect.getsource(getattr(OLD, node.name))
    assert NEW.SOURCE_PATHS == OLD.SOURCE_PATHS
    assert NEW.TRAIN_UNIT == OLD.TRAIN_UNIT and NEW.PAYLOAD == OLD.PAYLOAD


def test_execute_delta_is_only_phase_identity_and_scope():
    expected = inspect.getsource(OLD.execute)
    replacements = {
        'b01': 'b03', 'B01': 'B03',
        "row['phase_bin'] == 1": "row['phase_bin'] == 3",
        "phase['split'] == 'validation'": "phase['split'] == 'frozen_test'",
        '.6991961542646722': '2.3050412654150345',
        'ac412e9e3de151253dd3006a70ab195ef8f49f9c81edeeef28086b808fe9d230': '7bb667038ffcb6c3fb21e429b7ce29f41d86e4c6835f167670489709570e49b2',
        '1ddc27110bacd814e31b3425e3550927ba7fee3f57112e62a4cb2a18288aa52c': '0931ff502abc92cb66de4a98fc83c39ced438929531eac1e470a25f31b206665',
        'acquisition_validation_b03': 'acquisition_frozen_test_b03',
        '0b59387acf6365700a4a1b2e5b00f63f3c71abb6efcfcd10850ccd5ba4df1f7c': '923b82c01a0bd2b38d8b28a2db3b0875195d520552b9f1c4a906cfb917051da0',
        "('130', 'constant', 'system')": "('144', 'constant', 'system')",
        'predeclared b03 validation phase; already used historically, not fresh final test or independent sample': 'predeclared b03 physical policy trial; fixed-action H5 payload already opened; not universally unseen or statistically independent',
    }
    for old, new in replacements.items():
        assert old in expected
        expected = expected.replace(old, new)
    assert expected == inspect.getsource(NEW.execute)


def test_windows_and_real_restart_copy(tmp_path):
    assert NEW.declared_windows() == [
        ('early_12p4', 144., 156.4, False),
        ('early_first_6p2', 144., 150.2, False),
        ('early_trailing_6p2', 150.2, 156.4, False),
        ('primary_final_60', 164., 224., False),
        ('historical_inclusive_final_60', 164., 224., True),
        ('full_80', 144., 224., False),
    ]
    source, output = tmp_path/'source', tmp_path/'output'
    output.mkdir()
    for part in ('constant', 'system', '144'):
        (source/part).mkdir(parents=True)
        (source/part/'identity').write_text(part)
    for case in NEW.build_pair_at(source, output).values():
        assert (case/'144/identity').read_text() == '144'
        assert not (case/'130').exists() and not (case/'148').exists()


def test_summary_counts_and_missing_endpoint_rejected():
    t = 144 + .005*np.arange(1, 16001)
    data = np.column_stack((t, np.ones_like(t), np.zeros_like(t)))
    assert len(NEW.fixed_window(data, 164., 224.)) == 12000
    assert len(NEW.fixed_window(data, 164., 224., True)) == 12001
    with pytest.raises(ValueError, match='force grid'):
        NEW.fixed_window(data[:-1], 164., 224.)


def test_old_profile_cannot_authorize_new_execution():
    with pytest.raises(ValueError, match='explicit approval'):
        NEW.execute({'status': OLD.STATUS, 'execution_authorized': True}, None)
    with pytest.raises(ValueError, match='explicit approval'):
        NEW.execute({'status': NEW.STATUS, 'execution_authorized': False}, None)


def test_one_filter_after_projection_and_pre_model_gate():
    tree = ast.parse(inspect.getsource(NEW.execute))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    filters = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'apply_action_rate_limit']
    projects = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'projected_request']
    assert len(filters) == len(projects) == 1
    assert projects[0].lineno < filters[0].lineno
    text = inspect.getsource(NEW.execute)
    assert text.index('validate_b00_projection(b00_projected_result)') < text.index('pickle.loads(')
    assert text.index('validate_b00_projection(b00_projected_result)') < text.index('with base.PairSolvers')
