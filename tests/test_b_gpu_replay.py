import ast
import importlib.util
from pathlib import Path
S=Path(__file__).resolve().parents[1]
R=Path('/workspace/fluid_control')

def test_numerical_rollout_and_precision_supervisor_unchanged():
    old=ast.parse((R/'scripts/replay_exploratory_causal_h5_gpu.py').read_text())
    new=ast.parse((S/'scripts/replay_p064_b_causal_h5_gpu.py').read_text())
    def get(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('override_inference_precision','inference_precision','supervise','stop','cgroup_limits'):
        assert ast.dump(get(old,name),include_attributes=False)==ast.dump(get(new,name),include_attributes=False),name

def test_future_result_not_fabricated_and_b_loader_explicit():
    text=(S/'scripts/replay_p064_b_causal_h5_gpu.py').read_text()
    assert 'RESULT_SHA =' not in text
    assert 'b_driver.load_bound_b(' in text
    assert 'load_bound_k1(' not in text
    assert '"ranking_same_count"' in text
    assert '"selection_consistency_criterion": "10/10 same selected index and exact action"' in text

def test_k1_cpu_result_rejected():
    spec=importlib.util.spec_from_file_location('replay_b',S/'scripts/replay_p064_b_causal_h5_gpu.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    import pytest
    with pytest.raises(ValueError,match='exact B CPU result'):
        m.validate_result({'identity':{'k1_manifest_sha256':'old'}})

def test_importlib_not_shadowed_inside_worker():
    tree=ast.parse((S/'scripts/replay_p064_b_causal_h5_gpu.py').read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    assert not any(isinstance(n,ast.Import) and any(a.name.startswith('importlib') for a in n.names) for n in ast.walk(worker))
