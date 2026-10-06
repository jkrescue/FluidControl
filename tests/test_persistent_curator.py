"""Synthetic engineering fixtures only; no Curator/VTU/model/solver execution."""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
SOURCE_DIR = HERE.parent / 'scripts' if (HERE.parent / 'scripts/persistent_curator_frame.py').is_file() else HERE


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sampler(monkeypatch):
    calls = []
    class Mesh:
        points = torch.zeros((4, 3), dtype=torch.float32)
        global_data = {'TimeValue': torch.tensor([148.1], dtype=torch.float32)}
        def sample_data_at_points(self, query, data_source):
            assert query.shape == (4, 3)
            assert data_source == 'points'
            return {'U': torch.tensor([[1., 2., 0.], [3., 4., 0.],
                                       [float('nan'), 0., 0.], [5., 6., 0.]]),
                    'p': torch.tensor([10., 12., 99., 14.])}
    class Source:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return iter([Mesh()])
    stub = types.ModuleType('physicsnemo_curator.domains.mesh.sources.vtk')
    stub.VTKSource = Source
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return load(SOURCE_DIR / 'persistent_curator_frame.py', 'sample_test'), calls


def test_fresh_source_each_call_and_exact_gauge_mask(sampler, tmp_path):
    module, calls = sampler
    for name in ('a', 'b', 'a_repeat'):
        module.sample_frame(Path(name), tmp_path / f'{name}.npz', 2, 2)
    assert [row[0] for row in calls] == ['a', 'b', 'a_repeat']
    assert calls[0][1]['backend'] == 'pyvista'
    with np.load(tmp_path / 'a.npz') as data:
        assert data['state'].dtype == np.float32
        assert data['mask'].dtype == np.uint8
        np.testing.assert_array_equal(data['mask'], [[[1, 1], [0, 1]]])
        np.testing.assert_array_equal(data['state'][2], [[-2., 0.], [0., 2.]])
        np.testing.assert_array_equal(data['state'][:, 1, 0], [0., 0., 0.])
        assert data['time'][0] == float(torch.tensor(148.1).item())


def test_array_comparison_rejects_signed_zero_and_dtype(tmp_path):
    replay = load(SOURCE_DIR / 'replay_persistent_curator.py', 'replay_test')
    values = dict(state=np.array([0.], dtype=np.float32), mask=np.array([1], dtype=np.uint8),
                  time=np.array([1.]), x=np.array([1.]), y=np.array([1.]))
    np.savez(tmp_path / 'a.npz', **values)
    np.savez(tmp_path / 'b.npz', **values)
    replay.compare_arrays(tmp_path / 'a.npz', tmp_path / 'b.npz')
    values['state'] = np.array([-0.], dtype=np.float32)
    np.savez(tmp_path / 'b.npz', **values)
    with pytest.raises(ValueError, match='state'):
        replay.compare_arrays(tmp_path / 'a.npz', tmp_path / 'b.npz')
    values['state'] = np.array([0.], dtype=np.float64)
    np.savez(tmp_path / 'b.npz', **values)
    with pytest.raises(ValueError, match='state'):
        replay.compare_arrays(tmp_path / 'a.npz', tmp_path / 'b.npz')


def test_numerical_body_ast_identical_to_pinned_legacy():
    legacy = Path('/workspace/fluid_control/scripts/sample_tandem_vtk_frame.py')
    old = next(x for x in ast.parse(legacy.read_text()).body
               if isinstance(x, ast.FunctionDef) and x.name == 'main')
    new = next(x for x in ast.parse((SOURCE_DIR / 'persistent_curator_frame.py').read_text()).body
               if isinstance(x, ast.FunctionDef) and x.name == 'sample_frame')
    # Remove argument-parser setup and new docstring only; normalize args.field access.
    first = next(i for i, x in enumerate(old.body) if isinstance(x, ast.Assign)
                 and any(isinstance(y, ast.Name) and y.id == 'source' for y in x.targets))
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'args':
                return ast.Name(id=node.attr, ctx=node.ctx)
            return self.generic_visit(node)
    a = Normalize().visit(ast.Module(body=old.body[first:], type_ignores=[]))
    b = ast.Module(body=new.body[1:], type_ignores=[])
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
