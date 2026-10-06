import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    here = Path(__file__).resolve()
    source = here.parents[1] / 'scripts/supervise_curator_replay.py'
    if not source.is_file():
        source = here.with_name('supervise_curator_replay.py')
    spec = importlib.util.spec_from_file_location('replay_supervisor',
                                                source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_does_not_import_helpers(module, monkeypatch, capsys):
    monkeypatch.setattr(module.sys, 'argv', ['supervisor'])
    monkeypatch.setattr(module, 'load_base', lambda: pytest.fail('must not import on preparation'))
    module.main()
    assert 'PREPARATION_ONLY' in capsys.readouterr().out


def test_limits_accept_exact_and_reject_swap_memory_cpu(module, monkeypatch):
    content = {'cgroup': '0::/fixture', 'memory.max': str(4 * 2**30),
               'memory.swap.max': '0', 'cpu.max': '100000 100000'}
    monkeypatch.setattr(Path, 'read_text', lambda path: content[path.name])
    assert module.limits()['swap_max'] == 0
    for name, bad in [('memory.max', 'max'), ('memory.swap.max', '1'),
                      ('cpu.max', 'max 100000'), ('cpu.max', '200000 100000')]:
        old = content[name]
        content[name] = bad
        with pytest.raises(ValueError):
            module.limits()
        content[name] = old


def test_missing_approval_cannot_launch(module):
    def require(ok, reason):
        if not ok:
            raise ValueError(reason)
    with pytest.raises(ValueError, match='approval SHA'):
        module.validate_approval(SimpleNamespace(require=require), None, None)


def test_lifecycle_keeps_fixed_bounds_and_owned_stop(module):
    source = Path(module.__file__).read_text()
    assert "row['elapsed'] < 300" in source
    assert 'time.sleep(.5)' in source
    assert 'start_new_session=True' in source
    assert 'base.stop(process)' in source
    assert 'startup=True' in source
    assert "not OUTPUT.exists() and not OUTPUT.is_symlink()" in source
