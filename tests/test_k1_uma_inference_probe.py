"""Synthetic CPU tests; no official models, CUDA or data loads."""
import importlib.util
from pathlib import Path
import signal
import subprocess

import pytest


@pytest.fixture
def module():
    here = Path(__file__).resolve()
    source = here.parents[1] / 'scripts/probe_k1_uma_inference.py'
    if not source.is_file():
        source = here.with_name('probe_k1_uma_inference.py')
    spec = importlib.util.spec_from_file_location('uma_probe', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_available_not_memfree_or_cuda_admission(module):
    row = {'MemAvailable': 50 * 2**30, 'MemFree': 1, 'cuda_free': 1}
    assert module.memory_ok(row, startup=True)
    row['MemAvailable'] = 22 * 2**30
    assert module.memory_ok(row)
    assert not module.memory_ok(row, startup=True)
    row['MemAvailable'] -= 1
    assert not module.memory_ok(row)


def test_missing_memory_fails_closed(module):
    with pytest.raises(KeyError):
        module.memory_ok({'MemFree': 100 * 2**30})


def test_owned_process_group_escalation(module, monkeypatch):
    calls = []
    class Process:
        pid = 12345
        def poll(self):
            return None
        def wait(self, timeout):
            calls.append(('wait', timeout))
            if len(calls) == 2:
                raise subprocess.TimeoutExpired('fixture', timeout)
    monkeypatch.setattr(module.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    module.stop(Process())
    assert calls == [(12345, signal.SIGTERM), ('wait', 5),
                     (12345, signal.SIGKILL), ('wait', 5)]


def test_completed_process_not_signalled(module, monkeypatch):
    class Process:
        def poll(self):
            return 0
    monkeypatch.setattr(module.os, 'killpg', lambda *_: pytest.fail('unexpected signal'))
    module.stop(Process())


def test_worker_cannot_skip_approval(module, monkeypatch):
    monkeypatch.setattr(module.sys, 'argv', ['probe', '--worker', '--execute'])
    monkeypatch.setattr(module, 'worker', lambda: pytest.fail('unauthorized worker'))
    with pytest.raises(ValueError, match='approval hash'):
        module.main()


def test_default_is_no_execution(module, monkeypatch, capsys):
    monkeypatch.setattr(module.sys, 'argv', ['probe'])
    module.main()
    assert capsys.readouterr().out.strip() == 'PREPARATION_ONLY_NOT_EXECUTED'
