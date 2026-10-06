"""Synthetic CPU tests; no official models, CUDA or data loads."""
import importlib.util
from pathlib import Path
import signal
import subprocess
import ast
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    here = Path(__file__).resolve().parent
    source = here / 'probe_k1_uma_no_tf32.py'
    if not source.exists():
        source = here.parent / 'scripts/probe_k1_uma_no_tf32.py'
    spec = importlib.util.spec_from_file_location('uma_no_tf32_probe', source)
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


def test_precision_override_records_effective_flags(module):
    fake = SimpleNamespace(precision='high', backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True)))
    fake.get_float32_matmul_precision = lambda: fake.precision
    fake.set_float32_matmul_precision = lambda value: setattr(fake, 'precision', value)
    record = module.override_inference_precision(fake)
    assert record['before_override'] == {'float32_matmul_precision': 'high',
        'cuda_matmul_allow_tf32': True, 'cudnn_allow_tf32': True}
    assert record['effective'] == {'float32_matmul_precision': 'highest',
        'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False}
    assert 'NOT_ORIGINAL_PROTOCOL' in record['scope']


def test_non_effective_override_rejected(module):
    fake = SimpleNamespace(backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True)))
    fake.get_float32_matmul_precision = lambda: 'high'
    fake.set_float32_matmul_precision = lambda value: None
    with pytest.raises(ValueError, match='not effective'):
        module.override_inference_precision(fake)


def test_order_and_unchanged_guard_functions(module):
    source = Path(module.__file__).read_text()
    assert source.index('flow, aero, identity = load_bound_k1(') < source.index('precision = override_inference_precision(torch)')
    assert source.index("OUTPUT / 'inference_precision.json'") < source.index('decision = plan_from_current_observation(')
    base = Path('/workspace/fluid_control/scripts/probe_k1_uma_inference.py')
    if not base.exists():
        pytest.skip('canonical base available on Main only')
    def functions(text):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef)}
    original, variant = functions(base.read_text()), functions(source)
    for name in ('memory', 'memory_ok', 'cgroup_limits', 'metadata', 'stop', 'supervise'):
        assert variant[name] == original[name], name


def test_old_approval_status_rejected(module, monkeypatch, tmp_path):
    import json
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps({'status': 'K1_UMA_GPU_INFERENCE_APPROVED',
        'execution_authorized': True, 'source_sha256': module.sha(Path(module.__file__)),
        'output': str(module.OUTPUT)}))
    monkeypatch.setattr(module.sys, 'argv', ['probe', '--execute', '--approval', str(approval),
        '--approval-sha256', module.sha(approval)])
    with pytest.raises(ValueError, match='separate exact execution approval'):
        module.main()
