"""Bounded CPU-only supervisor for separately approved existing-frame replay."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path('/workspace/fluid_control')
BASE = REPO / 'scripts/probe_k1_uma_inference.py'
BASE_SHA = '56c17443245c60f51b4e4c8f15b1851136a091c73441e5ecd76eebe0695f5e74'
REPLAY_SHA = '521b26b52b987e7a38c9b2faa3020b6871557337bd52791e990211564eee8a86'
MODULE_SHA = '6c1ae12c0b7382347547b2049a3f82d9f18cc9b5c5c4ff71a5b9d5895d067416'
OUTPUT = REPO / 'artifacts/persistent_curator_equivalence_20261006'
SUPERVISION = REPO / 'artifacts/persistent_curator_supervision_20261006'


def load_base():
    import hashlib
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('reviewed process-group helper source differs')
    spec = importlib.util.spec_from_file_location('reviewed_uma_supervisor_helpers', BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def limits():
    line = next(x for x in Path('/proc/self/cgroup').read_text().splitlines()
                if x.startswith('0::'))
    group = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
    memory_max = (group / 'memory.max').read_text().strip()
    swap_max = (group / 'memory.swap.max').read_text().strip()
    quota, period = (group / 'cpu.max').read_text().split()
    if memory_max == 'max' or not 0 < int(memory_max) <= 4 * 2**30 or swap_max != '0':
        raise ValueError('4GiB maximum and no swap required')
    if quota == 'max' or not 0 < int(quota) <= int(period):
        raise ValueError('at most one CPU quota required')
    return {'group': str(group), 'memory_max': int(memory_max), 'swap_max': 0,
            'cpu_max': [int(quota), int(period)]}


def validate_approval(base, path, expected):
    base.require(path is not None and base.sha(path) == expected, 'approval SHA differs')
    approval = json.loads(path.read_text())
    base.require(approval.get('status') == 'PERSISTENT_CURATOR_REPLAY_APPROVED'
                 and approval.get('execution_authorized') is True,
                 'separate replay execution approval required')
    base.require(approval.get('supervisor_sha256') == base.sha(Path(__file__))
                 and approval.get('output') == str(OUTPUT)
                 and approval.get('supervision_output') == str(SUPERVISION), 'scope differs')
    replay, module = Path(approval['replay_script']), Path(approval['sampling_module'])
    for source, digest in ((replay, REPLAY_SHA), (module, MODULE_SHA)):
        base.require(source.is_absolute() and source.is_file() and not source.is_symlink()
                     and base.sha(source) == digest, 'replay source differs')
    return replay, module


def execute(base, replay, module):
    base.require(Path(sys.prefix).resolve() == (REPO / '.venv-curator-py312').resolve(), 'wrong venv')
    base.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    bound = limits()
    base.require(base.memory_ok(base.memory(), startup=True), 'startup Available <50GiB')
    base.require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'replay output exists')
    SUPERVISION.mkdir(exist_ok=False)
    process = None
    started = time.monotonic()
    def interrupted(*_):
        raise RuntimeError('signal requested shutdown')
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        with (SUPERVISION / 'run.log').open('x') as log:
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            process = subprocess.Popen(
                [sys.executable, str(replay), '--module', str(module),
                 '--module-sha256', MODULE_SHA, '--execute'],
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            while True:
                row = {'elapsed': time.monotonic() - started, **base.memory()}
                with (SUPERVISION / 'memory.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                base.require(base.memory_ok(row), 'runtime Available <22GiB')
                base.require(row['elapsed'] < 300, '300 second deadline')
                code = process.poll()
                if code is not None:
                    base.require(code == 0, f'replay exit {code}')
                    break
                time.sleep(.5)
        result_path = OUTPUT / 'result.json'
        result = json.loads(result_path.read_text())
        base.require(result['status'] == 'CPU_ENGINEERING_EQUIVALENCE_ONLY'
                     and result['scientific_admission'] is False
                     and result['fresh_frames'] == 20 and result['repeat_frames'] == 1
                     and len(result['records']) == 21
                     and all(row['exact_arrays'] is True for row in result['records']),
                     'replay completion contract differs')
        with (SUPERVISION / 'result.json').open('x') as stream:
            json.dump({'status': 'CPU_REPLAY_SUPERVISION_COMPLETE_NOT_ADMISSION',
                       'result_sha256': base.sha(result_path), 'cgroup': bound,
                       'scientific_admission': False, 'gpu_executed': False,
                       'supervisor_sha256': base.sha(Path(__file__)),
                       'replay_sha256': REPLAY_SHA, 'module_sha256': MODULE_SHA}, stream, indent=2)
    finally:
        if process is not None:
            base.stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--approval-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PREPARATION_ONLY_NOT_EXECUTED')
        return
    base = load_base()
    replay, module = validate_approval(base, args.approval, args.approval_sha256)
    execute(base, replay, module)


if __name__ == '__main__':
    main()
