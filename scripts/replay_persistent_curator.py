"""PREPARATION ONLY until separately approved; replay existing frames, no solver/model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO = Path('/workspace/fluid_control')
TRIAL = REPO / 'artifacts/exploratory_paired_h2_real_cfd_20261006'
OLD = REPO / 'scripts/sample_tandem_vtk_frame.py'
OLD_SHA = 'bbd04828a4de3c9f54b1c2deb8227c4e8616173ee945d905fed41d905af9cfa6'
RESULT_SHA = '45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb'
OUTPUT = REPO / 'artifacts/persistent_curator_equivalence_20261006'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) * 1024


def compare_arrays(left, right):
    import numpy as np
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != {'state', 'mask', 'time', 'x', 'y'} or set(a.files) != set(b.files):
            raise ValueError('array keys differ')
        for key in a.files:
            x, y = a[key], b[key]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                raise ValueError(f'exact array mismatch: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--module-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if sha(OLD) != OLD_SHA or sha(TRIAL / 'result.json') != RESULT_SHA:
        raise ValueError('historical source/result identity differs')
    if sha(args.module) != args.module_sha256:
        raise ValueError('new module identity differs')
    if Path(sys.prefix).resolve() != (REPO / '.venv-curator-py312').resolve():
        raise ValueError('wrong pinned Curator environment')
    rows = []
    for index in range(10):
        for role in ('mpc', 'zero'):
            stamp = f'{148 + index / 10:.1f}'
            frame = TRIAL / f'case_{role}' / f'VTK_current_{role}_{stamp}'.replace('.', '_')
            if not frame.is_dir() or frame.is_symlink():
                raise ValueError(f'missing/linked existing frame: {frame}')
            rows.append((role, stamp, frame))
    if not args.execute:
        print(json.dumps({'status': 'PREPARATION_ONLY', 'existing_frames': len(rows)}))
        return
    if available() < 50 * 2**30:
        raise ValueError('startup MemAvailable below 50 GiB')
    OUTPUT.mkdir(exist_ok=False)
    # Outer systemd scope must enforce MemoryMax=4G, MemorySwapMax=0,
    # CPUQuota=100%, RuntimeMaxSec=300 plus independent 22-GiB availability watch.
    start = time.monotonic()
    def guard():
        if available() < 22 * 2**30 or time.monotonic() - start > 300:
            raise RuntimeError('runtime memory/deadline guard')
    begin = time.perf_counter()
    spec = importlib.util.spec_from_file_location('persistent_curator_frame', args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import_seconds = time.perf_counter() - begin
    records = []
    # Alternate roles/time, then revisit the first frame to detect stale source state.
    for index, (role, stamp, frame) in enumerate(rows + rows[:1]):
        guard()
        old_output = OUTPUT / f'old_{index:02d}.npz'
        new_output = OUTPUT / f'new_{index:02d}.npz'
        begin = time.perf_counter()
        with (OUTPUT / f'old_{index:02d}.log').open('x') as log:
            subprocess.run([sys.executable, str(OLD), str(frame), str(old_output)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
        old_seconds = time.perf_counter() - begin
        guard()
        begin = time.perf_counter()
        module.sample_frame(frame, new_output)
        new_seconds = time.perf_counter() - begin
        compare_arrays(old_output, new_output)
        compare_arrays(TRIAL / f'current_{role}_{stamp}.npz', new_output)
        guard()
        record = {'index': index, 'role': role, 'time_label': stamp,
                  'old_cli_seconds': old_seconds, 'persistent_call_seconds': new_seconds,
                  'self_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'children_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  'available_bytes': available(), 'exact_arrays': True}
        records.append(record)
        with (OUTPUT / 'records.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
    result = {'status': 'CPU_ENGINEERING_EQUIVALENCE_ONLY', 'scientific_admission': False,
              'old_source_sha256': OLD_SHA, 'module_sha256': args.module_sha256,
              'historical_result_sha256': RESULT_SHA, 'fresh_frames': 20, 'repeat_frames': 1,
              'persistent_import_seconds': import_seconds, 'records': records,
              'rss_note': 'Linux ru_maxrss is cumulative peak, not per-call RSS or cgroup peak',
              'timing_note': 'old CLI includes startup/import/IO; persistent excludes one-time import; warm-cache replay'}
    with (OUTPUT / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
