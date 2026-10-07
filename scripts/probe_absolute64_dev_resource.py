"""Isolated post-load no-TF32 inference override; not original precision or admission."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path('/workspace/fluid_control')
OLD_APPROVAL = REPO / 'docs/EXPLORATORY_PAIRED_H2_APPROVAL_20261006.json'
OLD_SHA = '1679f6bee293eae200c95bc078a20db5873e81afcd54d81b81bb06445015f8b2'
OUTPUT = REPO / 'artifacts/k1_uma_gpu_no_tf32_probe_20261006'
SOURCES = ('src/fluid_control/dual_fno.py', 'src/fluid_control/calibrated_checkpoint.py',
           'scripts/train_tandem_fno.py', 'scripts/p026_history_inference.py',
           'scripts/p026_state_history.py', 'src/fluid_control/online_current_frame.py',
           'src/fluid_control/exploratory_short_mpc.py', 'src/fluid_control/canonical_joint_v1.py')
RUNTIME = {'physicsnemo/models/fno/fno.py':
           'e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9',
           'physicsnemo/utils/checkpoint.py':
           '0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e'}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def memory():
    rows = {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemAvailable:', 'MemFree:', 'Cached:'))}
    require('MemAvailable' in rows, 'missing available-memory observation')
    return rows


def memory_ok(row, startup=False):
    return row['MemAvailable'] >= (50 if startup else 22) * 2**30


def cgroup_limits():
    line = next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    group = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
    require((group / 'memory.swap.max').read_text().strip() == '0', 'MemorySwapMax=0 required')
    limit = (group / 'memory.max').read_text().strip()
    require(limit != 'max' and 0 < int(limit) <= 24 * 2**30, 'MemoryMax<=24G required')
    return {'path': str(group), 'memory_max': int(limit), 'memory_swap_max': 0}


def metadata():
    require(sha(OLD_APPROVAL) == OLD_SHA, 'historical identity source changed')
    spec = json.loads(OLD_APPROVAL.read_text())
    for name in SOURCES:
        require(sha(REPO / name) == spec['source_files'][name], f'source changed: {name}')
    keys = ('k1_manifest', 'config', 'normalization', 'baseline', 'reference_sample', 'k1_formal_receipt')
    inputs = {key: REPO / spec['inputs'][key]['path'] for key in keys}
    for key, path in inputs.items():
        require(sha(path) == spec['inputs'][key]['sha256'], f'input changed: {key}')
    require(Path(sys.prefix).resolve() == (REPO / '.venv-curator-py312').resolve(), 'wrong venv')
    for package, version in {'torch': '2.14.1', 'numpy': '2.5.3', 'nvidia-physicsnemo': '2.2.2'}.items():
        require(importlib.metadata.version(package) == version, f'wrong package: {package}')
    for relative, expected in RUNTIME.items():
        require(sha(Path(sys.prefix) / 'lib/python3.12/site-packages' / relative) == expected,
                f'official runtime source changed: {relative}')
    return spec, inputs


def inference_precision(torch):
    return {'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}


def override_inference_precision(torch):
    before = inference_precision(torch)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    effective = inference_precision(torch)
    require(effective == {'float32_matmul_precision': 'highest',
                          'cuda_matmul_allow_tf32': False,
                          'cudnn_allow_tf32': False}, 'precision override not effective')
    return {'scope': 'POST_VERIFIED_LOAD_INFERENCE_OVERRIDE_NOT_ORIGINAL_PROTOCOL',
            'before_override': before, 'effective': effective}


def worker():
    spec, inputs = metadata()
    sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
    import numpy as np
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one CUDA device required')
    torch.cuda.set_per_process_memory_fraction(.06, 0)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from omegaconf import OmegaConf
    from train_tandem_fno import build_model
    from fluid_control.dual_fno import load_dual_fno
    from fluid_control.online_current_frame import normalize_current
    from fluid_control.exploratory_short_mpc import (
        load_bound_k1, load_bound_b00_baseline, plan_from_current_observation,
        validate_recorded_k1_failure)
    from p026_state_history import build_input
    validate_recorded_k1_failure(inputs['k1_formal_receipt'])
    baseline = load_bound_b00_baseline(inputs['baseline'])
    flow, aero, identity = load_bound_k1(inputs['k1_manifest'], OmegaConf.load(inputs['config']),
                                       torch.device('cuda:0'), load_dual_fno=load_dual_fno,
                                       build_model=build_model)
    precision = override_inference_precision(torch)
    with (OUTPUT / 'inference_precision.json').open('x') as stream:
        json.dump(precision, stream, allow_nan=False, indent=2)
    for model in (flow, aero):
        model.eval().requires_grad_(False)
    def state_digest():
        digest = hashlib.sha256()
        for model in (flow, aero):
            for name, tensor in model.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial = state_digest()
    norm_bytes = inputs['normalization'].read_bytes()
    norm = json.loads(norm_bytes)
    with np.load(inputs['reference_sample'], allow_pickle=False) as packet:
        arrays = {key: packet[key].copy() for key in packet.files}
    current = normalize_current(arrays, expected_time=148., expected_mask=arrays['mask'],
                                normalization_bytes=norm_bytes,
                                expected_normalization_sha256=spec['inputs']['normalization']['sha256'])
    mean = torch.tensor(norm['all_force_mean'], device='cuda', dtype=torch.float32)
    std = torch.tensor(norm['all_force_std'], device='cuda', dtype=torch.float32)
    records = []
    with torch.inference_mode():
        for index in range(3):
            require(inference_precision(torch) == precision['effective'], 'precision changed')
            require(memory_ok(memory()), 'available reserve before inference')
            torch.cuda.synchronize()
            begin = time.perf_counter()
            decision = plan_from_current_observation(
                flow, aero, current['state'].to('cuda'), current['mask'].to('cuda'), 0., mean, std,
                build_input=build_input, state_abs_limit=spec['state_abs_limit'],
                baseline_total_drag=baseline[0], baseline_rear_cl_rms=baseline[1])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            free, total = torch.cuda.mem_get_info()
            row = {'index': index, 'wall_seconds_synchronized': elapsed, 'decision': decision,
                   'inference_precision': inference_precision(torch),
                   'cuda_free_observational': free, 'cuda_total': total,
                   'allocated_peak': torch.cuda.max_memory_allocated(),
                   'reserved_peak': torch.cuda.max_memory_reserved(), 'host': memory()}
            require(math.isfinite(elapsed), 'nonfinite elapsed')
            records.append(row)
            with (OUTPUT / 'inference.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    require(initial == state_digest(), 'model tensors changed')
    require(all(p.grad is None for model in (flow, aero) for p in model.parameters()), 'unexpected gradient')
    with (OUTPUT / 'worker_result.json').open('x') as stream:
        json.dump({'status': 'GPU_NO_TF32_INFERENCE_ENGINEERING_ONLY_NOT_ADMISSION',
                   'inference_precision_override': precision,
                   'model_tensors_unchanged': True, 'scientific_admission': False,
                   'optimizer_used': False, 'cfd_executed': False,
                   'manifest_sha256': identity.manifest_sha256, 'records': records}, stream,
                  allow_nan=False, indent=2)


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def supervise(approval_path, approval_sha256):
    limits = cgroup_limits()
    require(memory_ok(memory(), True), 'startup Available <50GiB')
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    process = None
    def interrupted(*_):
        raise RuntimeError('signal requested shutdown')
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        with (OUTPUT / 'run.log').open('x') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                                        '--execute', '--approval', str(approval_path),
                                        '--approval-sha256', approval_sha256],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                row = {'elapsed': time.monotonic() - started, **memory()}
                with (OUTPUT / 'memory.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                require(memory_ok(row), 'runtime Available <22GiB')
                require(row['elapsed'] < 180, '180 second deadline')
                code = process.poll()
                if code is not None:
                    require(code == 0, f'worker exit {code}')
                    break
                time.sleep(.5)
        require((OUTPUT / 'worker_result.json').is_file(), 'missing worker result')
        with (OUTPUT / 'supervisor_result.json').open('x') as stream:
            json.dump({'exit_code': 0, 'limits': limits, 'scientific_admission': False,
                       'source_sha256': sha(Path(__file__)), 'cuda_free_is_observational': True}, stream)
    finally:
        if process is not None:
            stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--approval-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PREPARATION_ONLY_NOT_EXECUTED')
        return
    require(args.approval is not None and sha(args.approval) == args.approval_sha256, 'approval hash')
    approval = json.loads(args.approval.read_text())
    require(approval.get('status') == 'K1_UMA_GPU_NO_TF32_INFERENCE_APPROVED'
            and approval.get('execution_authorized') is True
            and approval.get('source_sha256') == sha(Path(__file__))
            and approval.get('output') == str(OUTPUT), 'separate exact execution approval required')
    if args.worker:
        cgroup_limits()
        require(memory_ok(memory()), 'worker runtime reserve')
        worker()
    else:
        supervise(args.approval, args.approval_sha256)


if __name__ == '__main__':
    main()
