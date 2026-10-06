"""Read only the first two frames of the fixed 20 train HDFs; no model imports."""
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

ROOT = Path('/workspace/fluid_control')
DATA = ROOT / 'data/curated/tandem_cylinders_matched_start_full40_v1'
OUT = ROOT / 'artifacts/p064_train_first_pair_preflight_20261007'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert available() >= 50 * 2**30
    OUT.mkdir(exist_ok=False)
    started = time.time()
    manifest, split, norm = (DATA / p for p in ['manifest.json', 'splits/train.json', 'normalization.json'])
    m, s = json.loads(manifest.read_text()), json.loads(split.read_text())
    assert s['split'] == 'train' and len(s['cases']) == 20
    assert sha(split) == m['split_manifests']['train']['sha256']
    assert sha(norm) == m['normalization_sha256']
    receipt = dict(status='TRAIN_FIRST_TWO_FRAMES_REVIEW_COMPLETE_NOT_TRAINING',
                   invocation=os.environ.get('INVOCATION_ID'), source_sha256=sha(Path(__file__)),
                   inputs={str(p.relative_to(ROOT)): sha(p) for p in [manifest, split, norm]},
                   hdf_access='only datasets[0:2]; whole-file SHA is streaming bytes, not decoding all frames',
                   rows=[], phases=[], minimum_available_bytes=available())
    groups = {}
    for name in s['cases']:
        assert time.time() - started < 110 and available() >= 22 * 2**30
        path = DATA / 'train' / (name + '.h5')
        config_path = ROOT / 'cfd/tandem_cylinders/cases' / name / 'case_config.json'
        config = json.loads(config_path.read_text())
        assert config['split'] == 'train' and config['phase_bin'] in [0, 2, 4, 6]
        before = sha(path)
        assert before == s['hdf5_sha256'][name]
        with h5py.File(path, 'r') as f:
            values = {key: np.asarray(f[key][0:2]) for key in ['state', 'mask', 'time', 'omega', 'force']}
            shapes = {key: list(f[key].shape) for key in values}
            attrs = {key: str(value) for key, value in f.attrs.items()}
        assert all(np.isfinite(v).all() for v in values.values())
        times = values['time'].reshape(2)
        omega = values['omega'].reshape(2)
        expected_times = np.array([config['start_time'], config['start_time'] + .1], dtype=times.dtype)
        table = np.array(config['action_points'])
        expected_omega = np.interp(times, table[:, 0], table[:, 1]).astype(omega.dtype)
        row = dict(case=name, phase=config['phase_bin'], target=config['action_target'], shapes=shapes, attrs=attrs,
                   time=times.tolist(), omega=omega.tolist(), force=values['force'].tolist(),
                   expected_float32_times=expected_times.tolist(), times_exact=bool(np.array_equal(times, expected_times)),
                   nominal_time_quantization=(times.astype(float)-[config['start_time'], config['start_time']+.1]).tolist(),
                   action_table_endpoint_exact=bool(np.array_equal(omega, expected_omega)),
                   frame_hashes={k: [array_sha(v[0]), array_sha(v[1])] for k, v in values.items()},
                   hdf_sha256=before, config_sha256=sha(config_path))
        assert sha(path) == before
        receipt['rows'].append(row)
        groups.setdefault(config['phase_bin'], []).append((row, values))
        receipt['minimum_available_bytes'] = min(receipt['minimum_available_bytes'], available())
    for phase, group in sorted(groups.items()):
        assert len(group) == 5
        zero = next(v for r, v in group if r['target'] == 0)
        comparisons = []
        for row, v in group:
            comparisons.append(dict(case=row['case'], q0_exact=bool(np.array_equal(v['state'][0], zero['state'][0])),
                                    mask_both_exact=bool(np.array_equal(v['mask'], zero['mask'])),
                                    time_both_exact=bool(np.array_equal(v['time'], zero['time'])),
                                    omega0_exact=bool(np.array_equal(v['omega'][0], zero['omega'][0])),
                                    force0_exact=bool(np.array_equal(v['force'][0], zero['force'][0])),
                                    q0_max_abs=float(np.max(np.abs(v['state'][0].astype(float)-zero['state'][0]))),
                                    q1_force_delta=(v['force'][1].astype(float)-zero['force'][1]).tolist()))
        receipt['phases'].append(dict(phase=phase, comparisons=comparisons,
            unique_actual_omega1=sorted(set(float(v['omega'][1].reshape(-1)[0]) for _, v in group)),
            matched_q0_and_omega0=all(x['q0_exact'] and x['omega0_exact'] for x in comparisons)))
    receipt['elapsed_seconds'] = time.time()-started
    receipt['all_clocks_and_action_endpoints_exact'] = all(r['times_exact'] and r['action_table_endpoint_exact'] for r in receipt['rows'])
    receipt['limits'] = ['No raw CFD resampling or model calls; HDF action provenance follows saved config table.',
                         'No force causal claim beyond matched initial fields/current action; target amplitudes may share first ramp endpoint.',
                         'Only first two decoded frames; no dataset modification, normalization fitting, or training.']
    with (OUT / 'receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    print(json.dumps({'status':receipt['status'], 'phases':receipt['phases'], 'clocks':receipt['all_clocks_and_action_endpoints_exact']}))


if __name__ == '__main__':
    main()
