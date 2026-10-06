import copy
import importlib.util
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/compare_p064_same6_h1_ar_cached.py'
spec = importlib.util.spec_from_file_location('cached_comparison', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    evaluation = dict(split='validation', action_mode='observed', segment_stride=1,
        checkpoint_metadata=dict(manifest_sha256=m.MANIFEST, aerodynamic_model_sha256=m.MODEL),
        force_channels=list(m.CHANNELS), cases=[])
    segments = dict(split='validation', action_mode='observed', segments=[])
    force = dict(status='SAMPLED_FORCE_WINDOW_DIAGNOSTIC_COMPLETE',
        dual_fno_manifest_sha256=m.MANIFEST, model_sha256=m.MODEL,
        normalization_sha256=m.NORM, manifest_sha256=m.DATA,
        force_channels=list(m.CHANNELS), cases=[])
    for case in m.CASES:
        times = [130 + t * .1 for t in range(101)]
        omega = [t * .001 for t in range(101)]
        truth = [[1 + t * .001, -.2, 1.2, t * .002] for t in range(101)]
        pred = [list(truth[0])] + [[r[0] + .1, r[1] + .2, r[2] - .1, r[3] + .4] for r in truth[1:]]
        force['cases'].append(dict(case=case, times=times, omega_endpoints=omega,
                                  true_forces=truth, predicted_forces=pred))
        evaluation['cases'].append(dict(case=case, horizons={'1': {'failed_segments': 0}}))
        for t in range(1, 101):
            target = truth[t][0] + truth[t][2]
            rate = abs(omega[t] - omega[t-1]) / (times[t] - times[t-1])
            segments['segments'].append(dict(case=case, horizon=1, start=t-1,
                mean_abs_omega=(omega[t] + omega[t-1])/2, max_abs_omega=omega[t],
                mean_abs_domega_dt=rate, max_abs_domega_dt=rate,
                force_channel_absolute_error=dict(zip(m.CHANNELS, [.05, .1, .05, .2])),
                predicted_total_drag=target + .03, target_total_drag=target,
                total_drag_absolute_error=.03))
    return evaluation, segments, force


def test_complete_pairing_seed_exclusion_and_signed_drag_sum():
    out = m.compare(*fixture())
    assert out['pooled']['count'] == 600
    assert out['final62_targets39_to100']['count'] == 372
    assert len(out['groups']['target']) == 100
    assert all(x['count'] == 300 for x in out['groups']['phase'].values())
    assert all(x['count'] == 200 for x in out['groups']['action'].values())
    rear = out['pooled']['metrics']['rear_cl']
    assert rear['ar_mae'] == pytest.approx(.4)
    assert rear['mean_ar_minus_h1_absolute_error'] == pytest.approx(.2)
    assert rear['ar_worse_count'] == 600
    drag = out['pooled']['metrics']['total_cd']
    assert drag['ar_mae'] == pytest.approx(0, abs=1e-14)
    assert drag['ar_better_count'] == 600
    assert out['diagnostic_only'] and not out['scientific_admission']


@pytest.mark.parametrize('mutation,match', [
    (lambda e,s,f: s['segments'].pop(), 'missing'),
    (lambda e,s,f: s['segments'].append(copy.deepcopy(s['segments'][0])), 'duplicate'),
    (lambda e,s,f: f['cases'][0]['times'].pop(), '101 samples'),
    (lambda e,s,f: f['cases'][0]['predicted_forces'][1].__setitem__(3, float('nan')), 'nonfinite'),
    (lambda e,s,f: e['checkpoint_metadata'].__setitem__('manifest_sha256', 'wrong'), 'identity'),
    (lambda e,s,f: s['segments'][0].__setitem__('max_abs_omega', 9), 'action max'),
    (lambda e,s,f: s['segments'][0].__setitem__('target_total_drag', 9), 'target Cd clock'),
    (lambda e,s,f: f.__setitem__('force_channels', list(reversed(m.CHANNELS))), 'channel order'),
    (lambda e,s,f: e['cases'][0]['horizons']['1'].__setitem__('failed_segments', 1), 'failed H1'),
])
def test_reject_mismatches(mutation, match):
    inputs = fixture()
    mutation(*inputs)
    with pytest.raises(ValueError, match=match):
        m.compare(*inputs)


def test_other_horizons_not_substituted_for_h1():
    e,s,f = fixture()
    extra = copy.deepcopy(s['segments'][0])
    extra['horizon'] = 100
    extra['force_channel_absolute_error']['rear_cl'] = 1000
    s['segments'].append(extra)
    assert m.compare(e,s,f)['pooled']['metrics']['rear_cl']['h1_mae'] == pytest.approx(.2)


def test_cli_hash_provenance_exclusive_output(tmp_path):
    args = []
    for name, value in zip(('evaluation', 'segments', 'force-window'), fixture()):
        path = tmp_path / (name + '.json')
        data = json.dumps(value).encode()
        path.write_bytes(data)
        args += ['--' + name, str(path), '--' + name + '-sha256', hashlib.sha256(data).hexdigest()]
    output = tmp_path / 'comparison.json'
    command = [sys.executable, str(SCRIPT), *args, '--output', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    value = json.loads(output.read_text())
    assert len(value['input_provenance']) == 3
    assert value['script_sha256'] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
