"""JSON-only fixed P027 train44 coverage; no model, HDF, or inference imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

CACHE = '7785ebb92ca932b4fb572175b4bd66497fc3b7495f6ecdb534f3b587a0096366'
PHASE = '57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92'
MODES = ('k1_h1', 'k1_ar', 'persistence')

def read(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('input hash')
    return json.loads(raw)

def actions(a):
    if len(a) != 62 or not all(math.isfinite(x) for x in a):
        raise ValueError('actions')
    delta = [a[i+1]-a[i] for i in range(51, 56)]
    # Exact stored signs: expose raw deltas; no error-dependent threshold.
    signs = [1 if x > 0 else -1 for x in delta if x != 0]
    return dict(current=a[51], next=a[52:57], deltas=delta,
                max_abs=max(map(abs, a[51:57])),
                changed=sum(x != 0 for x in delta),
                reversals=sum(x != y for x, y in zip(signs, signs[1:])))

def error(pred, target):
    e = [p-t for p, t in zip(pred, target)]
    return [abs(x) for x in e] + [abs(e[0]+e[2])]

def analyze(cache, mapping, manifests):
    expected = {Path(n).stem: f for f,d in cache['source_spec']['data'].items()
                for n in d['train_files']}
    rows = cache['rows']
    if len(expected) != 44 or len(rows) != 44 or {r['case'] for r in rows} != set(expected):
        raise ValueError('exact train44')
    phases = {Path(r['file']).stem:r['canonical_physical_phase'] for r in mapping['trajectories']}
    output = []
    for r in rows:
        family = expected[r['case']]
        if r['family'] != family or r['origin'] != 51 or r['phase'] != phases[r['case']]:
            raise ValueError('case identity')
        times = r['times']
        if len(times) != 62 or not all(math.isfinite(x) for x in times) or any(b <= a for a,b in zip(times,times[1:])):
            raise ValueError('time')
        target = r['metrics']['k1_h1']['target']
        metrics = {}
        for mode in MODES:
            m = r['metrics'][mode]
            arrays = [m['prediction'], m['target']]
            if any(len(a) != 10 or any(len(v) != 4 or not all(math.isfinite(x) for x in v) for v in a) for a in arrays):
                raise ValueError('force arrays')
            if m['target'] != target:
                raise ValueError('matched targets')
            if mode == 'persistence' and any(v != m['prediction'][0] for v in m['prediction']):
                raise ValueError('fixed persistence')
            metrics[mode] = [error(p,t) for p,t in zip(m['prediction'][:5],target[:5])]
        output.append(dict(case=r['case'],family=family,phase=r['phase'],
            frames=manifests[family]['frames_per_trajectory'],
            nominal_duration=(manifests[family]['frames_per_trajectory']-1)*.1,
            physical_start_time=times[0], cached_origin_time=times[51],
            cached_target_times=times[52:57], elapsed_origin=times[51]-times[0],
            action=actions(r['stored_physical_actions']), errors=metrics))
    groups = {}
    for row in output:
        a = row['action']
        for key in ('all', 'family:'+row['family'], 'action:'+('constant' if not a['changed'] else 'reversing' if a['reversals'] else 'changing')):
            groups.setdefault(key, []).append(row)
    summaries = {}
    for key, group in groups.items():
        summaries[key] = {'count':len(group), 'cases':[r['case'] for r in group],
            'mean_abs_error_by_offset':{m:[[sum(r['errors'][m][j][c] for r in group)/len(group) for c in range(5)] for j in range(5)] for m in MODES}}
    return dict(status='TRAIN_CACHE_COVERAGE_OFFLINE_NOT_ADMISSION', rows=output, groups=summaries,
        columns=['frontCd','frontCl','rearCd','rearCl','totalCd'],
        interpretation='Offsets1..5: k1_h1 is truth-conditioned one-step at each offset; k1_ar is free rollout from51; persistence holds force51. Old high/TF32 cache is not precision-matched to current controlled replay. Elapsed time is not proof of state OOD. Actions describe this cached window, not full trajectories.',
        new_inference=False, model_changed=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.repo)
    cache=read(root/'artifacts/fcp027_short_horizon_diagnostic_20261006/result.json',CACHE)
    mapping=read(root/'artifacts/fcp003c_full_train_source_phase_mapping_20261005/mapping.json',PHASE)
    manifests={f:read(Path(d['root'])/'manifest.json',d['manifest_sha256']) for f,d in cache['source_spec']['data'].items()}
    result=analyze(cache,mapping,manifests)
    result['input_sha256']={'cache':CACHE,'phase':PHASE,'manifests':{f:d['manifest_sha256'] for f,d in cache['source_spec']['data'].items()}}
    result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with open(args.output,'x') as out: json.dump(result,out,indent=2,allow_nan=False)

if __name__=='__main__':main()
