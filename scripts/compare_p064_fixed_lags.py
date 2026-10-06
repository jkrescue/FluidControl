"""Fixed-lag saved-force diagnostic; never selects or corrects a model output."""
import argparse
import hashlib
import json
import math
from pathlib import Path

H1_SHA = '1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef'
AR_SHA = '6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173'
CHANNELS = ['front_cd', 'front_cl', 'rear_cd', 'rear_cl', 'total_cd']
CASES = {f'full40_dynamic_validation_b{p}_{a}' for p in ('01', '05') for a in ('minus', 'zero', 'plus')}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def vector(row):
    require(len(row) == 4 and all(math.isfinite(x) for x in row), 'finite four-force vector')
    return list(row) + [row[0] + row[2]]

def metrics(pred, truth):
    require(len(pred) == len(truth) and len(pred) > 0, 'paired count')
    out = {}
    for j, name in enumerate(CHANNELS):
        p, y = [r[j] for r in pred], [r[j] for r in truth]
        n = len(p)
        pm, ym = sum(p)/n, sum(y)/n
        e = [a-b for a, b in zip(p, y)]
        pr = math.sqrt(sum((a-pm)**2 for a in p)/n)
        yr = math.sqrt(sum((a-ym)**2 for a in y)/n)
        out[name] = dict(count=n, mae=sum(map(abs,e))/n,
            rmse=math.sqrt(sum(a*a for a in e)/n), signed_bias=sum(e)/n,
            predicted_centered_rms=pr, truth_centered_rms=yr,
            centered_rms_difference=pr-yr)
    return out

def analyze(h1, ar):
    for key in ('model_sha256','normalization_sha256','dual_fno_manifest_sha256',
                'flow_model_sha256','flow_state_sha256','aerodynamic_state_sha256'):
        require(h1[key] == ar[key], 'identity '+key)
    require(h1['batch_size'] == 1, 'batch1')
    hc = {c['case']:c for c in h1['cases']}
    ac = {c['case']:c for c in ar['cases']}
    require(len(h1['cases']) == len(ar['cases']) == 6 and set(hc) == set(ac) == CASES, 'exact six cases')
    series = {}
    for name in sorted(CASES):
        h, a = hc[name], ac[name]
        require(h['hdf5_sha256'] == a['hdf5_sha256'], 'HDF identity')
        rows = h['rows']
        require(len(rows) == 100 and [r['step'] for r in rows] == list(range(1,101)), '100 ordered H1 rows')
        require(all(len(a[k]) == 101 for k in ('times','omega_endpoints','true_forces','predicted_forces')), '101 AR samples including seed')
        for t, r in enumerate(rows, 1):
            require(r['true_force_s_plus_1'] == a['true_forces'][t], 'matched truth')
            require(r['input_time'] == a['times'][t-1] and r['target_time'] == a['times'][t], 'matched times')
            require(r['omega_s'] == a['omega_endpoints'][t-1] and r['omega_s_plus_1'] == a['omega_endpoints'][t], 'matched actions')
            require(abs(a['times'][t]-a['times'][t-1]-.1) < 2e-5, 'dt=.1')
        series[name] = dict(truth=[vector(x) for x in a['true_forces']],
            ar=[vector(x) for x in a['predicted_forces']],
            h1=[None]+[vector(r['predicted_force_s_plus_1']) for r in rows])
    results = {}
    for stream in ('h1','ar'):
        results[stream] = {}
        for lag in (-1,0,1):
            per_case, pp, yy, responses = {}, [], [], {}
            for name, s in series.items():
                p = [s[stream][t] for t in range(2,100)]
                y = [s['truth'][t+lag] for t in range(2,100)]
                per_case[name] = metrics(p,y)
                pp.extend(p); yy.extend(y)
                if not name.endswith('_zero'):
                    z = series[name.rsplit('_',1)[0]+'_zero']
                    dp = [[s[stream][t][j]-z[stream][t][j] for j in range(5)] for t in range(2,100)]
                    dy = [[s['truth'][t+lag][j]-z['truth'][t+lag][j] for j in range(5)] for t in range(2,100)]
                    responses[name] = dict(metrics=metrics(dp,dy), predicted=dp, truth=dy)
            results[stream][str(lag)] = dict(per_case=per_case, pooled=metrics(pp,yy), action_minus_zero=responses)
    first = {}
    for name,s in series.items():
        if name.endswith('_zero'):
            continue
        z = series[name.rsplit('_',1)[0]+'_zero']
        first[name] = {}
        for stream in ('h1','ar'):
            p = [[s[stream][1][j]-z[stream][1][j] for j in range(5)]]
            y = [[s['truth'][1][j]-z['truth'][1][j] for j in range(5)]]
            first[name][stream] = dict(predicted=p[0],truth=y[0],metrics=metrics(p,y))
    return dict(status='E093_FIXED_LAG_SAVED_JSON_DIAGNOSTIC_NOT_ADMISSION',
        diagnostic_only=True, targets=list(range(2,100)), lags=[-1,0,1],
        lag_definition='prediction[t] versus truth[t+lag]; no correction or best-lag selection',
        results=results, first_step_action_minus_zero=first,
        limitations=['dynamic6 high/TF32 batch1; not controlled-development attribution',
            'only initial fields were matched by producer; later action contrasts include state divergence',
            'lag associations are descriptive, not proof of delay or causality'],
        scientific_admission=False, model_loaded=False, cfd_executed=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--h1',type=Path,required=True);p.add_argument('--ar',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    require(available >= 50*2**30, 'startup Available50GiB')
    values={}; pins={}
    for name,path,expected in [('h1',a.h1,H1_SHA),('ar',a.ar,AR_SHA)]:
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'input SHA '+name)
        values[name]=json.loads(raw);pins[name]=dict(path=str(path),sha256=expected)
    result=analyze(values['h1'],values['ar'])
    result.update(inputs=pins,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),startup_available_bytes=available)
    a.output.mkdir(parents=False,exist_ok=False)
    with (a.output/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],output=str(a.output/'result.json'))))

if __name__ == '__main__':main()
