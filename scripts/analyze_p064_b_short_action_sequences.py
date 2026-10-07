"""Read-only saved B high/TF32 dynamic6 first-five-point arithmetic, no model."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess

PINS = {
    'ar': ('artifacts/fcp064_arm_b_formal_resume_r3_20261006/force_window/result.json', '6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173'),
    'h1': ('artifacts/p064_teacher_forced_h1_signed_force_20261006/result.json', '1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef'),
    'alignment': ('artifacts/dynamic6_raw_time_alignment_v2_20261005.json', 'ea9d0a857d55bf6c5c52e4019f0ce8aa134067a384f3bb123f1d4f7274b6f376'),
}
ROLES = ('minus', 'zero', 'plus')
IDENTITY = {'dual_fno_manifest_sha256': '92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891',
    'model_sha256': '57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e',
    'flow_model_sha256': 'dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31',
    'normalization_sha256': 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1',
    'manifest_sha256': 'bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae'}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def f32(x): return struct.unpack('f', struct.pack('f', x))[0]
def sign(x): return 0 if x == 0 else (1 if x > 0 else -1)
def mean(xs): return math.fsum(xs) / len(xs)
def interp(t, points):
    for (a, x), (b, y) in zip(points, points[1:]):
        if a <= t <= b: return x + (y-x)*(t-a)/(b-a)
    raise ValueError('time outside action table')

def components(forces):
    return {'mean_total_cd': mean([x[0]+x[2] for x in forces]),
            'mean_rear_cl': mean([x[3] for x in forces]),
            'mean_squared_rear_cl': mean([x[3]*x[3] for x in forces])}

def assess(branches):
    result = []
    for h in range(1, 6):
        metrics = {r: {s: components(branches[r][s][:h]) for s in ('truth', 'prediction')} for r in ROLES}
        contrasts = {}
        for role in ('minus', 'plus'):
            contrasts[role] = {}
            for metric in metrics[role]['truth']:
                t = metrics[role]['truth'][metric] - metrics['zero']['truth'][metric]
                p = metrics[role]['prediction'][metric] - metrics['zero']['prediction'][metric]
                contrasts[role][metric] = {'truth_minus_zero': t, 'prediction_minus_zero': p, 'prediction_delta_error': p-t}
        pairs = []
        for a,b in itertools.combinations(ROLES,2):
            t = metrics[a]['truth']['mean_total_cd']-metrics[b]['truth']['mean_total_cd']
            p = metrics[a]['prediction']['mean_total_cd']-metrics[b]['prediction']['mean_total_cd']
            pairs.append({'a':a,'b':b,'truth_a_minus_b':t,'prediction_a_minus_b':p,
                          'truth_exact_sign':sign(t),'prediction_exact_sign':sign(p),
                          'exact_sign_agreement':sign(t)==sign(p),'physical_tie_classification':'unresolved_no_uncertainty_bound'})
        tmin = min(metrics[r]['truth']['mean_total_cd'] for r in ROLES)
        pmin = min(metrics[r]['prediction']['mean_total_cd'] for r in ROLES)
        selected = [r for r in ROLES if metrics[r]['prediction']['mean_total_cd']==pmin]
        optimum = [r for r in ROLES if metrics[r]['truth']['mean_total_cd']==tmin]
        result.append({'horizon_steps':h,'nominal_duration_D_over_U':h*.1,'components':metrics,
                       'action_minus_zero':contrasts,'total_cd_pairs':pairs,
                       'predicted_minimizer_set':selected,'cfd_minimizer_set':optimum,
                       'selected_realized_costs':[{'role':r,'cfd_regret_vs_exact_optimum':metrics[r]['truth']['mean_total_cd']-tmin,
                           'cfd_cost_minus_zero':metrics[r]['truth']['mean_total_cd']-metrics['zero']['truth']['mean_total_cd']} for r in selected]})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--unit',required=True)
    a=p.parse_args();assert not a.output.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1')
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',a.unit,'-p','MainPID','-p','InvocationID','-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec','-p','RuntimeMaxUSec'],text=True).splitlines())
    assert int(props['MainPID'])==os.getpid() and props['InvocationID']==os.environ['INVOCATION_ID']
    assert int(props['MemoryMax'])==2*2**30 and props['MemorySwapMax']=='0' and props['CPUQuotaPerSecUSec']=='1s' and props['RuntimeMaxUSec']=='2min'
    bound={}; data={}
    for name,(rel,digest) in PINS.items():
        path=a.repo/rel;assert sha(path)==digest;data[name]=json.loads(path.read_text());bound[rel]=digest
    for name in ('ar','h1'):
        assert all(data[name][k]==v for k,v in IDENTITY.items())
        assert data[name]['frozen_test_accessed'] is False
    precision={'float32_matmul_precision':'high','cuda_matmul_allow_tf32':True,'cudnn_allow_tf32':True}
    assert data['h1']['precision']==precision
    cases={x['case']:x for x in data['ar']['cases']}; h1={x['case']:x for x in data['h1']['cases']}
    expected={f'full40_dynamic_validation_{phase}_{role}' for phase in ('b01','b05') for role in ROLES}
    assert set(cases)==set(h1)==expected
    phases={}
    for phase in ('b01','b05'):
        branches={};configs={}
        for role in ROLES:
            name=f'full40_dynamic_validation_{phase}_{role}';r=cases[name]
            rel=f'cfd/tandem_cylinders/cases/{name}/case_config.json';path=a.repo/rel
            digest=data['alignment']['cases'][name]['config_sha256'];assert sha(path)==digest;bound[rel]=digest
            c=json.loads(path.read_text());configs[role]=c
            assert c['split']=='validation' and r['hdf5_sha256']==data['alignment']['cases'][name]['hdf_sha256']==h1[name]['hdf5_sha256']
            assert r['horizon_steps']==100 and len(r['times'])==len(r['true_forces'])==len(r['predicted_forces'])==len(r['omega_endpoints'])==101
            t0=c['source_restart_time'];times=r['times'][:6];omega=r['omega_endpoints'][:6]
            assert times==[f32(t0+i*.1) for i in range(6)]
            assert omega==[f32(interp(t,c['action_points'])) for t in times]
            assert all(abs(x)<=.75 for x in omega)
            assert all(abs((v-u)/(b-a))<=1+1e-10 for (a,u),(b,v) in zip(c['action_points'],c['action_points'][1:]))
            assert all(len(v)==4 and all(math.isfinite(x) for x in v) for v in r['true_forces'][:6]+r['predicted_forces'][:6])
            assert r['true_forces'][0]==r['predicted_forces'][0]
            assert r['predicted_forces'][1]==h1[name]['rows'][0]['predicted_force_s_plus_1']
            for i in range(1,6):
                z=h1[name]['rows'][i-1]
                assert z['target_time']==times[i] and z['true_force_s_plus_1']==r['true_forces'][i] and z['omega_s_plus_1']==omega[i]
            branches[role]={'truth':r['true_forces'][1:6],'prediction':r['predicted_forces'][1:6],'times':times[1:6],'omega':omega,'hdf_sha256':r['hdf5_sha256']}
        zero=cases[f'full40_dynamic_validation_{phase}_zero']
        for role in ROLES:
            assert configs[role]['source_restart_time']==configs['zero']['source_restart_time']
            assert configs[role]['source_state_sha256']==configs['zero']['source_state_sha256']
            assert cases[f'full40_dynamic_validation_{phase}_{role}']['true_forces'][0]==zero['true_forces'][0]
            assert branches[role]['omega'][0]==0
        phases[phase]={'q0_source_state_sha256':configs['zero']['source_state_sha256'],
                       'source_restart_time':configs['zero']['source_restart_time'],'branches':branches,'horizons':assess(branches)}
    result={'status':'SAVED_B_SHORT_ACTION_SEQUENCE_DIAGNOSTIC_NOT_ADMISSION','source_sha256':sha(Path(__file__)),
            'unit':a.unit,'invocation':props['InvocationID'],'limits':props,'inputs':bound,'model_identity':IDENTITY,
            'saved_prediction_precision':precision,'phases':phases,
            'limitations':['Saved high/TF32 AR, not recent highest/noTF32 development protocol.',
             'q0 equality uses pinned source-state declarations and existing audited pairing; no fields reread.',
             'Only two shared initial states; pairwise comparisons share zero and are not independent samples.',
             'Exact arithmetic signs/ties only; physical/numerical uncertainty unresolved, no invented tolerance or pass gate.',
             'Costs are separate physical components, not a bound PPO reward or an MPC competence certificate.',
             'Mean squared rearCl is a short-window proxy, not centered full-cycle RMS or the physical lift constraint.'],
            'model_forward_calls':0,'optimizer_steps':0,'new_cfd':False,'scientific_admission':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'output':str(a.output),'sha256':sha(a.output)}))

if __name__=='__main__':main()
