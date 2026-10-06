#!/usr/bin/env python3
"""Saved-JSON same-state/AR error comparison; no data reader or model imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

CHANNELS = ('front_cd', 'front_cl', 'rear_cd', 'rear_cl')
CASES = tuple(f'full40_dynamic_validation_b{p:02d}_{a}' for p in (1,5) for a in ('minus','zero','plus'))
MANIFEST = '92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891'
MODEL = '57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e'
NORM = 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1'
DATA = 'bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae'

def require(ok, label):
    if not ok: raise ValueError(label)

def number(x):
    require(type(x) in (int,float) and math.isfinite(x), 'nonfinite/non-numeric')
    return x

def close(x,y,label,atol=2e-6):
    require(math.isclose(number(x),number(y),rel_tol=2e-6,abs_tol=atol),label)

def summarize(rows):
    result={'count':len(rows), 'metrics':{}}
    for key in (*CHANNELS,'total_cd'):
        a=[r['ar_absolute_error'][key] for r in rows];h=[r['h1_absolute_error'][key] for r in rows]
        result['metrics'][key]={'h1_mae':math.fsum(h)/len(h),'ar_mae':math.fsum(a)/len(a),
            'mean_ar_minus_h1_absolute_error':math.fsum(x-y for x,y in zip(a,h))/len(a),
            'ar_worse_count':sum(x>y for x,y in zip(a,h)), 'ar_better_count':sum(x<y for x,y in zip(a,h)),
            'exact_tie_count':sum(x==y for x,y in zip(a,h))}
    return result

def compare(evaluation, segments, force):
    require(evaluation['split']=='validation' and evaluation['action_mode']=='observed' and evaluation['segment_stride']==1,'evaluation protocol')
    meta=evaluation['checkpoint_metadata']
    require(meta['manifest_sha256']==MANIFEST and meta['aerodynamic_model_sha256']==MODEL,'H1 candidate identity')
    require(force['dual_fno_manifest_sha256']==MANIFEST and force['model_sha256']==MODEL,'AR candidate identity')
    require(force['normalization_sha256']==NORM and force['manifest_sha256']==DATA,'AR data identity')
    require(tuple(force['force_channels'])==CHANNELS and tuple(evaluation['force_channels'])==CHANNELS,'channel order')
    require(segments['split']=='validation' and segments['action_mode']=='observed','segment protocol')
    require(force['status']=='SAMPLED_FORCE_WINDOW_DIAGNOSTIC_COMPLETE','incomplete force window')
    require(len(force['cases'])==6 and {r['case'] for r in force['cases']}==set(CASES),'six AR cases')
    require(len(evaluation['cases'])==6 and {r['case'] for r in evaluation['cases']}==set(CASES),'six H1 cases')
    for case in evaluation['cases']:
        require(case['horizons']['1']['failed_segments']==0,'failed H1 segments')
    index={}
    for row in segments['segments']:
        require(row['case'] in CASES,'unexpected case')
        if row['horizon']==1 and 0<=row['start']<100:
            require(type(row['start']) is int and type(row['horizon']) is int,'integer indices')
            key=(row['case'],row['start']+1)
            require(key not in index,'duplicate H1 endpoint');index[key]=row
    require(set(index)=={(c,t) for c in CASES for t in range(1,101)},'missing H1 endpoint')
    output=[]
    for case in sorted(force['cases'],key=lambda x:x['case']):
        times,omega,truth,pred=(case[k] for k in ('times','omega_endpoints','true_forces','predicted_forces'))
        require(all(len(a)==101 for a in (times,omega,truth,pred)),'101 samples including seed required')
        require(all(len(a)==4 for a in truth+pred),'four force columns')
        for row in truth+pred:
            for x in row:number(x)
        for t in range(101):number(times[t]);number(omega[t])
        require(truth[0]==pred[0],'initial force seed differs')
        for t in range(1,101):
            dt=times[t]-times[t-1];close(dt,.1,'time grid',2e-5)
            h=index[(case['case'],t)];now,nxt=omega[t-1:t+1]
            close(h['mean_abs_omega'],(abs(now)+abs(nxt))/2,'action mean')
            close(h['max_abs_omega'],max(abs(now),abs(nxt)),'action max')
            for key in ('mean_abs_domega_dt','max_abs_domega_dt'):close(h[key],abs(nxt-now)/dt,'action rate',3e-5)
            require(set(h['force_channel_absolute_error'])==set(CHANNELS),'H1 channels')
            he={c:number(h['force_channel_absolute_error'][c]) for c in CHANNELS}
            require(all(x>=0 for x in he.values()),'negative absolute error')
            ae={c:abs(pred[t][i]-truth[t][i]) for i,c in enumerate(CHANNELS)}
            target=truth[t][0]+truth[t][2];close(h['target_total_drag'],target,'target Cd clock')
            he['total_cd']=abs(number(h['predicted_total_drag'])-h['target_total_drag'])
            close(he['total_cd'],h['total_drag_absolute_error'],'H1 total Cd')
            ae['total_cd']=abs(pred[t][0]+pred[t][2]-target)
            output.append(dict(case=case['case'],phase=case['case'].split('_')[-2],action=case['case'].split('_')[-1],target=t,time=times[t],omega_now=now,omega_next=nxt,h1_absolute_error=he,ar_absolute_error=ae))
    grouped={}
    for key in ('case','phase','action','target'):
        grouped[key]={str(v):summarize([r for r in output if r[key]==v]) for v in sorted({r[key] for r in output})}
    return dict(status='P064_SAME6_CACHED_H1_AR_DIAGNOSTIC_COMPLETE_NOT_ADMISSION',diagnostic_only=True,
        scientific_admission=False,model_loaded=False,hdf_read=False,endpoint_rows=output,pooled=summarize(output),
        final62_targets39_to100=summarize([r for r in output if r['target']>=39]),groups=grouped,
        interpretation='AR minus true-state absolute error is descriptive, not a causal additive decomposition. H1 signed rearCl is unavailable; do not infer its mean/RMS. Index0 truth seed is excluded. Exact tie counts are descriptive; no acceptance threshold.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','segments','force-window'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'exclusive output exists')
    inputs={};provenance={}
    for name in ('evaluation','segments','force_window'):
        path=getattr(args,name);data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        require(digest==getattr(args,name+'_sha256'),'input SHA differs')
        inputs[name]=json.loads(data);provenance[name]={'path':str(path.resolve()),'sha256':digest}
    result=compare(inputs['evaluation'],inputs['segments'],inputs['force_window'])
    result['input_provenance']=provenance;result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__':main()
