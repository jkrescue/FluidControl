"""Read-only future-time terminal audit; no models/CFD or live-payload audit."""
import json, os, re, subprocess, sys, hashlib, math
from pathlib import Path
import numpy as np

UNIT='fluid-control-p064-b-future-time-cfd-20261007.service'
INV='4b3eb2a2fcab4585aafd5740f8d0cea3'
APPROVAL_SHA='dbcedcce5d680c98e9e611ca092fcd4953879ffef9e8ef9fbf613cd11c93da25'

def check_baseline_rows(rows):
    assert len(rows)==200
    for i,row in enumerate(rows,1):
        assert row['step']==i and abs(row['start_time']-(228+.1*(i-1)))<1e-8 and abs(row['end_time']-(228+.1*i))<1e-8
        h=row['solver_health']['baseline']
        assert h['steps']==20 and h['solver_ended_cleanly'] and h['max_courant']<.8 and h['max_abs_global_continuity_per_step']<1e-5

def zero_table(text,begin,end):
    patch=text.split('rearCylinder',1)[1].split('frontBack',1)[0]
    pairs=[(float(a),float(b)) for a,b in re.findall(r'\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)',patch)]
    assert len(pairs)==2 and all(w==0 for _,w in pairs)
    assert np.allclose([t for t,_ in pairs],[begin,end],rtol=0,atol=1e-8)

def main():
    props=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',UNIT,'-p','InvocationID','-p','MainPID','-p','ExecMainStatus','-p','Result'],text=True).splitlines())
    assert props=={'InvocationID':INV,'MainPID':'0','ExecMainStatus':'0','Result':'success'},'require successful terminal before payload read'
    os.environ.update(CFD_AUDIT_OUTPUT='artifacts/p064_b_future_time_248_328_cfd_20261007',CFD_AUDIT_APPROVAL='docs/P064_B_FUTURE_TIME_CFD_APPROVAL_20261007.json',CFD_START='248',CANONICAL_CFD='1')
    R=Path('/workspace/fluid_control');P=R/os.environ.get('CFD_AUDIT_OUTPUT','artifacts/p064_initial_projected_ppo_long_cfd_20261006')
    start=float(os.environ.get('CFD_START','148'))
    def read(p):return json.loads(p.read_text())
    def sha(p):
     with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    r=read(P/'result.json');approval=R/os.environ.get('CFD_AUDIT_APPROVAL','docs/P064_INITIAL_PROJECTED_PPO_LONG_CFD_APPROVAL_20261006.json');s=read(approval)
    assert r['cycles']==800 and len(r['rows'])==800
    assert sha(approval)==r['approval_sha256']
    # Source/input identities were independently checked before launch; do not
    # repeat that unchanged closure. Audit actual new outputs and scientific pairs.
    assert len(r['raw_file_sha256'])==3200
    arrays={}
    for path,h in r['raw_file_sha256'].items():
     path=Path(path);assert sha(path)==h,path
     role='ppo' if 'case_mpc' in path.parts else 'zero';which='front' if 'forceFront' in path.parts else 'rear'
     values=np.loadtxt(path,comments='#',usecols=(0,1,4));arrays.setdefault((role,which),[]).append(values)
    for k,v in arrays.items():
     a=np.concatenate(v);a=a[np.argsort(a[:,0])];assert a.shape==(16000,3) and np.isfinite(a).all()
     assert np.max(np.abs(a[:,0]-(start+.005*np.arange(1,16001))))<1e-8
     arrays[k]=a
    maxdiff=0.;summary={}
    for name,w in r['windows'].items():
     lo,hi=w['interval'];stats={}
     for role in ['ppo','zero']:
      a=arrays[role,'front'];b=arrays[role,'rear'];t=a[:,0]
      mask=((t>=lo-1e-8) if w['left_endpoint_included'] else (t>lo+1e-8))&(t<=hi+1e-8)
      a=a[mask];b=b[mask]
      expected=w['branches'][role];assert len(a)==expected['samples']
      values={'samples':len(a),'total_cd_mean':np.mean(a[:,1]+b[:,1]),'front_cd_mean':np.mean(a[:,1]),'rear_cd_mean':np.mean(b[:,1]),'front_cl_mean':np.mean(a[:,2]),'rear_cl_mean':np.mean(b[:,2]),'front_cl_fluctuation_rms':np.std(a[:,2]),'rear_cl_fluctuation_rms':np.std(b[:,2]),'front_cl_total_rms':np.sqrt(np.mean(a[:,2]**2)),'rear_cl_total_rms':np.sqrt(np.mean(b[:,2]**2)),'rear_cl_abs_peak':np.max(np.abs(b[:,2]))}
      for key,value in values.items():maxdiff=max(maxdiff,abs(value-expected[key]));assert abs(value-expected[key])<1e-12
      stats[role]=values
     p,z=stats['ppo'],stats['zero'];metrics={'paired_drag_reduction':1-p['total_cd_mean']/z['total_cd_mean'],'paired_rear_cl_fluctuation_rms_ratio':p['rear_cl_fluctuation_rms']/z['rear_cl_fluctuation_rms'],'absolute_mean_rear_cl_over_paired_zero_rms':abs(p['rear_cl_mean'])/z['rear_cl_fluctuation_rms']}
     for key,value in metrics.items():maxdiff=max(maxdiff,abs(value-w[key]));assert abs(value-w[key])<1e-12
     summary[name]={**metrics,'samples':p['samples']}
    previous=0.;maxdelta=0.;maxomega=0.
    def canonicalize(raw):
     o=np.asarray(raw,dtype=np.float32);assert o.shape==(69,) and np.isfinite(o).all()
     reflected=o.copy();reflected[:64]=(o[:64].reshape(32,2)[::-1]*np.array([1.,-1.],dtype=np.float32)).reshape(-1)
     reflected[[65,67,68]]=-o[[65,67,68]]
     odd=o-reflected;pivot=int(np.argmax(np.abs(odd)));margin=float(abs(odd[pivot]));fixed=margin==0
     orientation=1 if fixed or odd[pivot]>0 else -1
     return o if orientation==1 else reflected,orientation,pivot,margin,fixed
    canonical_mode=os.environ.get('CANONICAL_CFD')=='1'
    for i,row in enumerate(r['rows'],1):
     assert row['step']==i and abs(row['start_time']-(start+.1*(i-1)))<1e-8 and abs(row['end_time']-(start+.1*i))<1e-8
     if canonical_mode:
      obs,orientation,pivot,margin,fixed=canonicalize(row['input_observation'])
      assert np.array_equal(obs,np.asarray(row['canonical_observation'],dtype=np.float32))
      assert orientation==row['canonical_orientation_applied'] and pivot==row['canonical_pivot_index']
      assert margin==row['canonical_odd_margin'] and fixed==row['canonical_reflection_fixed']
      request=float(np.float32(orientation*row['canonical_policy_request']))
      assert request==row['physical_requested_omega_before_filter']==row['requested_omega']
      assert canonicalize(row['output_observation'])[1]==row['next_canonical_orientation']
     else:
      request=.5*(row['raw_policy_request']-row['reflected_policy_request']);assert request==row['projected_policy_request']==row['requested_omega']
     applied=float(np.clip(np.clip(request,previous-.1,previous+.1),-.75,.75));assert abs(applied-row['applied_omega'])<1e-7
     assert len(row['input_observation'])==len(row['output_observation'])==len(row['zero_observation'])==69
     assert abs(row['input_observation'][-1]-previous)<1e-7 and abs(row['output_observation'][-1]-applied)<1e-7
     maxdelta=max(maxdelta,abs(applied-previous));maxomega=max(maxomega,abs(applied));previous=row['applied_omega']
    logs=[p for p in P.glob('solver_*.log') if not p.name.startswith('solver_baseline_')];assert len(logs)==1600
    for path in logs:
     text=path.read_text();assert 'FOAM FATAL' not in text and re.search(r'^End\s*$',text,re.M)
     assert len(re.findall(r'^Time = ',text,re.M))==20,path
    assert r['source_restart_unchanged'] and r['owned_containers_cleaned'] and not r['fno_inference'] and not r['mpc_action_selection'] and not r['scientific_admission']
    containers=list(P.glob('container_terminal_*.json'));assert len(containers)==3
    for path in containers:
     c=read(path);c=c[0] if isinstance(c,list) else c
     assert not c['State']['Running'] and not c['State']['OOMKilled']
     assert c['HostConfig']['Memory']==c['HostConfig']['MemorySwap']==8*2**30
     assert not subprocess.check_output(['docker','ps','-aq','--filter','id='+c['Id']],text=True).strip()
    resources=[json.loads(x) for x in (P/'resources.jsonl').read_text().splitlines()]
    minimum=min(x['MemAvailable'] for x in resources);assert minimum>=22*2**30
    print(json.dumps({'result_sha':sha(P/'result.json'),'raw_files':3200,'solver_logs':len(logs),'max_metric_difference':maxdiff,'max_abs_omega':maxomega,'max_abs_delta':maxdelta,'minimum_available_bytes':minimum,'resource_samples':len(resources),'wall_seconds':r['wall_seconds'],'windows':summary},indent=2))


    extra_audit(R,P,r,s,arrays,sha,read)

def extra_audit(R,P,r,s,arrays,sha,read):
    assert sha(R/'docs/P064_B_FUTURE_TIME_CFD_APPROVAL_20261007.json')==APPROVAL_SHA
    assert read(P/'progress.json')=={'completed_cycles':800,'rows':r['rows']}
    baseline=P/'baseline_zero';b=read(P/'baseline_result.json');bp=read(P/'baseline_progress.json')
    assert bp['completed_cycles']==200;check_baseline_rows(bp['rows'])
    assert b==r['baseline_generation'] and b['start_time']==228 and b['end_time']==248 and b['zero_cycles']==200
    for name,digest in b['source_file_sha256'].items():assert sha(baseline/name)==digest
    for name,digest in b['initial_observation_source_sha256'].items():
        p=Path(name);assert p.resolve().is_relative_to(baseline.resolve()) and sha(p)==digest
    for part,files in s['source_restart_tree_sha256'].items():
        for name,digest in files.items():assert sha(R/s['source_restart']/part/name)==digest
    for part,files in r['generated_restart_tree_sha256'].items():
        for name,digest in files.items():assert sha(baseline/part/name)==digest
    assert len(list(P.glob('solver_baseline_*.log')))==200
    for i in range(1,201):
        begin,end=round(228+.1*(i-1),10),round(228+.1*i,10)
        text=(P/f'solver_baseline_{i:02d}.log').read_text()
        times=np.asarray([float(x) for x in re.findall(r'^Time = ([0-9.eE+-]+)',text,re.M)])
        assert times.shape==(20,) and np.allclose(times,begin+.005*np.arange(1,21),rtol=0,atol=1e-8)
        assert re.search(r'^End\s*$',text,re.M) and 'FOAM FATAL' not in text
        zero_table((baseline/f'{begin:g}'/'U').read_text(),begin,end)
    zero_table((baseline/'248/U').read_text(),247.9,248.)
    sys.path.insert(0,str(R/'src'))
    from fluid_control.openfoam_observation import total_drag_observation_at
    obs,sources=total_drag_observation_at(baseline,248.,0.)
    assert sources==b['initial_observation_sources']==r['initial_observation_sources']
    assert np.array_equal(obs,np.asarray(b['initial_observation'],np.float32))
    assert r['rows'][0]['input_observation']==b['initial_observation']
    for i in range(1,800):assert r['rows'][i]['input_observation']==r['rows'][i-1]['output_observation']
    for role,folder in [('ppo','case_mpc'),('zero','case_zero')]:
        for name in ('p','phi','phi_0','U_0'):
            assert (P/folder/'248'/name).read_bytes()==(baseline/'248'/name).read_bytes()
        strip=lambda text:re.sub(r'rearCylinder.*?(?=frontBack)','rearCylinder\n',text,flags=re.S)
        assert strip((P/folder/'248/U').read_text())==strip((baseline/'248/U').read_text())
        for i,row in enumerate(r['rows']):
            j=(i+1)*20-1;actual=row['output_observation' if role=='ppo' else 'zero_observation']
            expected=np.asarray([arrays[role,'front'][j,1],arrays[role,'front'][j,2],arrays[role,'rear'][j,1],arrays[role,'rear'][j,2]],np.float32)
            assert np.allclose(np.asarray(actual[64:68],np.float32),expected,rtol=1e-6,atol=1e-7)
            if role=='zero':
                assert actual[-1]==0
                zero_table((P/folder/f'{row["start_time"]:g}'/'U').read_text(),row['start_time'],row['end_time'])
    print(json.dumps({'status':'FUTURE_TIME_RAW_AUDIT_COMPLETE_NOT_GENERALIZATION','invocation':INV,'baseline_segments':200,'baseline_solver_steps':4000,'paired_segments':1600,'feedback_links':799,'initial69_real_baseline_exact':True,'initial_fields_equal_except_applied_rear_boundary':True,'historical_zero_comparison':'not applicable: new time segment; no old b01 equality requirement'}))

if __name__=='__main__':main()
