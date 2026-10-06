import json,hashlib,math,re,subprocess,os
from pathlib import Path
import numpy as np
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
logs=list(P.glob('solver_*.log'));assert len(logs)==1600
for path in logs:
 text=path.read_text();assert 'FOAM FATAL' not in text and re.search(r'^End\s*$',text,re.M)
 assert len(re.findall(r'^Time = ',text,re.M))==20,path
assert r['source_restart_unchanged'] and r['owned_containers_cleaned'] and not r['fno_inference'] and not r['mpc_action_selection'] and not r['scientific_admission']
containers=list(P.glob('container_terminal_*.json'));assert len(containers)==2
for path in containers:
 c=read(path);c=c[0] if isinstance(c,list) else c
 assert not c['State']['Running'] and not c['State']['OOMKilled']
 assert c['HostConfig']['Memory']==c['HostConfig']['MemorySwap']==8*2**30
 assert not subprocess.check_output(['docker','ps','-aq','--filter','id='+c['Id']],text=True).strip()
resources=[json.loads(x) for x in (P/'resources.jsonl').read_text().splitlines()]
minimum=min(x['MemAvailable'] for x in resources);assert minimum>=22*2**30
print(json.dumps({'result_sha':sha(P/'result.json'),'raw_files':3200,'solver_logs':len(logs),'max_metric_difference':maxdiff,'max_abs_omega':maxomega,'max_abs_delta':maxdelta,'minimum_available_bytes':minimum,'resource_samples':len(resources),'wall_seconds':r['wall_seconds'],'windows':summary},indent=2))

# Matched trained reference and action-square proxy; no solver/model execution.
trained=read(R/os.environ.get('CFD_REFERENCE','artifacts/p064_b_projected_ppo_long_cfd_20261006/result.json'))
oldzero={}
for path,h in trained['raw_file_sha256'].items():
 path=Path(path)
 if 'case_zero' not in path.parts:continue
 assert sha(path)==h
 which='front' if 'forceFront' in path.parts else 'rear'
 oldzero.setdefault(which,[]).append(np.loadtxt(path,comments='#'))
newzero={}
for path in r['raw_file_sha256']:
 path=Path(path)
 if 'case_zero' not in path.parts:continue
 which='front' if 'forceFront' in path.parts else 'rear'
 newzero.setdefault(which,[]).append(np.loadtxt(path,comments='#'))
for which in ('front','rear'):
 a=np.concatenate(oldzero[which]);b=np.concatenate(newzero[which])
 a=a[np.argsort(a[:,0])];b=b[np.argsort(b[:,0])]
 assert a.shape==b.shape and np.array_equal(a,b),(which,a.shape,b.shape)
def action_proxy(data):
 a=np.array([row['applied_omega'] for row in data['rows']],dtype=np.float64)
 assert a.shape==(800,)
 delta=np.diff(np.concatenate(([0.],a)))
 ms=math.fsum(float(x*x) for x in a)/800
 return dict(mean_omega_squared=ms,omega_rms=math.sqrt(ms),
  endpoint_rectangle_integral_omega_squared_D_over_U=.1*math.fsum(float(x*x) for x in a),
  delta_omega_rms=math.sqrt(math.fsum(float(x*x) for x in delta)/800))
comparison={}
for name,w in r['windows'].items():
 t=trained['windows'][name]
 assert w['interval']==t['interval'] and w['left_endpoint_included']==t['left_endpoint_included']
 comparison[name]={'candidate':summary[name],
  'trained':{k:t[k] for k in ('paired_drag_reduction','paired_rear_cl_fluctuation_rms_ratio','absolute_mean_rear_cl_over_paired_zero_rms')}}
print(json.dumps(dict(all_zero_columns_exact=True,
  candidate_action_proxy=action_proxy(r),trained_action_proxy=action_proxy(trained),
  comparison=comparison),indent=2))
