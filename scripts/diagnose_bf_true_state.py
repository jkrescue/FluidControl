"""Fixed saved-development true-state substitution; no optimization/CFD."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

ROOT=Path('/workspace/fluid_control')
STATUS='P064_BF_TRUE_STATE_EXECUTION_APPROVED'
KEYS=[(p,s) for p in ('b01','b03') for s in range(0,701,100)]
PRECISION=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False)
PROTOCOL=dict(origins=16,horizon=5,models=['B','F'],aero_calls=160,flow_calls=0,
              optimizer_steps=0,history_k=1,batch_size=1,h1_atol=0.0,h1_rtol=0.0,
              deadline_seconds=600,allocator_bytes=6*2**30)
RESOURCES=dict(memory_bytes=12*2**30,swap_bytes=0,cpu_quota_percent=100,tasks_max=64,
               inner_seconds=600,outer_seconds=630,stop_seconds=20,allocator_bytes=6*2**30,
               startup_available_gib=50,runtime_available_gib=22,physical_reserve_gib=20)
VERSIONS={'torch':'2.14.1','numpy':'2.5.3','nvidia-physicsnemo':'2.2.2'}
KINDS={'B':'FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO','F':'FC_P064_H1_ONLY_K1_FRESH_FORCE_FNO'}

def require(ok,why):
    if not ok: raise ValueError(why)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def bound(item):
    p=ROOT/item['path']
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT),'bound path')
    require(sha(p)==item['sha256'],'bound SHA '+str(p))
    return p

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def array_sha(a):
    import numpy as np
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def validate_arrays(a,row):
    import numpy as np
    shapes={'initial_state':(3,128,256),'truth_states':(5,3,128,256),
            'predicted_states':(5,3,128,256),'mask':(1,128,256),
            'omega':(6,1),'truth_forces':(6,4),'predicted_forces':(5,4)}
    for k,shape in shapes.items():
        require(a[k].shape==shape and np.isfinite(a[k]).all(),'array shape/finite '+k)
    for k in ('initial_state','truth_states','omega','truth_forces','predicted_forces'):
        require(a[k].dtype==np.float32,'stored FP32 '+k)
    require(np.isfinite(a['x']).all() and np.isfinite(a['y']).all(),'finite grid')
    require(set(np.unique(a['mask'])).issubset({0.,1.}) and a['mask'].sum()>0,'mask')
    t=a['time'].reshape(-1);require(t.shape==(6,) and np.isfinite(t).all(),'time shape')
    start={'b01':130.,'b03':144.}[row['branch']]+row['start_index']*.1
    require(np.max(np.abs(t-(start+np.arange(6)*.1)))<1e-5,'nominal time clock')
    tr=row['transitions'];require(len(tr)==5,'transition count')
    for j,x in enumerate(tr,1):
        require(x['lead']==j,'lead order')
        # Original evaluator casts JSON commands to torch.float32 before / .75.
        require(a['omega'][j-1,0]==np.float32(x['omega_now']) and a['omega'][j,0]==np.float32(x['omega_next']),'action clock')
        require(np.array_equal(a['truth_forces'][j],np.asarray(x['truth_force'],np.float32)),'force clock')
    require(np.array_equal(a['predicted_forces'],np.asarray(row['predicted_forces'],np.float32)),'saved AR force')
    require(np.max(np.abs(a['omega']))<=.75,'action bound')

def current_state(a,j):
    require(1<=j<=5,'lead must be1..5')
    return a['initial_state'] if j==1 else a['truth_states'][j-2]

def prepare_input(a,j,norm,build_input,torch,device):
    # Never normalize saved predicted_states: AR is referenced, not recomputed.
    q=torch.as_tensor(current_state(a,j),dtype=torch.float32,device=device)[None]
    mask=torch.as_tensor(a['mask'],dtype=torch.float32,device=device)
    sm=torch.tensor(norm['state_mean'],dtype=torch.float32,device=device).reshape(1,3,1,1)
    ss=torch.tensor(norm['state_std'],dtype=torch.float32,device=device).reshape(1,3,1,1)
    q=((q-sm)/ss)*mask
    now=torch.tensor([float(a['omega'][j-1,0])],dtype=torch.float32,device=device)/.75
    nxt=torch.tensor(float(a['omega'][j,0]),dtype=torch.float32,device=device)/.75
    return build_input(q,mask,now,nxt)[None],mask

def metrics(pred,true):
    import numpy as np
    p=np.asarray(pred,np.float64);t=np.asarray(true,np.float64)
    p=np.column_stack((p,p[:,0]+p[:,2]));t=np.column_stack((t,t[:,0]+t[:,2]))
    e=p-t
    return dict(mae=np.abs(e).mean(0).tolist(),rmse=np.sqrt((e*e).mean(0)).tolist(),signed_bias=e.mean(0).tolist())

def aero_force(aero,packed,mask,fm,fs,torch):
    raw=aero(packed)
    require(tuple(raw.shape)==(1,7,128,256) and bool(torch.isfinite(raw).all()),'aero output')
    return (((raw[:,3:7]*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1))*fs+fm)[0]

def summarize(rows):
    result={}
    for label in ('B','F'):
        result[label]={}
        for phase in ('pooled','b01','b03'):
            result[label][phase]={}
            for lead in range(1,6):
                selected=[r for r in rows if r['model']==label and r['lead']==lead and (phase=='pooled' or r['phase']==phase)]
                result[label][phase][str(lead)]={stream:metrics([r[stream] for r in selected],[r['truth'] for r in selected]) for stream in ('teacher_forced','saved_ar','hold_origin_force')}
    return result

def descriptive_differences(summary):
    def delta(a,b):return {k:[x-y for x,y in zip(a[k],b[k])] for k in ('mae','rmse','signed_bias')}
    out={}
    for phase in ('pooled','b01','b03'):
        out[phase]={}
        for lead in map(str,range(1,6)):
            b=summary['B'][phase][lead];f=summary['F'][phase][lead]
            out[phase][lead]=dict(F_minus_B={stream:delta(f[stream],b[stream]) for stream in b},
                teacher_minus_savedAR={'B':delta(b['teacher_forced'],b['saved_ar']),
                                      'F':delta(f['teacher_forced'],f['saved_ar'])})
    return out

def paired_input_check(rows):
    keys=lambda r:(r['phase'],r['start'],r['lead'])
    b={keys(r):r for r in rows if r['model']=='B'}
    f={keys(r):r for r in rows if r['model']=='F'}
    expected={(p,s,j) for p,s in KEYS for j in range(1,6)}
    require(set(b)==set(f)==expected and len(rows)==160,'complete paired160 rows')
    for k in expected:
        require(all(b[k][name]==f[k][name] for name in ('current_state_sha256','normalized_input_sha256','time_current','time_target','omega_current','omega_next','truth')),'paired input identity')

def preflight(s,execute=False):
    require(s['protocol']==PROTOCOL,'fixed protocol')
    require(s['resources']==RESOURCES and s['runtime_versions']==VERSIONS,'fixed resources/runtime')
    require(s['execution_authorized'] is True and s['status']==STATUS if execute else s['execution_authorized'] is False,'authorization')
    driver=bound(s['driver']) if execute else Path(s['driver']['path'])
    require(driver.resolve()==Path(__file__).resolve() and sha(driver)==s['driver']['sha256'],'driver identity')
    require(s['reference_results']['B']['sha256']=='47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665'
            and s['reference_results']['F']['sha256']=='642445fc3ad17491d3898c1bd3f49f34062b277302896796b67090a473482cee','fixed results')
    approvals={k:json.loads(bound(v).read_text()) for k,v in s['reference_approvals'].items()}
    for approval in approvals.values():
        for group in ('sources','runtime_sources','inputs'):
            for item in approval[group].values():bound(item)
    f=approvals['F']
    require(approvals['B']['inputs']['normalization']==f['inputs']['normalization'],'same B/F norm')
    require(f['sources']['dual']['sha256']=='f326cffe5d89d5d336dcc10c9a90fed13e0d61470947d2bc54d79cf2fe761131','reviewed common consumer')
    require(f['inputs']['normalization']['sha256']=='f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1','fixed norm')
    for label,approval in approvals.items():
        manifest=json.loads(bound(approval['inputs']['manifest']).read_text())
        require(manifest['kind']==KINDS[label],'exact B/F candidate kind')
        require(manifest['history_input']['history_length']==1,'K1 requires no omitted history')
        for role in ('flow','aerodynamic'):
            entry=manifest[role];parent=bound(approval['inputs']['manifest']).parent/entry['checkpoint_relative_directory']
            for field in ('model','state'):require(sha(parent/entry[field+'_file'])==entry[field+'_sha256'],'checkpoint SHA')
    results={k:json.loads(bound(v).read_text()) for k,v in s['reference_results'].items()}
    for label,r in results.items():
        require(r['status']=='P064_DEVELOPMENT_H1_H5_COMPLETE_NOT_ADMISSION' and r['endpoints']==80 and r['optimizer_steps']==0,'reference terminal')
        require([(x['branch'],x['start_index']) for x in r['records']]==KEYS,'fixed panel')
        require(r['manifest_sha256']==approvals[label]['inputs']['manifest']['sha256'],'model result binding')
        require(r['inference_precision']['effective']==PRECISION and r['action_scale']==.75,'reference precision/action scale')
        for row in r['records']:bound(row['physical_field_artifact'])
    out=ROOT/s['output'];require(out.resolve().is_relative_to(ROOT) and not out.is_symlink(),'output')
    for parent in out.parents:
        if parent==ROOT:break
        require(not parent.is_symlink(),'output ancestor')
    return approvals,results,out

def array_preflight(results):
    import numpy as np
    identity=[]
    for rb,rf in zip(results['B']['records'],results['F']['records']):
        require(rb['transitions']==rf['transitions'],'B/F transition identity')
        with np.load(bound(rb['physical_field_artifact']),allow_pickle=False) as b,np.load(bound(rf['physical_field_artifact']),allow_pickle=False) as f:
            validate_arrays(b,rb);validate_arrays(f,rf)
            for key in ('initial_state','truth_states','predicted_states','mask','x','y','time','omega','truth_forces'):
                require(np.array_equal(b[key],f[key]),'B/F matched array '+key)
            identity.append(dict(phase=rb['branch'],start=rb['start_index'],initial_sha256=array_sha(b['initial_state']),truth_states_sha256=array_sha(b['truth_states'])))
    return identity

def worker(s,approvals,results,out,base,reference):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    started=json.loads((out/'supervisor_child.json').read_text())['started_monotonic']
    def check():
        require(time.monotonic()-started<600 and base.memory_ok(base.memory()),'deadline/reserve')
    check();identities=array_preflight(results);f=approvals['F']
    sys.path[:0]=[str(ROOT/p) for p in f['pythonpath']]
    from fluid_control.dual_fno import load_dual_fno
    from train_tandem_fno import build_model
    from p026_state_history import build_input
    for key,name in (('dual','fluid_control.dual_fno'),('trainer','train_tandem_fno'),('history','p026_state_history')):
        require(Path(sys.modules[name].__file__).resolve()==bound(f['sources'][key]).resolve(),'import origin')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU')
    torch.cuda.set_per_process_memory_fraction(min(1.,6*2**30/torch.cuda.get_device_properties(0).total_memory),0)
    norm=json.loads(bound(f['inputs']['normalization']).read_text())
    for key,n in (('state_mean',3),('state_std',3),('all_force_mean',4),('all_force_std',4)):
        a=np.asarray(norm[key],np.float32);require(a.shape==(n,) and np.isfinite(a).all(),'norm')
        if key.endswith('std'):require((a>0).all(),'positive std')
    rows=[];counts={'aero':0,'flow':0,'optimizer':0};tensor_checks={}
    for label in ('B','F'):
        check();torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        inp=approvals[label]['inputs']
        pair,identity=load_dual_fno(bound(inp['manifest']),OmegaConf.load(bound(inp['config'])),torch.device('cuda:0'),build_model=build_model,expected_manifest_sha256=inp['manifest']['sha256'])
        require(identity.manifest_sha256==inp['manifest']['sha256'] and identity.payload['kind']==KINDS[label],'loaded model identity')
        models=(pair.flow_model,pair.aerodynamic_model)
        for m in models:m.eval().requires_grad_(False)
        before=reference.tensor_digest(models)
        precision=base.override_inference_precision(torch);require(precision['effective']==PRECISION,'postload precision')
        # Runtime hooks make the zero-flow/160-aero claims observable.
        handles=[models[0].register_forward_hook(lambda *_: counts.__setitem__('flow',counts['flow']+1)),models[1].register_forward_hook(lambda *_: counts.__setitem__('aero',counts['aero']+1))]
        fm=torch.tensor(norm['all_force_mean'],device='cuda',dtype=torch.float32);fs=torch.tensor(norm['all_force_std'],device='cuda',dtype=torch.float32)
        with torch.inference_mode():
            for row in results[label]['records']:
                with np.load(bound(row['physical_field_artifact']),allow_pickle=False) as a:
                    for j in range(1,6):
                        check();require(base.inference_precision(torch)==PRECISION,'precision changed')
                        packed,mask=prepare_input(a,j,norm,build_input,torch,'cuda')
                        pred=aero_force(models[1],packed,mask,fm,fs,torch).cpu().numpy()
                        if j==1:require(np.array_equal(pred,a['predicted_forces'][0]),'H1 exact reproduction failed; no retry/tolerance relaxation')
                        truth=a['truth_forces'][j]
                        rows.append(dict(model=label,phase=row['branch'],start=row['start_index'],lead=j,
                            time_current=float(a['time'].reshape(-1)[j-1]),time_target=float(a['time'].reshape(-1)[j]),
                            omega_current=float(a['omega'][j-1,0]),omega_next=float(a['omega'][j,0]),
                            current_state_sha256=array_sha(current_state(a,j)),normalized_input_sha256=array_sha(packed.cpu().numpy()),
                            teacher_forced=pred.tolist(),truth=truth.tolist(),saved_ar=a['predicted_forces'][j-1].tolist(),
                            hold_origin_force=a['truth_forces'][0].tolist(),signed_error=(pred.astype(np.float64)-truth).tolist(),
                            precision=PRECISION,npz=row['physical_field_artifact']))
        after=reference.tensor_digest(models);require(before==after and all(p.grad is None for m in models for p in m.parameters()),'model mutation')
        tensor_checks[label]=dict(before=before,after=after,manifest=inp['manifest'],kind=identity.payload['kind'],
            flow_tensor_sha256=reference.tensor_digest((models[0],)),aero_tensor_sha256=reference.tensor_digest((models[1],)),
            flow_model_sha256=identity.flow.model_sha256,flow_state_sha256=identity.flow.state_sha256,
            aero_model_sha256=identity.aerodynamic.model_sha256,aero_state_sha256=identity.aerodynamic.state_sha256)
        for h in handles:h.remove()
        del pair,models,packed,fm,fs
        torch.cuda.empty_cache()
    require(len(rows)==160 and counts==dict(aero=160,flow=0,optimizer=0),'actual counts')
    require(tensor_checks['B']['flow_tensor_sha256']==tensor_checks['F']['flow_tensor_sha256'],'same frozen flow tensors')
    paired_input_check(rows);summary=summarize(rows)
    preflight(s,True);check()
    result=dict(status='P064_BF_TRUE_STATE_COMPLETE_DIAGNOSTIC_ONLY',rows=rows,summary=summary,
                descriptive_differences=descriptive_differences(summary),
                difference_scope='descriptive_nonadditive_noncausal_not_error_decomposition',
                model_tensors=tensor_checks,actual_calls=counts,inputs=identities,protocol=PROTOCOL,
                reference_results=s['reference_results'],reference_approvals=s['reference_approvals'],scientific_admission=False)
    with (out/'result.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);p.add_argument('--approval-sha256',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true');args=p.parse_args()
    require(sha(args.approval)==args.approval_sha256,'approval SHA')
    s=json.loads(args.approval.read_text());approvals,results,out=preflight(s,args.execute)
    if not args.execute:array_preflight(results);print('BF_REAL_NPZ_PREFLIGHT_ONLY_PASS');return
    reference=module(bound(approvals['F']['driver']),'bf_original_evaluator')
    base=module(bound(approvals['F']['base']),'bf_resource_base')
    require(not args.worker or out.is_dir(),'worker output')
    reference.verify_execution(s,base,out,child=args.worker)
    if args.worker:worker(s,approvals,results,out,base,reference);return
    require(base.memory_ok(base.memory(),True),'startup50');out.mkdir(exist_ok=False)
    process=None;error=None;started=time.monotonic();token=secrets.token_hex(32)
    with (out/'supervisor_child.json').open('x') as f:json.dump(dict(pid=os.getpid(),token=token,started_monotonic=started),f)
    def stop(*_):raise RuntimeError('termination requested')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        with (out/'run.log').open('x') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--approval',str(args.approval),'--approval-sha256',args.approval_sha256,'--execute','--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,REPLAY_CHILD_TOKEN=token))
            while True:
                obs=dict(elapsed=time.monotonic()-started,**base.memory())
                with (out/'memory.jsonl').open('a') as f:f.write(json.dumps(obs)+'\n')
                require(base.memory_ok(obs) and obs['elapsed']<600,'resource/deadline')
                if process.poll() is not None:require(process.returncode==0,'worker failed');break
                time.sleep(.5)
            require((out/'result.json').is_file(),'missing result')
    except BaseException as e:error=repr(e);raise
    finally:
        if process is not None:base.stop(process)
        with (out/'supervisor_result.json').open('x') as f:json.dump(dict(error=error,returncode=None if process is None else process.returncode,approval_sha256=args.approval_sha256),f)

if __name__=='__main__':main()
