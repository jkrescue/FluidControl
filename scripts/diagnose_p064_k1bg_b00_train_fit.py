"""Fixed train-only b00 true-state K1/B/G force-fit diagnostic."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, secrets, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path('/workspace/fluid_control')
STATUS='P064_K1BG_B00_TRAIN_FIT_EXECUTION_APPROVED'
STARTS=tuple(range(0,701,100)); LABELS=('K1','B','G')
KINDS={'K1':'FC_P026_K1_HISTORY_FORCE_FNO','B':'FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO','G':'FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO'}
PRECISION=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False)
PROTOCOL=dict(dataset='b00_projected_ppo_train',split='train',starts=list(STARTS),leads=[1,2,3,4,5],models=list(LABELS),origins=8,true_state_single_step_predictions=120,aero_calls=120,flow_calls=0,optimizer_steps=0,history_k=1,batch_size=1,deadline_seconds=600,allocator_bytes=6*2**30,selection='mechanical_starts_no_dev_selection')
RESOURCES=dict(memory_bytes=12*2**30,swap_bytes=0,cpu_quota_percent=100,tasks_max=64,inner_seconds=600,outer_seconds=630,stop_seconds=20,allocator_bytes=6*2**30,startup_available_gib=50,runtime_available_gib=22,physical_reserve_gib=20)
VERSIONS={'torch':'2.14.1','numpy':'2.5.3','nvidia-physicsnemo':'2.2.2'}

def require(ok,why):
    if not ok: raise ValueError(why)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(item):
    p=ROOT/item['path'];require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT),'bound path');require(sha(p)==item['sha256'],'bound SHA '+str(p));return p
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def array_sha(a):
    import numpy as np
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def metrics(pred,true):
    import numpy as np
    p=np.asarray(pred,np.float64);t=np.asarray(true,np.float64);p=np.column_stack((p,p[:,0]+p[:,2]));t=np.column_stack((t,t[:,0]+t[:,2]));e=p-t
    return dict(mae=np.abs(e).mean(0).tolist(),rmse=np.sqrt((e*e).mean(0)).tolist(),signed_bias=e.mean(0).tolist(),channels=['front_Cd','front_Cl','rear_Cd','rear_Cl','total_Cd'])
def summarize(rows):
    out={}
    for label in LABELS:
        out[label]={}
        for origin in ('pooled',)+tuple(str(x) for x in STARTS):
            out[label][origin]={}
            for lead in range(1,6):
                selected=[r for r in rows if r['model']==label and r['lead']==lead and (origin=='pooled' or r['start']==int(origin))]
                out[label][origin][str(lead)]=metrics([r['prediction'] for r in selected],[r['truth'] for r in selected])
    return out

def validate_sample(sample,metadata):
    import torch
    require(metadata=={'case':PROTOCOL['dataset'],'step':metadata['step'],'rollout_steps':5,'split':'train'} and metadata['step'] in STARTS,'train-only fixed panel')
    shapes={'state':(3,128,256),'target_state':(5,3,128,256),'omega':(6,1),'target_force':(5,4),'mask':(1,128,256)}
    require(all(tuple(sample[k].shape)==v and bool(torch.isfinite(sample[k]).all()) for k,v in shapes.items()),'sample shape/finite')
def prepare_input(sample,lead,build_input,device):
    state=sample['state'] if lead==1 else sample['target_state'][lead-2];mask=sample['mask'];omega=sample['omega']
    packed=build_input(state[None].to(device),mask.to(device),omega[lead-1].reshape(1).to(device),omega[lead].reshape(()).to(device))[None]
    return packed,mask.to(device),state
def aero_force(aero,packed,mask,fm,fs,torch):
    raw=aero(packed);require(tuple(raw.shape)==(1,7,128,256) and bool(torch.isfinite(raw).all()),'aero output')
    return (((raw[:,3:7]*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1))*fs+fm)[0]

def preflight(s,execute=False):
    require(s['protocol']==PROTOCOL and s['resources']==RESOURCES and s['runtime_versions']==VERSIONS,'fixed contract')
    require(s['execution_authorized'] is True and s['status']==STATUS if execute else s['execution_authorized'] is False,'authorization')
    driver=bound(s['driver']) if execute else ROOT/s['driver']['path'];require(driver.resolve()==Path(__file__).resolve() and sha(driver)==s['driver']['sha256'],'driver identity')
    for group in ('sources','data'):
        for item in s[group].values():bound(item)
    runtime_approval=json.loads(bound(s['runtime_approval']).read_text())
    for group in ('sources','runtime_sources'):
        for item in runtime_approval[group].values():bound(item)
    bound(s['config']);bound(s['reference_driver']);bound(s['base'])
    data_root=(ROOT/s['data_root']).resolve();require(data_root.is_dir() and not data_root.is_symlink(),'data root')
    require(bound(s['data']['manifest'])==data_root/'manifest.json' and bound(s['data']['normalization'])==data_root/'normalization.json','data metadata belongs to root')
    require(bound(s['data']['hdf']).parent==data_root/'train','train HDF belongs to root')
    manifests={}
    for label in LABELS:
        mp=bound(s['models'][label]['manifest']);m=json.loads(mp.read_text());manifests[label]=m
        require(m['kind']==KINDS[label] and m['history_input']['history_length']==1,'model identity '+label)
        for role in ('flow','aerodynamic'):
            e=m[role];root=mp.parent/e['checkpoint_relative_directory']
            for field in ('model','state'):require(sha(root/e[field+'_file'])==e[field+'_sha256'],'checkpoint SHA')
    require(len({manifests[x]['flow']['model_sha256'] for x in LABELS})==1 and len({manifests[x]['flow']['state_sha256'] for x in LABELS})==1,'same frozen flow')
    out=ROOT/s['output'];require(out.resolve().is_relative_to(ROOT) and not out.is_symlink(),'output')
    for parent in out.parents:
        if parent==ROOT:break
        require(not parent.is_symlink(),'output ancestor')
    return out

def paired_check(rows):
    expected={(s,j) for s in STARTS for j in range(1,6)};maps={x:{(r['start'],r['lead']):r for r in rows if r['model']==x} for x in LABELS}
    require(len(rows)==120 and all(set(m)==expected for m in maps.values()),'complete120')
    fields=('state_sha256','input_sha256','target_sha256','omega_current','omega_next','time_current','time_target','truth')
    for key in expected:
        require(all(all(maps['K1'][key][f]==maps[x][key][f] for f in fields) for x in ('B','G')),'paired input/target/action')

def worker(s,out,base,reference):
    import numpy as np, torch
    from omegaconf import OmegaConf
    started=json.loads((out/'supervisor_child.json').read_text())['started_monotonic']
    def guard():require(time.monotonic()-started<600 and base.memory_ok(base.memory()),'deadline/reserve')
    sys.path[:0]=[str(ROOT/p) for p in s['pythonpath']]
    from fluid_control.dual_fno import load_dual_fno
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from train_tandem_fno import build_model
    from p026_state_history import build_input
    for key,name in (('dual','fluid_control.dual_fno'),('reader','fluid_control.tandem_datapipe'),('trainer','train_tandem_fno'),('history','p026_state_history')):require(Path(sys.modules[name].__file__).resolve()==bound(s['sources'][key]).resolve(),'import origin')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU');torch.cuda.set_per_process_memory_fraction(min(1.,6*2**30/torch.cuda.get_device_properties(0).total_memory),0)
    norm=json.loads(bound(s['data']['normalization']).read_text());fm=torch.tensor(norm['all_force_mean'],dtype=torch.float32,device='cuda');fs=torch.tensor(norm['all_force_std'],dtype=torch.float32,device='cuda')
    dataset=TandemRolloutDataset(ROOT/s['data_root'],'train',5,stride=1,num_workers=1,force_indices=(0,1,2,3));by_start={}
    require(len(dataset.paths)==1 and dataset.paths[0].resolve()==bound(s['data']['hdf']).resolve(),'single bound train HDF')
    index_for_start={step:index for index,(file_index,step) in enumerate(dataset.index) if file_index==0 and step in STARTS}
    require(set(index_for_start)==set(STARTS),'all indexed starts')
    reader=dataset._reader(0)
    for start,index in index_for_start.items():
        sample,metadata=dataset[index]
        times=[float(reader[start+offset][0]['time']) for offset in range(6)]
        require(abs(times[0]-(148.+start*.1))<1e-5 and all(abs(times[offset]-(times[0]+offset*.1))<1e-5 for offset in range(6)),'actual HDF clock')
        by_start[start]=(sample,metadata,times)
    require(set(by_start)==set(STARTS),'all starts')
    rows=[];counts={'aero':0,'flow':0,'optimizer':0};tensors={}
    for label in LABELS:
        inp=s['models'][label];torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        pair,identity=load_dual_fno(bound(inp['manifest']),OmegaConf.load(bound(s['config'])),torch.device('cuda:0'),build_model=build_model,expected_manifest_sha256=inp['manifest']['sha256']);require(identity.payload['kind']==KINDS[label],'loaded kind')
        models=(pair.flow_model,pair.aerodynamic_model)
        for m in models:m.eval().requires_grad_(False)
        before=reference.tensor_digest(models);require(base.override_inference_precision(torch)['effective']==PRECISION,'precision')
        hooks=[models[0].register_forward_hook(lambda *_:counts.__setitem__('flow',counts['flow']+1)),models[1].register_forward_hook(lambda *_:counts.__setitem__('aero',counts['aero']+1))]
        with torch.inference_mode():
            for start in STARTS:
                sample,metadata,times=by_start[start];validate_sample(sample,metadata)
                for lead in range(1,6):
                    guard();require(base.inference_precision(torch)==PRECISION,'precision changed');packed,mask,state=prepare_input(sample,lead,build_input,'cuda');pred=aero_force(models[1],packed,mask,fm,fs,torch).cpu().numpy();truth=(sample['target_force'][lead-1]*dataset.force_std+dataset.force_mean).numpy()
                    rows.append(dict(model=label,start=start,lead=lead,time_current=times[lead-1],time_target=times[lead],omega_current=float(sample['omega'][lead-1]*dataset.action_scale),omega_next=float(sample['omega'][lead]*dataset.action_scale),state_sha256=array_sha(state.numpy()),input_sha256=array_sha(packed.cpu().numpy()),target_sha256=array_sha(truth),prediction=pred.tolist(),truth=truth.tolist()))
        after=reference.tensor_digest(models);require(before==after and all(p.grad is None for m in models for p in m.parameters()),'model mutation');tensors[label]=dict(before=before,after=after,kind=identity.payload['kind'],manifest_sha256=identity.manifest_sha256)
        for hook in hooks:hook.remove()
        del pair,models,packed;torch.cuda.empty_cache()
    dataset.close();require(counts=={'aero':120,'flow':0,'optimizer':0},'call counts');paired_check(rows);preflight(s,True);guard()
    with (out/'result.json').open('x') as f:json.dump(dict(status='P064_K1BG_B00_TRAIN_FIT_COMPLETE_DIAGNOSTIC_ONLY',interpretation='five true-state single-step predictions, not H5 autoregression',rows=rows,summary=summarize(rows),actual_calls=counts,model_tensors=tensors,protocol=PROTOCOL,scientific_admission=False),f,indent=2,allow_nan=False)

def supervise(out,base,command,environment,approval_sha256,*,process_factory=subprocess.Popen,sleep_fn=time.sleep):
    """E100 lifecycle: continuous memory ledger and owned child cleanup."""
    process=None;error=None;started=time.monotonic()
    try:
        with (out/'run.log').open('x') as log:
            process=process_factory(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
            while True:
                obs=dict(elapsed=time.monotonic()-started,**base.memory())
                with (out/'memory.jsonl').open('a') as stream:stream.write(json.dumps(obs)+'\n')
                require(base.memory_ok(obs) and obs['elapsed']<600,'resource/deadline')
                if process.poll() is not None:require(process.returncode==0,'worker failed');break
                sleep_fn(.5)
            require((out/'result.json').is_file(),'missing result')
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        if process is not None:base.stop(process)
        with (out/'supervisor_result.json').open('x') as stream:json.dump(dict(error=error,returncode=None if process is None else process.returncode,approval_sha256=approval_sha256),stream)

def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);p.add_argument('--approval-sha256',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args();require(sha(a.approval)==a.approval_sha256,'approval SHA');s=json.loads(a.approval.read_text());out=preflight(s,a.execute)
    if not a.execute:print('K1BG_B00_TRAIN_FIT_METADATA_PREFLIGHT_ONLY_PASS');return
    reference=module(bound(s['reference_driver']),'trainfit_reference');base=module(bound(s['base']),'trainfit_base');require(not a.worker or out.is_dir(),'worker output');reference.verify_execution(s,base,out,child=a.worker)
    if a.worker:worker(s,out,base,reference);return
    require(base.memory_ok(base.memory(),True),'startup50');out.mkdir(exist_ok=False);process=None;error=None;started=time.monotonic();token=secrets.token_hex(32)
    with (out/'supervisor_child.json').open('x') as f:json.dump(dict(pid=os.getpid(),token=token,started_monotonic=started),f)
    def stop(*_):raise RuntimeError('termination requested')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    command=[sys.executable,str(Path(__file__).resolve()),'--approval',str(a.approval),'--approval-sha256',a.approval_sha256,'--execute','--worker']
    supervise(out,base,command,dict(os.environ,REPLAY_CHILD_TOKEN=token),a.approval_sha256)
if __name__=='__main__':main()
