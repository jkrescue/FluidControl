"""B continuation on fixed representative256 true-state points; project training adapter."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import copy
import shutil
import sys
import time

ROOT = Path('/workspace/fluid_control')
FROZEN = ('spec_encoder.lift_network.0.conv.bias', 'spec_encoder.lift_network.2.conv.bias')
RESOURCES = dict(memory_bytes=24*2**30, swap_bytes=0, cpu_quota_percent=400,
                 tasks_max=64, inner_seconds=2400, outer_seconds=2430,
                 stop_seconds=20, allocator_bytes=16*2**30,
                 startup_available_gib=50, runtime_available_gib=22, physical_reserve_gib=20)
PROTOCOL = dict(parent='B', source_counts={'original44':192,'controlled_b00':64},
                lead_rule='1+(37*k%100); k zero-based within ScheduleRow.source',
                points=256, microbatch=10, max_outer=200, max_closures=300,
                history_size=5, lr=1., line_search='strong_wolfe', clip=None,
                weight_decay=0., fit_channel_normalized_rmse=.01,
                precision='highest_no_tf32', no_candidate_save=False, no_dev=True)

def require(value, message):
    if not value: raise RuntimeError(message)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def bound(item):
    path=ROOT/item['path']; require(path.resolve().is_relative_to(ROOT) or str(path).startswith('/tmp/p064-representative-fit256-review/'),'source outside scope')
    require(path.is_file() and sha(path)==item['sha256'],'SHA '+str(path)); return path

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m

def preflight(s,execute):
    require(s['protocol']==PROTOCOL and s['resources']==RESOURCES,'fixed contract')
    require(s['execution_authorized'] is execute,'execution authorization')
    require(bound(s['driver']).resolve()==Path(__file__).resolve(),'driver')
    bound(s['core']); reference=module(bound(s['reference_driver']),'fixed_fit_reference')
    bound(s['panel_adapter']);bound(s['panel_receipt']);bound(s['b_training_approval'])
    bound(s['consumer'])
    b=json.loads(bound(s['b_training_approval']).read_text());source_root=Path(b['source_root'])
    require(sha(source_root/'source_manifest.json')==b['source_manifest_sha256'],'B source manifest')
    for name,digest_ in json.loads((source_root/'source_manifest.json').read_text()).items():
        require(sha(source_root/name)==digest_,'B source '+name)
    for item in s['data_files']:bound(item)
    old=json.loads(bound(s['reference_approval']).read_text())
    require(old['models']['B']['manifest']['sha256']=='92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891','B parent')
    copy=dict(old,execution_authorized=False); reference.preflight(copy,False)
    require(s['runtime_versions']==old['runtime_versions'],'runtime contract')
    for name,version in s['runtime_versions'].items():
        require(importlib.metadata.version(name).split('+')[0]==version,'runtime '+name)
    bound(s['reference_result'])
    require(s['reference_result']['sha256']=='bb256396fe174f7d1f8ef3cd68a06bc1653c0e8ff039604714479f44391da26c','matched panel reference')
    out=ROOT/s['output']; require(out.resolve().is_relative_to(ROOT),'output containment')
    require(not any(x.is_symlink() for x in (out,*out.parents)),'output symlink')
    return out,reference,old

def verify_execution(s,child=False):
    props=dict(line.split('=',1) for line in subprocess.check_output(
        ['systemctl','--user','show',s['unit'],'-p','ControlGroup','-p','MainPID',
         '-p','InvocationID','-p','RuntimeMaxUSec','-p','TimeoutStopUSec'],text=True).splitlines())
    require(len(props['InvocationID'])==32,'actual invocation')
    if not child:require(int(props['MainPID'])==os.getpid(),'supervisor owns unit')
    cg=Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/')
    require(int((cg/'memory.max').read_text())==24*2**30,'memory cgroup')
    require((cg/'memory.swap.max').read_text().strip()=='0','no swap')
    require((cg/'pids.max').read_text().strip()=='64','tasks cap')
    quota,period=map(int,(cg/'cpu.max').read_text().split());require(quota/period==4,'CPU4')
    require(props['RuntimeMaxUSec']=='40min 30s' and props['TimeoutStopUSec']=='20s','outer limits')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','one visible GPU')
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'bounded threads')
    return props

def available():
    values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0])*1024

def digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def worker(s,out,reference,old):
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=json.loads((out/'child.json').read_text())['started']
    def guard():
        require(time.monotonic()-started<2400,'deadline')
        require(available()>=22*2**30,'memory floor')
    sys.path[:0]=[str(ROOT/p) for p in old['pythonpath']]
    from fluid_control.dual_fno import load_dual_fno
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from train_tandem_fno import build_model
    from p026_state_history import build_input
    for key,name in [('dual','fluid_control.dual_fno'),('reader','fluid_control.tandem_datapipe'),('trainer','train_tandem_fno'),('history','p026_state_history')]:
        require(Path(sys.modules[name].__file__).resolve()==bound(old['sources'][key]).resolve(),'actual import '+name)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU')
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory)
    parent=old['models']['B']['manifest']
    torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    pair,identity=load_dual_fno(bound(parent),OmegaConf.load(bound(old['config'])),torch.device('cpu'),build_model=build_model,expected_manifest_sha256=parent['sha256'])
    require(identity.manifest_sha256==parent['sha256'] and identity.payload['kind']=='FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO','loaded B')
    pair.flow_model.eval().requires_grad_(False)
    aero=pair.aerodynamic_model.to('cuda').eval()
    for name,p in aero.named_parameters():p.requires_grad_(name not in FROZEN)
    params=[p for p in aero.parameters() if p.requires_grad]
    require(len(params)==28 and tuple(n for n,p in aero.named_parameters() if not p.requires_grad)==FROZEN,'28 exact scope')
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(sum(p.numel()*p.element_size() for p in params)==188890844,'parameter byte identity')
    model_before=digest(aero);flow_before=digest(pair.flow_model)
    bias_before={n:p.detach().clone() for n,p in aero.named_parameters() if n in FROZEN}
    counts={'aero':0,'flow':0};aero.register_forward_hook(lambda *_:counts.__setitem__('aero',counts['aero']+1));pair.flow_model.register_forward_hook(lambda *_:counts.__setitem__('flow',counts['flow']+1))
    panel_module=module(bound(s['panel_adapter']),'representative256_panel')
    b_approval=json.loads(bound(s['b_training_approval']).read_text())
    panel,coverage,tensors=panel_module.build_panel(b_approval,keep_tensors=True)
    import fluid_control.p064_controlled_aero_ab as schedule
    require(sha(schedule.__file__)==b_approval['schedule_adapter_sha256'],'actual schedule module')
    recorded=json.loads(bound(s['panel_receipt']).read_text())
    require(panel==recorded['rows'] and json.loads(json.dumps(coverage))==recorded['summary'],'all256 input/target/time/action identities')
    x=torch.cat([v[0] for v in tensors]).to('cuda');mask=torch.stack([v[1] for v in tensors]).to('cuda');target=torch.stack([v[2] for v in tensors]).to('cuda')
    require(len(x)==256 and len(target)==256,'panel size')
    weights=target.new_tensor([.125,.125,.125,.625]);norm=json.loads(bound(old['data']['normalization']).read_text());fs=target.new_tensor(norm['all_force_std']);fm=target.new_tensor(norm['all_force_mean'])
    def prediction(i):
        guard();raw=aero(x[i:i+10]);require(tuple(raw.shape)==(len(x[i:i+10]),7,128,256),'unchanged seven-output FNO')
        return (raw[:,3:7]*mask[i:i+10]).sum((-2,-1))/mask[i:i+10].sum((-2,-1)).clamp_min(1)
    def objective():
        value=target.new_zeros(())
        for i in range(0,256,10):
            loss=((prediction(i)-target[i:i+10]).square()*weights).sum()/256
            loss.backward();value+=loss.detach()
        return value
    evaluations=[]
    def evaluate():
        pred=torch.cat([prediction(i) for i in range(0,256,10)]);err=pred-target;physical=err*fs
        row=dict(loss=float((err.square()*weights).sum()/256),normalized_rmse=err.square().mean(0).sqrt().tolist(),physical_mae=physical.abs().mean(0).tolist(),physical_rmse=physical.square().mean(0).sqrt().tolist(),physical_bias=physical.mean(0).tolist(),prediction=(pred*fs+fm).cpu().tolist())
        cd=physical[:,0]+physical[:,2]
        row['total_cd_physical']=dict(mae=float(cd.abs().mean()),rmse=float(cd.square().mean().sqrt()),bias=float(cd.mean()))
        evaluations.append(row);return row
    def progress(row):
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    core=module(bound(s['core']),'fixed_fit_core')
    result,optimizer=core.fit_fixed_panel(params,objective,evaluate,progress=progress,seconds=max(0,2340-(time.monotonic()-started)))
    require(counts['flow']==0 and flow_before==digest(pair.flow_model),'frozen flow')
    require(all(torch.equal(dict(aero.named_parameters())[n],v) for n,v in bias_before.items()),'frozen biases')
    require(result['status'] not in ('OPTIMIZATION_FAILURE','RESTORE_FAILURE'),'optimization engineering failure: no candidate')
    result.update(panel=panel,coverage=coverage,model_before=model_before,model_after=digest(aero),flow_digest=flow_before,counts=counts,no_grad_evaluations=len(evaluations),protocol=PROTOCOL,finalization_reserve_seconds=60,scientific_admission=False)
    result['candidate']=save_candidate(s,out,aero,optimizer,pair,identity,old,build_model,OmegaConf,torch,guard,result)
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    require(result['status'] not in ('OPTIMIZATION_FAILURE','RESTORE_FAILURE'),'optimization engineering failure')

def save_candidate(s,out,aero,optimizer,pair,identity,old,build_model,OmegaConf,torch,guard,result):
    """Existing official checkpoint API; new project profile, not a new model."""
    from physicsnemo.utils import save_checkpoint,load_checkpoint
    candidate=out/'candidate';candidate.mkdir()
    payload=json.dumps(PROTOCOL,sort_keys=True,separators=(',',':'))
    (candidate/'training_protocol.json').write_text(payload)
    protocol_sha=sha(candidate/'training_protocol.json')
    info=dict(status='P064_REPRESENTATIVE256_AERODYNAMIC_CHECKPOINT',checkpoint_epoch=1,
              training_experiment='P064-REPRESENTATIVE256',history_profile='p026_k1',history_k=1,
              model_in_channels=6,training_protocol_sha256=protocol_sha,
              training_protocol_file='training_protocol.json',training_points=256,
              optimizer_closures=result['closures'],parent_manifest_sha256=identity.manifest_sha256,
              panel_receipt_sha256=s['panel_receipt']['sha256'],selection_performed=False,
              validation_accessed=False,frozen_test_accessed=False,ppo_executed=False)
    guard();terminal=digest(aero)
    save_checkpoint(candidate/'aerodynamic',models=aero,optimizer=optimizer,epoch=1,metadata=info)
    aero.cpu();optimizer.state.clear();torch.cuda.empty_cache()
    cfg=OmegaConf.load(bound(old['config']));require(int(cfg.model.in_channels)==6,'unchanged K1 six-input aero config');fresh=build_model(cfg);meta={}
    require(load_checkpoint(candidate/'aerodynamic',models=fresh,metadata_dict=meta,device='cpu')==1 and meta==info and digest(fresh)==terminal,'official fresh aero reload')
    del fresh
    parent=json.loads(bound(old['models']['B']['manifest']).read_text());flow_dir=candidate/'flow';flow_dir.mkdir()
    for key in ('model_file','state_file'):
        source=bound(old['models']['B']['manifest']).parent/parent['flow']['checkpoint_relative_directory']/parent['flow'][key]
        shutil.copy2(source,flow_dir/source.name)
    fresh=build_model(cfg)
    require(load_checkpoint(flow_dir,models=fresh,metadata_dict={},device='cpu')==0 and digest(fresh)==result['flow_digest'],'official frozen flow reload')
    del fresh
    manifest=copy.deepcopy(parent)
    for key in ('arm','training_windows','optimizer_steps','accumulation_windows','actual_learning_rate'):
        manifest.pop(key,None)
    manifest.update(status='P064_REPRESENTATIVE256_DUAL_FNO_MANIFEST_VERIFIED',
                    kind='P064_REPRESENTATIVE256_FORCE_FNO',training_experiment='P064-REPRESENTATIVE256',
                    training_points=256,optimizer_closures=result['closures'],
                    training_protocol_sha256=protocol_sha,training_semantics=PROTOCOL,
                    parent_manifest_sha256=identity.manifest_sha256,panel_receipt_sha256=s['panel_receipt']['sha256'],
                    precision_protocol=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False))
    manifest['aerodynamic'].update(checkpoint_relative_directory='aerodynamic',checkpoint_epoch=1,
        model_file='FNO.0.1.mdlus',state_file='checkpoint.0.1.pt',metadata_kind=info['status'],
        model_sha256=sha(candidate/'aerodynamic/FNO.0.1.mdlus'),state_sha256=sha(candidate/'aerodynamic/checkpoint.0.1.pt'))
    manifest['flow']['checkpoint_relative_directory']='flow'
    manifest['aerodynamic_parent_model_sha256']=parent['aerodynamic']['model_sha256']
    manifest['aerodynamic_parent_state_sha256']=parent['aerodynamic']['state_sha256']
    manifest['parent_source_sha256']=manifest.pop('source_sha256',{})
    manifest['source_sha256']={key:s[key]['sha256'] for key in ('driver','core','panel_adapter','consumer')}
    # Consumer protocol/precision contract is reviewed before any execution.
    (candidate/'dual_model_manifest.json').write_text(json.dumps(manifest,indent=2))
    guard()
    return dict(manifest_sha256=sha(candidate/'dual_model_manifest.json'),official_fresh_reload=True,
                aerodynamic_tensor_digest=terminal,flow_tensor_digest=result['flow_digest'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    require(sha(args.spec)==args.sha256,'spec SHA');s=json.loads(args.spec.read_text());out,reference,old=preflight(s,args.execute)
    if not args.execute:print('FIXED_SMALL_FIT_PREPARATION_ONLY_NO_FORWARD');return
    execution=verify_execution(s,args.worker)
    if args.worker:
        child=json.loads((out/'child.json').read_text());require(os.environ.get('FIXED_FIT_CHILD_TOKEN')==child['token'],'child ownership');worker(s,out,reference,old);return
    require(not out.exists() and available()>=50*2**30,'exclusive output/startup memory')
    out.mkdir();started=time.monotonic();token=secrets.token_hex(32)
    (out/'child.json').write_text(json.dumps(dict(started=started,token=token,execution=execution)))
    process=None;error=None
    def stop(*_):raise RuntimeError('termination requested')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        with (out/'worker.log').open('x') as stream:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--spec',str(args.spec),'--sha256',args.sha256,'--execute','--worker'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,FIXED_FIT_CHILD_TOKEN=token))
            while True:
                obs=dict(elapsed=time.monotonic()-started,available_bytes=available())
                with (out/'memory.jsonl').open('a') as f:f.write(json.dumps(obs)+'\n')
                require(obs['elapsed']<2400 and obs['available_bytes']>=22*2**30,'continuous guard')
                if process.poll() is not None:require(process.returncode==0,'worker failed');break
                time.sleep(.5)
    except BaseException as exc:error=repr(exc);raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        with (out/'supervisor_receipt.json').open('x') as f:json.dump(dict(error=error,returncode=None if process is None else process.returncode,approval_sha256=args.sha256),f)
if __name__=='__main__':main()
