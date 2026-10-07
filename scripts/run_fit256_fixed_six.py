"""Preparation-default, bounded B/candidate original-six same-precision evaluation."""
import argparse, hashlib, importlib.util, importlib.metadata, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path('/workspace/fluid_control')
RES=dict(memory_bytes=24*2**30,allocator_bytes=16*2**30,cpu=4,tasks=64,
         seconds=1200,outer_seconds=1230,start_gib=50,run_gib=22)
def require(v,m):
    if not v:raise RuntimeError(m)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(x):
    p=ROOT/x['path'];require(sha(p)==x['sha256'],'binding '+str(p));return p
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def avail():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
def preflight(s,execute):
    require(s['execution_authorized'] is execute and s['resources']==RES,'authorization/resources')
    require(bound(s['driver']).resolve()==Path(__file__).resolve(),'driver')
    for k in ('helper','consumer','training_approval'):bound(s[k])
    train=json.loads(bound(s['training_approval']).read_text());old=json.loads(bound(train['b_training_approval']).read_text())
    for name,version in train['runtime_versions'].items():
        require(importlib.metadata.version(name).split('+')[0]==version,'runtime '+name)
    root=Path(old['source_root']);manifest=root/'source_manifest.json'
    require(sha(manifest)==old['source_manifest_sha256'],'base closure manifest')
    for rel,h in json.loads(manifest.read_text()).items():require(sha(root/rel)==h,'base source '+rel)
    if execute:
        for k in ('training_result','training_review','candidate_manifest','b_manifest'):bound(s[k])
        terminal=json.loads(bound(s['training_result']).read_text())
        require(s['training_engineering_accept'] is True,'prior terminal engineering acceptance')
        require(terminal['candidate']['official_fresh_reload'] is True,'fresh official reload required')
        require(terminal['candidate']['manifest_sha256']==s['candidate_manifest']['sha256'],'terminal candidate manifest')
    return train,old,root
def worker(s,out,started):
    train,old,root=preflight(s,True)
    sys.path[:0]=[str(root/'src'),str(root/'scripts')]
    import torch
    from omegaconf import OmegaConf
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.augmented_datapipe import compose_training_data
    from train_tandem_fno import build_model,predict
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU')
    torch.cuda.set_per_process_memory_fraction(RES['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
    def guard():require(time.monotonic()-started<RES['seconds'] and avail()>=RES['run_gib']*2**30,'resource/deadline')
    def mod(rel,n):return load(root/rel,n)
    original=mod('scripts/train_fcp064_controlled_aero_ab.py','six_original_runner')
    diagnostic=mod('scripts/diagnose_fcp014_train_objective.py','six_diagnostic')
    objective=diagnostic.load_objective(root)
    trainer=objective.load_frozen_trainer(root/'scripts/train_fcp011_decoder_scope.py')
    chunk=mod('scripts/p026_history_objective.py','six_chunk')
    history=mod('scripts/p026_state_history.py','six_history')
    helper=mod('scripts/probe_fcp019_gradient_alignment.py','six_metric_helper')
    p020=mod('scripts/probe_fcp020_symmetric_statistics.py','six_p020')
    evaluation=load(bound(s['helper']),'six_evaluation')
    consumer=load(bound(s['consumer']),'six_consumer')
    cfg=OmegaConf.load(root/'training_config.yaml')
    roots=[Path(old['training_views'][k]['root']) for k in ('base','train8','train16')]
    for key,path in zip(('base','train8','train16'),roots):
        entry=old['training_views'][key]
        require(sha(path/'manifest.json')==entry['manifest_sha256'],'data manifest')
        require(sha(path/'normalization.json')==entry['normalization_sha256'],'old normalization')
    base=TandemRolloutDataset(roots[0],'train',100,stride=20,num_workers=1,force_indices=(0,1,2,3))
    dataset=None;panels={};identities={}
    try:
        dataset,_=compose_training_data(base,roots[1:],rollout_steps=100,stride=2,workers=1,force_indices=(0,1,2,3))
        require(len(dataset)==1368 and len(dataset._datasets)==3,'original44 composition')
        # Verify only the actual six HDF sources against the bound training inventory.
        inventory={str((ROOT/x['path']).resolve()):x['sha256'] for x in train['data_files']}
        for item in trainer.diagnostic_windows(dataset):
            ident=item['identity'];child=dataset._datasets[ident['dataset_index']]
            paths=[p for p in child.paths if p.stem==ident['case']];require(len(paths)==1,'six HDF identity')
            require(sha(paths[0])==inventory[str(paths[0].resolve())],'six HDF SHA')
        for label,key in (('B','b_manifest'),('candidate','candidate_manifest')):
            guard()
            torch.set_float32_matmul_precision('high' if label=='B' else 'highest')
            torch.backends.cuda.matmul.allow_tf32=label=='B';torch.backends.cudnn.allow_tf32=label=='B'
            pair,identity=consumer.load_dual_fno(bound(s[key]),cfg,torch.device('cuda:0'),build_model=build_model,expected_manifest_sha256=s[key]['sha256'])
            for m in (pair.flow_model,pair.aerodynamic_model):m.eval().requires_grad_(False)
            panels[label]=evaluation.evaluate_fixed_six(flow=pair.flow_model,aero=pair.aerodynamic_model,dataset=dataset,base=base,device='cuda:0',trainer=trainer,original_runner=original,objective=objective,chunk=chunk,history=history,p020=p020,helper=helper,predict=predict,guard=guard)
            identities[label]=dict(manifest_sha256=identity.manifest_sha256,kind=identity.payload['kind'])
            del pair
        with (out/'result.json').open('x') as f:json.dump(dict(status='P064_FIT256_FIXED_SIX_COMPLETE_NOT_ADMISSION',panels=panels,identities=identities,optimizer_steps=0,candidate_saved=False,validation_accessed=False,frozen_test_accessed=False),f,indent=2,allow_nan=False)
    finally:
        if dataset is not None:dataset.close()
        else:base.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--child',action='store_true');a=p.parse_args()
    require(sha(a.spec)==a.sha256,'spec');s=json.loads(a.spec.read_text());preflight(s,a.execute)
    if not a.execute:print('FIXED_SIX_PREPARATION_NO_MODEL_NO_FORWARD');return
    out=ROOT/s['output'];require(out.resolve().is_relative_to(ROOT),'output containment')
    if a.child:
        receipt=json.loads((out/'launch.json').read_text());require(os.environ.get('SIX_CHILD_TOKEN')==receipt['token'],'child token');worker(s,out,receipt['started']);return
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',s['unit'],'-p','MainPID','-p','InvocationID','-p','ControlGroup','-p','RuntimeMaxUSec','-p','TimeoutStopUSec'],text=True).splitlines())
    require(int(props['MainPID'])==os.getpid() and len(props['InvocationID'])==32,'owned unit')
    cg=Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/')
    require(int((cg/'memory.max').read_text())==RES['memory_bytes'] and (cg/'memory.swap.max').read_text().strip()=='0','memory limits')
    q,period=map(int,(cg/'cpu.max').read_text().split());require(q/period==4 and int((cg/'pids.max').read_text())==64,'CPU/tasks')
    require(props['RuntimeMaxUSec']=='20min 30s' and props['TimeoutStopUSec']=='20s','time limits')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','one visible GPU')
    require(not out.exists() and avail()>=50*2**30,'startup')
    out.mkdir();started=time.monotonic();token=os.urandom(32).hex();(out/'launch.json').write_text(json.dumps(dict(started=started,token=token,unit=props,approval_sha256=a.sha256)))
    process=None;error=None
    def stop(*_):raise RuntimeError('termination requested')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        with (out/'worker.log').open('x') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--spec',str(a.spec.resolve()),'--sha256',a.sha256,'--execute','--child'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,SIX_CHILD_TOKEN=token))
            while True:
                row=dict(elapsed=time.monotonic()-started,available_bytes=avail())
                with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                require(row['elapsed']<RES['seconds'] and row['available_bytes']>=22*2**30,'continuous guard')
                if process.poll() is not None:require(process.returncode==0,'worker failure');break
                time.sleep(.5)
    except BaseException as ex:error=repr(ex);raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        (out/'supervisor_receipt.json').write_text(json.dumps(dict(error=error,returncode=None if process is None else process.returncode,approval_sha256=a.sha256)))
if __name__=='__main__':main()
