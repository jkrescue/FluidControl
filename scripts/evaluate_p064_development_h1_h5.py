"""Fixed retrospective 16 x H5 inference; explicit approval, no optimization."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import time

ROOT = Path('/workspace/fluid_control')
STATUS = 'P064_DEVELOPMENT_H1_H5_EVALUATION_EXECUTION_APPROVED'
BASE_SHA = '7f8567b0f41474294a45a864d4024cf0d64508b7e8351e19feb8e8aee9219b73'
K1_SHA = '7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7'
NORM_SHA = 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1'
IMPORTS = {'selector':'fluid_control.exploratory_short_mpc','dual':'fluid_control.dual_fno',
    'trainer':'train_tandem_fno','history':'p026_state_history'}
VERSIONS = {'torch':'2.14.1','numpy':'2.5.3','nvidia-physicsnemo':'2.2.2'}
RESOURCES = {'memory_bytes':12*2**30,'swap_bytes':0,'cpu_quota_percent':100,
    'tasks_max':64,'supervisor_seconds':600,'outer_seconds':630,'stop_seconds':20,
    'startup_available_gib':50,'runtime_available_gib':22,'gpu_visible':'0'}
UNIT_PROPERTIES = {'MemoryMax':str(12*2**30),'MemorySwapMax':'0',
    'CPUQuotaPerSecUSec':'1s','TasksMax':'64','RuntimeMaxUSec':'10min 30s',
    'TimeoutStopUSec':'20s','KillMode':'control-group'}

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def bound(item):
    p = ROOT / item['path']
    require(p.resolve().is_relative_to(ROOT) and not p.is_symlink() and p.is_file(), 'bound path')
    require(sha(p) == item['sha256'], 'bound SHA')
    return p

def module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m)
    return m

def validate_spec(s):
    require(s.get('status') == STATUS and s.get('execution_authorized') is True, 'not approved')
    require(s.get('evaluation_protocol') == {'starts':[0,100,200,300,400,500,600,700],
        'branches':['b01','b03'], 'horizon':5, 'endpoints':80, 'deadline_seconds':600,
        'allocator_bytes':6*2**30, 'optimizer_steps':0}, 'fixed protocol')
    require(s.get('candidate_label') in ('K1','A','B'), 'fixed candidate label')
    require(s['inputs']['normalization']['sha256'] == NORM_SHA, 'fixed norm')
    require(s['candidate_label'] != 'K1' or s['inputs']['manifest']['sha256'] == K1_SHA, 'original K1')
    require(s['inputs']['config']['sha256'] == '07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9', 'canonical config')
    require(s['base']['sha256'] == BASE_SHA, 'supervisor helper')
    require(s.get('resources') == RESOURCES and s.get('runtime_versions') == VERSIONS,
        'exact resources/runtime contract')
    require(s['import_bindings'] == IMPORTS, 'required project imports')
    require(s['sources']['core']['sha256'] == '3dcfa6d028e49df845b5dbf0fb7e72c04a03c0662244b57a40f6637a386c5681'
        and s['sources']['reader']['sha256'] == '68870966e09ff0ed0beab9ca2021becae5f4ef92e16289e613974a325b4dbab1', 'reviewed adapters')
    require(bound(s['driver']).resolve() == Path(__file__).resolve(), 'executed source')
    for item in s['sources'].values(): bound(item)
    for item in s['runtime_sources'].values(): bound(item)
    inputs = {k:bound(v) for k,v in s['inputs'].items()}
    out = ROOT / s['output']
    require(out.resolve().is_relative_to(ROOT) and not out.is_symlink(), 'output escape')
    for parent in out.parents:
        if parent == ROOT: break
        require(not parent.is_symlink(), 'output ancestor')
    return inputs, out

def verify_execution(s,base,output,child=False):
    limits=base.cgroup_limits()
    require(limits['memory_max']==12*2**30 and limits['memory_swap_max']==0,'exact cgroup limits')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','explicit GPU0')
    require({k:importlib.metadata.version(k) for k in VERSIONS}==VERSIONS,'package versions')
    unit=s['unit'];require(isinstance(unit,str) and unit.endswith('.service'),'unit')
    keys=['MainPID',*UNIT_PROPERTIES]
    raw=subprocess.check_output(['systemctl','--user','show',unit,
        *[v for key in keys for v in ('-p',key)]],text=True,timeout=10)
    actual=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    require(all(actual.get(k)==v for k,v in UNIT_PROPERTIES.items()),'actual outer properties')
    expected_pid=os.getppid() if child else os.getpid()
    require(actual.get('MainPID')==str(expected_pid),'actual supervisor PID')
    if child:
        link=json.loads((output/'supervisor_child.json').read_text())
        require(link['pid']==os.getppid() and link['token']==os.environ.get('REPLAY_CHILD_TOKEN')
            and type(link['token']) is str and len(link['token'])==64,'authorized worker child')
        require(time.monotonic()-link['started_monotonic']<600,'worker deadline')
    return limits

def conversion_records(inputs):
    r=json.loads(inputs['conversion_result'].read_text())
    require(r['status']=='DEVELOPMENT_PHASE_CONVERSION_COMPLETE_NOT_ADMISSION'
        and r['frames']==96 and r['trajectories']==16 and r['endpoints']==80
        and r['split']=='development_already_opened' and r['source_unchanged'] is True
        and r['owned_containers_cleaned'] is True and r['model_loaded'] is False,
        'development conversion receipt')
    require(sha(inputs['selection'])==r['selection_sha256'],'selection binding')
    selection=json.loads(inputs['selection'].read_text())
    expected=[(phase,j) for phase in ('b01','b03') for j in range(0,701,100)]
    records=[]
    for phase in ('b01','b03'):
        panel=selection['phases'][phase]
        require(panel['split']=='development_already_opened' and panel['frames']==48,'phase panel')
        for row in panel['records']:
            require(row['branch']=='mpc' and row['phase']==phase,'controlled phase only')
            records.append(dict(row,branch=phase))
    require([(x['branch'],x['start_index']) for x in records]==expected,'fixed16 panel')
    require([(x['phase'],x['start_index']) for x in r['hdf']]==expected,'HDF order')
    for row in r['hdf']:
        path=Path(row['path'])
        require(row['frames']==6 and row['official_reader_verified'] is True
            and path.resolve().is_relative_to(inputs['conversion_result'].parent)
            and not path.is_symlink() and sha(path)==row['sha256'],'verified HDF identity')
    require(len({(x['mask_sha256'],x['x_sha256'],x['y_sha256']) for x in r['hdf']})==1,'common grid')
    return records,r['hdf']

def load_evaluation_pair(s,manifest,cfg,device,load_dual_fno,build_model,load_bound_k1):
    label=s['candidate_label']
    if label=='K1':
        return load_bound_k1(manifest,cfg,device,load_dual_fno=load_dual_fno,build_model=build_model)
    expected_kind='FC_P064_ARM_'+label+'_CONTROLLED_AERO_FORCE_FNO'
    payload=json.loads(manifest.read_text())
    require(payload['kind']==expected_kind,'explicit P064 arm kind')
    adapter,identity=load_dual_fno(manifest,cfg,device,build_model=build_model,
        expected_manifest_sha256=s['inputs']['manifest']['sha256'])
    require(identity.payload['kind']==expected_kind,'loaded arm identity')
    # Source-pinned P064 loader independently enforces fixed P009 flow,
    # K1 aero parent, arm schedule/training semantics and every checkpoint SHA.
    return adapter.flow_model,adapter.aerodynamic_model,identity

def tensor_digest(models):
    h = hashlib.sha256()
    for model in models:
        for key,tensor in model.state_dict().items():
            h.update(key.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rollout(q, mask, transitions, flow, aero, build_input, torch, force_mean, force_std,
            state_mean, state_std, check):
    """Keep recurrent state normalized; never round-trip through physical normalization."""
    states, forces = [], []
    for tr in transitions:
        check()
        now = torch.tensor([tr['omega_now']], dtype=torch.float32, device=q.device)/.75
        nxt = torch.tensor(tr['omega_next'], dtype=torch.float32, device=q.device)/.75
        packed = build_input(q, mask[0], now, nxt)[None]
        raw, ar = flow(packed), aero(packed)
        require(tuple(raw.shape) == tuple(ar.shape) == (1,7,128,256), 'official output shape')
        require(bool(torch.isfinite(raw).all() and torch.isfinite(ar).all()), 'nonfinite model output')
        f = (ar[:,3:7]*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1)
        f = f*force_std+force_mean
        q = (q+raw[:,:3])*mask
        require(bool(torch.isfinite(q).all() and torch.isfinite(f).all()), 'nonfinite prediction')
        states.append(((q*state_std+state_mean)*mask)[0].cpu().numpy().copy())
        forces.append(f[0].cpu().numpy().copy())
    return states, forces

def worker(s, inputs, output, base):
    started=json.loads((output/'supervisor_child.json').read_text())['started_monotonic']
    def deadline():
        require(time.monotonic()-started<600 and base.memory_ok(base.memory()),'worker deadline/reserve')
    deadline()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    require(Path(sys.prefix).resolve() == (ROOT/'.venv-curator-py312').resolve(), 'official environment')
    for name in ('core','reader'):
        require(name in s['sources'], 'missing adapter')
    core = module(bound(s['sources']['core']), 'replay_core')
    reader = module(bound(s['sources']['reader']), 'replay_reader')
    sys.path[:0] = [str(ROOT/p) for p in s['pythonpath']]
    from fluid_control.exploratory_short_mpc import load_bound_k1
    from fluid_control.dual_fno import load_dual_fno
    from train_tandem_fno import build_model
    from p026_state_history import build_input
    # Every project import must resolve to the approval's actual bound source.
    for key, modname in s['import_bindings'].items():
        require(Path(sys.modules[modname].__file__).resolve() == bound(s['sources'][key]).resolve(), 'import origin')
    records, hdfs = conversion_records(inputs)
    deadline()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one GPU')
    cap = 6*2**30
    torch.cuda.set_per_process_memory_fraction(min(1.,cap/torch.cuda.get_device_properties(0).total_memory),0)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    flow,aero,identity = load_evaluation_pair(s,inputs['manifest'],OmegaConf.load(inputs['config']),
        torch.device('cuda:0'),load_dual_fno,build_model,load_bound_k1)
    deadline()
    precision = base.override_inference_precision(torch)
    for model in (flow,aero): model.eval().requires_grad_(False)
    before = tensor_digest((flow,aero))
    norm = json.loads(inputs['normalization'].read_text())
    require(norm['state_channels']==['u','v','gauge_pressure'], 'state channels')
    for key,n in (('state_mean',3),('state_std',3),('all_force_mean',4),('all_force_std',4)):
        a=np.asarray(norm[key],dtype=np.float32)
        require(a.shape==(n,) and np.isfinite(a).all(), 'normalization shape/finite')
        if key.endswith('std'): require(bool((a>0).all()), 'positive normalization scale')
    sm = torch.tensor(norm['state_mean'],device='cuda',dtype=torch.float32).reshape(1,3,1,1)
    ss = torch.tensor(norm['state_std'],device='cuda',dtype=torch.float32).reshape(1,3,1,1)
    fm = torch.tensor(norm['all_force_mean'],device='cuda',dtype=torch.float32)
    fs = torch.tensor(norm['all_force_std'],device='cuda',dtype=torch.float32)
    def check():
        deadline()
        require(base.memory_ok(base.memory()), 'physical reserve')
        require(base.inference_precision(torch) == precision['effective'], 'precision changed')
    metric_rows, serialized = [], []
    anchor = None
    with torch.inference_mode():
        for record,hdf in zip(records,hdfs):
            deadline()
            fields,static = reader.read_trajectory(Path(hdf['path']))
            deadline()
            require(tuple(fields['state'].shape)==(6,3,128,256)
                and tuple(fields['mask'].shape)==(6,1,128,256)
                and tuple(fields['force'].shape)==(6,4)
                and tuple(fields['omega'].shape)==(6,1), 'exact six-frame trajectory')
            for i in range(6):
                packet = {k:fields[k][i].numpy() for k in ('state','mask','time')}
                packet.update(static)
                core.validate_packet(packet,expected_time=record['frames'][i]['time'],expected_mask=anchor)
                if anchor is None: anchor = packet['mask'].copy()
            expected_omega = np.asarray([record['transitions'][0]['omega_now']]+[t['omega_next'] for t in record['transitions']],np.float32)[:,None]
            expected_force = np.asarray([record['initial_force']]+[t['truth_force'] for t in record['transitions']],np.float32)
            require(np.array_equal(fields['omega'].numpy(),expected_omega) and np.array_equal(fields['force'].numpy(),expected_force), 'action/force clock')
            mask = fields['mask'][0:1].float().to('cuda')
            q = ((fields['state'][0:1].to('cuda')-sm)/ss)*mask
            states,forces = rollout(q,mask,record['transitions'],flow,aero,build_input,torch,fm,fs,sm,ss,check)
            row = {'mask':anchor,'truth_states_initial':fields['state'][0].numpy(),
                'truth_states':[x.numpy() for x in fields['state'][1:]],'predicted_states':states,
                'truth_force_initial':expected_force[0],'truth_forces':list(expected_force[1:]),'predicted_forces':forces}
            metric_rows.append(row)
            field_path=output/f"fields_{record['branch']}_{record['start_index']:04d}.npz"
            with field_path.open('xb') as f:
                np.savez_compressed(f, initial_state=row['truth_states_initial'],
                    truth_states=np.asarray(row['truth_states']), predicted_states=np.asarray(states),
                    mask=anchor, x=static['x'], y=static['y'], time=fields['time'].numpy(),
                    omega=expected_omega, truth_forces=expected_force,
                    predicted_forces=np.asarray(forces))
            serial = {'branch':record['branch'],'start_index':record['start_index'],
                'transitions':record['transitions'],'predicted_forces':[x.tolist() for x in forces],
                'physical_field_artifact':{'path':str(field_path),'sha256':sha(field_path)},
                'metrics':core.sufficient_statistics([row])}
            serialized.append(serial)
            with (output/'progress.jsonl').open('a') as f: f.write(json.dumps(serial,allow_nan=False)+'\n')
    require(tensor_digest((flow,aero)) == before, 'model tensors changed')
    require(all(p.grad is None for m in (flow,aero) for p in m.parameters()), 'gradient appeared')
    result = {'status':'P064_DEVELOPMENT_H1_H5_COMPLETE_NOT_ADMISSION', 'candidate_label':s['candidate_label'], 'split':'development_already_opened', 'records':serialized,
        'summary':core.sufficient_statistics(metric_rows),'phases':{
            b:core.sufficient_statistics(metric_rows[i*8:(i+1)*8]) for i,b in enumerate(('b01','b03'))},
        'endpoints':80,'model_tensors_unchanged':True,'manifest_sha256':identity.manifest_sha256,
        'inference_precision':precision,'allocator_bytes':cap,'optimizer_steps':0,
        'action_scale':.75,'scope':'RETROSPECTIVE_REALIZED_ACTION_CONDITIONAL_REPLAY_NOT_ONLINE_FUTURE_ACTION_FORECAST',
        'scientific_admission':False,'conversion_result_sha256':sha(inputs['conversion_result'])}
    with (output/'result.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False)

def supervise(s,inputs,output,base,args):
    limits = verify_execution(s,base,output); require(base.memory_ok(base.memory(),True),'startup reserve')
    output.mkdir(exist_ok=False)
    process = None; error = None; started = time.monotonic()
    token=secrets.token_hex(32)
    with (output/'supervisor_child.json').open('x') as f:
        json.dump({'pid':os.getpid(),'token':token,'started_monotonic':started},f)
    child_env=dict(os.environ,REPLAY_CHILD_TOKEN=token)
    def terminated(*_): raise RuntimeError('termination requested')
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,terminated)
    try:
        with (output/'run.log').open('x') as log:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--execute',
                '--approval',str(args.approval),'--approval-sha256',args.approval_sha256],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=child_env)
            while True:
                row = {'elapsed':time.monotonic()-started,**base.memory()}
                with (output/'memory.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                require(base.memory_ok(row) and row['elapsed']<600,'resource/deadline')
                code = process.poll()
                if code is not None:
                    require(code == 0,'worker failed'); break
                time.sleep(.5)
        require((output/'result.json').is_file(),'missing result')
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        if process is not None: base.stop(process)
        with (output/'supervisor_result.json').open('x') as f:
            json.dump({'error':error,'returncode':None if process is None else process.returncode,
                'limits':limits,'approval_sha256':args.approval_sha256},f,indent=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--approval-sha256',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true');args=p.parse_args()
    require(sha(args.approval)==args.approval_sha256,'approval digest')
    s=json.loads(args.approval.read_text());inputs,output=validate_spec(s)
    if not args.execute: print('METADATA_PREFLIGHT_ONLY');return
    base=module(bound(s['base']),'replay_resource_base')
    if args.worker:
        verify_execution(s,base,output,child=True);worker(s,inputs,output,base)
    else: supervise(s,inputs,output,base,args)

if __name__=='__main__': main()
