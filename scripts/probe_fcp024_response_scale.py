"""Project-only fixed response scales; no optimizer, backward, or candidate."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import time

P019_SHA='03e0ba375bf131bcc30bbe0d96c8b44bcc5c18ef1ed99de5d9e563e1b6525a64'
P020_SHA='139dee2c3dcd4ee97de78a7a0e343ca9ab9d9708a1cafab447f0d9e4988b5d8c'
CAUSAL_SHA='bcefcad2fa622e5b69133725aa8d39db5a1b0a41a1b5e6c7dcb1f6b7d99a4962'
BLOCK_SHA='367a532b395e2cef06332990ffcbfc769677102858f0ed3a46bb28689d7b92d2'
PRIOR_SHA='adfdd9a86cedf75019b655fe360b64b09d1aa51ce3de2f9166d0d80e296007cb'
PROTOCOL=dict(experiment='FC-P024',scales=[0,1,-1,8,64],endpoint_repeats=2,
    windows_per_panel=6,horizon=100,seed=20261003,allocator_fraction=.06,time_limit_seconds=900,
    objective='unchanged original J0',optimizer_steps=0,backwards=0,candidate_saved=False,heldout_accessed=False)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def load(path,digest,name):
    if sha(path)!=digest:raise ValueError(name+' SHA differs')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def protocol_sha():return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def limits(memory,elapsed,startup=False):
    if elapsed>900:raise RuntimeError('whole diagnostic900second deadline')
    if memory.get('MemFree',0)<(30 if startup else 20) or memory.get('MemAvailable',0)<(50 if startup else 20):raise RuntimeError('physical memory floor')

def finite(value):
    import math
    if isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite(item)
    elif isinstance(value,float) and not math.isfinite(value):raise FloatingPointError('nonfinite diagnostic value')

def read_prior(path):
    import torch
    if sha(path)!=PRIOR_SHA:raise ValueError('P023 result SHA differs')
    prior=json.loads(Path(path).read_text());high=[a for a in prior['arms'] if a['arm']=='HIGH']
    if len(high)!=1 or prior['status']!='FC_P023_INPUT_BLOCK_COMPARISON_COMPLETE_NOT_ADMISSION':raise ValueError('P023 identity differs')
    high=high[0];values=high['records'][-1]['block_update']['values']
    if len(values)!=96:raise ValueError('96 recorded values required')
    vector=torch.tensor(values,dtype=torch.float32).reshape(24,4,1,1)
    if not torch.isfinite(vector).all():raise FloatingPointError('nonfinite source block')
    return prior,high,vector

def apply_scale(model,vector,scale):
    import torch
    if scale not in PROTOCOL['scales'] or vector.shape!=(24,4,1,1) or vector.dtype!=torch.float32:raise ValueError('fixed scale/FP32 vector required')
    if not torch.isfinite(vector).all():raise FloatingPointError('nonfinite vector')
    # Always fresh from the canonical CPU float32 HIGH vector, never cumulative.
    # Multiplying a negative coefficient by zero preserves a negative-zero bit;
    # the original initialized block contains canonical positive zeros.
    scaled=torch.zeros_like(vector) if scale==0 else vector*scale
    if not torch.isfinite(scaled).all():raise FloatingPointError('nonfinite scaled vector')
    with torch.no_grad():model.block.copy_(scaled.to(model.block.device))
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('all parameters must be frozen without gradients')
    return dict(scale=scale,values=scaled.reshape(-1).tolist(),l2=float(scaled.double().norm()),max_abs=float(scaled.abs().max()),identity=model.diagnostic_identity())

def exact_reproduction(panel,reference):
    finite(panel);finite(reference)
    for repeat in (False,True):
        actual=panel['repeat'] if repeat else panel;expected=reference['repeat'] if repeat else reference
        if actual['rows']!=expected['rows'] or actual['aggregate']!=expected['aggregate']:raise RuntimeError('scale0/1 exact raw predictions/metrics/aggregate reproduction failed')

def run_scales(model,vector,prior,high,evaluate):
    results=[]
    for scale in PROTOCOL['scales']:
        record=apply_scale(model,vector,scale)
        if scale in (0,1):
            identity=prior['expanded_initial_identity'] if scale==0 else high['terminal_effective_identity']
            if record['identity']!=identity:raise RuntimeError('scale0/1 block/effective/base identity mismatch')
        panel=evaluate();finite(panel)
        if scale in (0,1):exact_reproduction(panel,high['initial'] if scale==0 else high['terminal'])
        results.append(dict(**record,panel=panel,exact_prior_reproduction=True if scale in (0,1) else None))
    return results

def immutable_panel(model,run,aggregate):
    import numpy as np
    import torch
    before=model.diagnostic_identity();modes={n:m.training for n,m in model.named_modules()}
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('diagnostic gradient scope differs')
    rows=run();result=dict(rows=rows,aggregate=aggregate(rows));finite(result)
    if before!=model.diagnostic_identity() or modes!={n:m.training for n,m in model.named_modules()} or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('panel mutated model/mode/gradient')
    newnp=np.random.get_state()
    if random.getstate()!=py or npstate[0]!=newnp[0] or not np.array_equal(npstate[1],newnp[1]) or npstate[2:]!=newnp[2:] or not torch.equal(cpu,torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if cuda else [])):raise RuntimeError('panel mutated RNG')
    return result

def execute(args,causal,block_module,p020,helper,diagnostic,objective,started):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model,predict,configured_force_indices
    prior,high,vector=read_prior(args.prior_result)
    resources=[];phase='preflight';calls=dict(flow=0,panel=0)
    def guard():
        memory={line.split(':')[0]:int(line.split()[1])/2**20 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemFree:','MemAvailable:'))}
        elapsed=time.monotonic()-started;resources.append(dict(elapsed_seconds=elapsed,phase=phase,**memory));limits(memory,elapsed);return memory
    cfg=OmegaConf.load(args.config);trainer=objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    data_identity=trainer.validate_data_contract(cfg)
    causal_items,mean,std=causal.load_causal_inputs(args.causal_audit,Path(cfg.data.root)/'normalization.json',args.raw_source_view)
    limits(guard(),time.monotonic()-started,startup=True)
    DistributedManager.initialize();dist=DistributedManager()
    if dist.distributed or not dist.cuda or torch.cuda.device_count()!=1:raise RuntimeError('one approved GPU required')
    torch.cuda.set_per_process_memory_fraction(.06,dist.device);precision=validate_runtime_precision()
    flow,legacy=build_model(cfg).to(dist.device),build_model(cfg).to(dist.device)
    for role,parent,epoch in [('flow',flow,0),('aerodynamic',legacy,1)]:
        if load_checkpoint(args.candidate/role,models=parent,metadata_dict={},device=dist.device)!=epoch or objective.tensor_state_sha256(parent)!=helper.TENSORS[role]:raise ValueError('P018 parent differs')
        for parameter in parent.parameters():parameter.requires_grad_(False)
    legacy.cpu();flow.eval()
    if tuple(configured_force_indices(cfg))!=(0,1,2,3):raise ValueError('force order differs')
    base=TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),num_workers=cfg.training.workers,force_indices=(0,1,2,3))
    train=None;model=None;initial_identity=None
    try:
        train,_=compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],rollout_steps=100,stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=(0,1,2,3))
        cached=[];phase='cache_six_cpu_histories'
        for entry in causal_items:
            guard();sample,metadata=train[entry['global_index']];identity=trainer.training_identity([metadata])
            if identity!=entry['identity']:raise ValueError('fixed six identity differs')
            batch={k:v[None].to(dist.device) for k,v in sample.items()}
            def flow_predict(*a):guard();calls['flow']+=1;return predict(*a)
            states=objective.frozen_flow_states(flow,batch['state'],batch['mask'],batch['omega'],flow_predict)
            h1=objective.true_state_inputs(batch['state'],batch['target_state']);mask=batch['mask'];omega=batch['omega'][0]
            cached.append(dict(global_index=entry['global_index'],identity=identity,
                h1=objective.make_inputs(h1[0],mask.expand(100,-1,-1,-1),omega[:-1],omega[1:]).detach().cpu(),
                ar=objective.make_inputs(states[0],mask.expand(100,-1,-1,-1),omega[:-1],omega[1:]).detach().cpu(),
                mask=mask.cpu(),current=entry['normalized'],target=batch['target_force'][0].cpu(),hdf_sha256=entry['hdf_sha256']))
            del sample,batch,states,h1,mask,omega
            print(json.dumps(dict(event='cached_window_cpu',global_index=entry['global_index'])),flush=True)
        flow.cpu();torch.cuda.empty_cache()
        expanded_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));expanded_cfg.model.in_channels=10
        expanded=build_model(expanded_cfg);causal.warmstart(legacy,expanded,inspect.getfile(type(expanded)))
        official_sources=block_module.verify_official_sources(expanded)
        expanded.to(dist.device);expanded.train();model=block_module.FrozenForceBlock(expanded);model.train();model.block.requires_grad_(False)
        initial_identity=model.diagnostic_identity()
        random.seed(PROTOCOL['seed']);np.random.seed(PROTOCOL['seed']);torch.manual_seed(PROTOCOL['seed']);torch.cuda.manual_seed_all(PROTOCOL['seed'])
        def run_window(item):
            guard();data=[item[k].to(dist.device) for k in ('h1','ar','mask','current','target')]
            def counted(*a):guard();calls['panel']+=1;return predict(*a)
            with torch.no_grad():out=causal.recurrent_objective(model,*data,counted,objective.balanced_force_objective,zero_conditioning=False,checkpointed=False)
            loss=dict(h1_balanced=float(out['h1_loss']['balanced']),ar_balanced=float(out['ar_loss']['balanced']),total=float(out['total']),h1_channel_mse=out['h1_loss']['channel_mse'].cpu().tolist(),ar_channel_mse=out['ar_loss']['channel_mse'].cpu().tolist())
            panel=p020.metrics(helper,{d:out[d].detach().cpu() for d in ('h1','ar')},data[-1],std,loss)
            row=dict(global_index=item['global_index'],identity=item['identity'],panel=panel,normalized_predictions={d:out[d].detach().cpu().tolist() for d in ('h1','ar')})
            finite(row);del out,data;guard();print(json.dumps(dict(event='scale_window_complete',phase=phase,global_index=item['global_index'])),flush=True);return row
        panel_count=0
        def evaluate():
            nonlocal phase,panel_count
            phase='scale_'+str(PROTOCOL['scales'][panel_count]);panel_count+=1
            return p020.repeated_panel(lambda:immutable_panel(model,lambda:[run_window(x) for x in cached],p020.aggregate))
        torch.cuda.reset_peak_memory_stats();scales=run_scales(model,vector,prior,high,evaluate)
        if calls!=dict(flow=600,panel=6000):raise RuntimeError('actual call counts differ')
        return dict(status='FC_P024_RESPONSE_SCALE_COMPLETE_NOT_ADMISSION',protocol=PROTOCOL,protocol_sha256=protocol_sha(),prior_result_sha256=PRIOR_SHA,
            scales=scales,aggregate_deltas_from_zero=[dict(scale=r['scale'],delta=p020.numeric_delta(scales[0]['panel']['aggregate'],r['panel']['aggregate'])) for r in scales],
            precision=precision,resources=resources,calls=calls,window_evaluations=60,optimizer_steps=0,backwards=0,expanded_initial_identity=initial_identity,
            official_sources=official_sources,parent_tensors=helper.TENSORS,data_identity=data_identity,hdf_identities=[dict(global_index=x['global_index'],sha256=x['hdf_sha256']) for x in cached],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),candidate_saved=False,validation_accessed=False,frozen_test_accessed=False,scientific_admission=False,
            interpretation='all fixed scales descriptive; no scale selection, no infinitesimal derivative or admission claim; observed repeats are not a rigorous uncertainty bound')
    finally:
        if model is not None and initial_identity is not None:
            model.restore_zero()
            if model.diagnostic_identity()!=initial_identity:raise RuntimeError('final zero restoration failed')
        for role,parent in [('flow',flow),('aerodynamic',legacy)]:
            if objective.tensor_state_sha256(parent)!=helper.TENSORS[role] or any(p.grad is not None for p in parent.parameters()):raise RuntimeError('original parent changed')
        (train if train is not None else base).close()

def main():
    parser=argparse.ArgumentParser()
    for key in ('source-root','diagnostic-script','gradient-helper','comparison-helper','config','candidate','candidate-audit','causal-module','block-adapter','causal-audit','raw-source-view','prior-result','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();started=time.monotonic()
    if not args.execute:raise RuntimeError('separate execution approval required')
    if args.output.exists():raise FileExistsError(args.output)
    try:
        causal=load(args.causal_module,CAUSAL_SHA,'p024_causal');block_module=load(args.block_adapter,BLOCK_SHA,'p024_block');p020=load(args.comparison_helper,P020_SHA,'p024_p020');helper=load(args.gradient_helper,P019_SHA,'p024_p019')
        diagnostic,objective=helper.load_dependencies(args)
        result=execute(args,causal,block_module,p020,helper,diagnostic,objective,started)
        result['source_sha256']={key:sha(getattr(args,key)) for key in ('diagnostic_script','gradient_helper','comparison_helper','causal_module','block_adapter','causal_audit','config','candidate_audit','prior_result')}
        result['harness_sha256']=sha(Path(__file__));result['elapsed_seconds']=time.monotonic()-started
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    except Exception as error:
        with args.output.with_name(args.output.name+'.failure.json').open('x') as stream:json.dump(dict(status='FC_P024_FAILED_NOT_ADMISSION',error=repr(error),elapsed_seconds=time.monotonic()-started),stream,indent=2)
        raise

if __name__=='__main__':main()
