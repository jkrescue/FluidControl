"""Project engineering probe, one train window; no optimizer or candidate."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

P019_SHA='03e0ba375bf131bcc30bbe0d96c8b44bcc5c18ef1ed99de5d9e563e1b6525a64'
MODULE_SHA='bcefcad2fa622e5b69133725aa8d39db5a1b0a41a1b5e6c7dcb1f6b7d99a4962'

def sha(path):
    import hashlib
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def pinned_module(path,digest,name):
    if sha(path)!=digest:raise ValueError(name+' source differs')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def host_memory():
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemFree:'))}
    return {k:v/2**30 for k,v in values.items()}

def check_limits(memory,elapsed,*,startup=False):
    if elapsed>900:raise RuntimeError('whole-probe900second deadline')
    if memory.get('MemFree',0)<(30 if startup else 20) or memory.get('MemAvailable',0)<(50 if startup else 20):raise RuntimeError('physical memory floor')

def gradient_report(model,lift):
    import torch
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if len(parameters)!=28 or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):raise RuntimeError('28 finite gradients required')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('frozen parameter received gradient')
    nbytes=sum(p.numel()*p.element_size() for p in parameters.values())
    return dict(trainable_tensors=28,new_force_column_gradient_norm=float(parameters[lift].grad[:,6:10].double().norm()),
        trainable_parameter_bytes=nbytes,adam_two_moment_bytes_projection=2*nbytes,update_temporary_bytes_allowance=nbytes,
        projection_only_not_measured_optimizer=True)

def verify_identities(flow,legacy,expanded,objective,helper,expanded_sha):
    for role,model in [('flow',flow),('aerodynamic',legacy)]:
        if objective.tensor_state_sha256(model)!=helper.TENSORS[role]:raise RuntimeError('original parent changed')
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError('original parent gradient')
    if objective.tensor_state_sha256(expanded)!=expanded_sha:raise RuntimeError('expanded initial tensors changed')
    if any(p.grad is not None for p in expanded.parameters()):raise RuntimeError('expanded gradients not cleared')

def execute(args,causal,helper,diagnostic,objective,started):
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model,predict,configured_force_indices
    resources=[];phase='preflight'
    def guard(startup=False):
        memory=host_memory();elapsed=time.monotonic()-started;check_limits(memory,elapsed,startup=startup)
        resources.append(dict(elapsed_seconds=elapsed,phase=phase,**memory))
        print(json.dumps(dict(event='resource_observation',**resources[-1])),flush=True)
    cfg=OmegaConf.load(args.config)
    trainer=objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    data_identity=trainer.validate_data_contract(cfg)
    causal_items,mean,std=causal.load_causal_inputs(args.causal_audit,Path(cfg.data.root)/'normalization.json',args.raw_source_view)
    selected=[x for x in causal_items if x['global_index']==816]
    if len(selected)!=1:raise ValueError('causal window816 missing')
    guard(startup=True)
    DistributedManager.initialize();dist=DistributedManager()
    if dist.distributed or not dist.cuda:raise RuntimeError('single approved CUDA device required')
    torch.cuda.set_per_process_memory_fraction(.06,dist.device);precision=validate_runtime_precision()
    if torch.cuda.device_count()!=1:raise RuntimeError('one visible GPU required')
    phase='parent_load';guard()
    flow,legacy=build_model(cfg).to(dist.device),build_model(cfg).to(dist.device)
    for role,model,epoch in [('flow',flow,0),('aerodynamic',legacy,1)]:
        if load_checkpoint(args.candidate/role,models=model,metadata_dict={},device=dist.device)!=epoch:raise ValueError('parent epoch differs')
        if objective.tensor_state_sha256(model)!=helper.TENSORS[role]:raise ValueError('parent tensor differs')
    flow.eval();legacy.train();causal.validate_paired_model(legacy)
    for p in flow.parameters():p.requires_grad_(False)
    if tuple(configured_force_indices(cfg))!=(0,1,2,3):raise ValueError('force schema differs')
    base=TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),num_workers=cfg.training.workers,force_indices=(0,1,2,3))
    train=None;expanded=None;initial_sha=None
    try:
        train,_=compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],rollout_steps=100,stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=(0,1,2,3))
        sample,metadata=train[816];identity=trainer.training_identity([metadata])
        if identity!=selected[0]['identity']:raise ValueError('fixed window identity differs')
        batch={k:v[None].to(dist.device) for k,v in sample.items()}
        phase='frozen_flow_history';flow_calls=0
        def flow_predict(*a):
            nonlocal flow_calls
            guard();flow_calls+=1;return predict(*a)
        states=objective.frozen_flow_states(flow,batch['state'],batch['mask'],batch['omega'],flow_predict)
        h1=objective.true_state_inputs(batch['state'],batch['target_state'])
        mask=batch['mask'];omega=batch['omega'][0];target=batch['target_force'][0]
        h1_inputs=objective.make_inputs(h1[0],mask.expand(100,-1,-1,-1),omega[:-1],omega[1:]).detach()
        ar_inputs=objective.make_inputs(states[0],mask.expand(100,-1,-1,-1),omega[:-1],omega[1:]).detach()
        del states,h1,batch,sample
        flow.cpu();torch.cuda.empty_cache()
        phase='legacy_paired_reference';reference=[]
        with torch.no_grad():
            for k in range(100):
                guard();_,forces=predict(legacy,torch.stack((h1_inputs[k],ar_inputs[k])),mask.expand(2,-1,-1,-1));reference.append(forces.cpu())
        reference=torch.stack(reference)
        if not torch.isfinite(reference).all():raise FloatingPointError('legacy reference nonfinite')
        legacy.cpu();torch.cuda.empty_cache()
        expanded_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));expanded_cfg.model.in_channels=10
        expanded=build_model(expanded_cfg);causal.warmstart(legacy,expanded,inspect.getfile(type(expanded)))
        expanded.train();expanded.to(dist.device);initial_sha=objective.tensor_state_sha256(expanded)
        current=selected[0]['normalized'].to(dist.device);arms=[]
        for arm,zero in [('A_zero',True),('B_causal',False)]:
            phase=arm;guard();expanded.zero_grad(set_to_none=True)
            verify_identities(flow,legacy,expanded,objective,helper,initial_sha)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic();calls={'forward':0,'recompute':0};backward=False
            def counted(*a):
                guard();calls['recompute' if backward else 'forward']+=1
                return predict(*a)
            out=causal.recurrent_objective(expanded,h1_inputs,ar_inputs,mask,current,target,counted,objective.balanced_force_objective,zero_conditioning=zero)
            backward=True;out['total'].backward();torch.cuda.synchronize();elapsed=time.monotonic()-begin
            gradients=gradient_report(expanded,causal.LIFT)
            if zero and gradients['new_force_column_gradient_norm']!=0:raise RuntimeError('zero-input arm has nonzero new-column gradient')
            predicted=torch.stack((out['h1'],out['ar']),1).detach().cpu()
            arms.append(dict(arm=arm,elapsed_seconds=elapsed,calls=calls,gradients=gradients,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                original_normalized_objective=dict(h1=float(out['h1_loss']['balanced'].detach()),ar=float(out['ar_loss']['balanced'].detach()),total=float(out['total'].detach())),
                initialization_legacy_output_max_abs_difference=float((predicted-reference).abs().max()),
                predicted_physical_forces=(predicted*std+mean).tolist()))
            expanded.zero_grad(set_to_none=True);del out,predicted
            verify_identities(flow,legacy,expanded,objective,helper,initial_sha);guard()
            print(json.dumps(dict(event='arm_backward_complete',arm=arm,elapsed_seconds=elapsed,calls=calls)),flush=True)
        return dict(status='FC_P021_ONE_WINDOW_RESOURCE_COMPLETE_NOT_ADMISSION',scientific_admission=False,identity=identity,
            arms=arms,flow_forward_calls=flow_calls,legacy_reference_calls=100,expanded_initial_tensor_sha256=initial_sha,
            selected_hdf_sha256_from_pinned_audit=selected[0]['hdf_sha256'],
            original_parent_tensors=helper.TENSORS,parent_and_expanded_unchanged=True,resources=resources,precision=precision,data_identity=data_identity,
            memory_strategy='frozen flow and legacy force CPU-offloaded after histories/reference; no optimizer',allocator_fraction=.06,
            projected_192_window_seconds=96*sum(x['elapsed_seconds'] for x in arms),projection_caveat='not measured full training; excludes optimizer/panels/setup',
            target_physical_forces=(target.detach().cpu()*std+mean).tolist())
    finally:
        if expanded is not None:expanded.zero_grad(set_to_none=True)
        if initial_sha is not None:
            verify_identities(flow,legacy,expanded,objective,helper,initial_sha)
            print(json.dumps(dict(event='cleanup_identities_verified',expanded_initial_tensor_sha256=initial_sha)),flush=True)
        if train is not None:train.close()
        else:base.close()

def main():
    parser=argparse.ArgumentParser()
    for name in ['source-root','diagnostic-script','gradient-helper','config','candidate','candidate-audit','causal-module','causal-audit','raw-source-view','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();started=time.monotonic()
    if not args.execute:raise RuntimeError('explicit execution approval required')
    if args.output.exists():raise FileExistsError(args.output)
    try:
        causal=pinned_module(args.causal_module,MODULE_SHA,'p021_causal');helper=pinned_module(args.gradient_helper,P019_SHA,'p021_p019')
        diagnostic,objective=helper.load_dependencies(args)
        result=execute(args,causal,helper,diagnostic,objective,started)
        result['source_sha256']={name:sha(getattr(args,name)) for name in ['gradient_helper','diagnostic_script','causal_module','config','candidate_audit','causal_audit']}
        result['harness_sha256']=sha(Path(__file__));result['elapsed_seconds']=time.monotonic()-started
        with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    except Exception as error:
        failure=args.output.with_name(args.output.name+'.failure.json')
        with failure.open('x') as stream:json.dump(dict(status='FC_P021_RESOURCE_FAILED_NOT_ADMISSION',error=repr(error),elapsed_seconds=time.monotonic()-started),stream,indent=2)
        raise

if __name__=='__main__':main()
