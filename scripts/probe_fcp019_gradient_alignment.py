#!/usr/bin/env python3
"""Staged train-only P018 gradient geometry; no optimizer or saved candidate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

P014_SHA='849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d'
AUDIT_SHA='03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9'
FILES={
 'dual_model_manifest.json':'91cc2c9a295a1ace5eadcffd8242234b93b73d14959bb902e17b59655f4acf13',
 'result.json':'5ca668810110676dc500f501fb602a3e2fbc1cd9a266ad6ad745484be403f926',
 'training_protocol.json':'310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d',
 'flow/FNO.0.0.mdlus':'dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31',
 'flow/checkpoint.0.0.pt':'4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e',
 'aerodynamic/FNO.0.1.mdlus':'8a89f4774923afa698328e8e65ae337e7e0efefdae0bc9452d6b7a78758fb70d',
 'aerodynamic/checkpoint.0.1.pt':'d78d43d43738dd63b9556819e22f6b57c16993affaaff94dc3a29b6250994d6c'}
TENSORS={'flow':'89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb',
 'aerodynamic':'6f58aea89ecdde46bfaafac0f181d2603bbc96e3bba1faa7d8a818dcf6f7984d'}
KINDS=('original','h1_mean','h1_rms','ar_mean','ar_rms')


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def load_dependencies(args):
    if sha(args.diagnostic_script)!=P014_SHA: raise ValueError('P014 helper SHA differs')
    spec=importlib.util.spec_from_file_location('p019_pinned_p014',args.diagnostic_script)
    diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
    for name,digest in diagnostic.SOURCE_SHA.items():
        if sha(args.source_root/name)!=digest: raise ValueError('immutable source differs: '+name)
    if sha(args.config)!=diagnostic.CONFIG_SHA or sha(args.candidate_audit)!=AUDIT_SHA:
        raise ValueError('configuration/audit identity differs')
    for name,digest in FILES.items():
        if sha(args.candidate/name)!=digest: raise ValueError('P018 terminal file differs: '+name)
    sys.path[:0]=[str(args.source_root/'src'),str(args.source_root/'scripts')]
    return diagnostic,diagnostic.load_objective(args.source_root)


def statistic_cotangent(predicted,target,std,kind):
    """Physical tail62 squared error and exact normalized-output cotangent."""
    import torch
    p=predicted.detach().cpu().double();y=target.detach().cpu().double()
    if p.shape!=(100,) or y.shape!=p.shape or kind not in ('mean','rms'):
        raise ValueError('H100 rear-Cl vectors/statistic required')
    if not torch.isfinite(p).all() or not torch.isfinite(y).all() or not 0<float(std)<float('inf'):
        raise ValueError('nonfinite force/std')
    p,y=p[38:],y[38:]; s=float(std); out=torch.zeros(100,dtype=torch.float64)
    if kind=='mean':
        error=s*(p-y).mean();out[38:]=2*error*s/62
    else:
        centered=p-p.mean();pr=centered.square().mean().sqrt();tr=(y-y.mean()).square().mean().sqrt()
        error=s*(pr-tr)
        if float(pr)==0: return float(error.square()),None
        out[38:]=2*s*s*(pr-tr)*centered/(62*pr)
    return float(error.square()),out


def verify_unchanged(flow,aero,objective,frozen):
    if any(objective.tensor_state_sha256(m)!=TENSORS[r] for r,m in (('flow',flow),('aerodynamic',aero))):
        raise RuntimeError('model tensors changed')
    if any(p.grad is not None for m in (flow,aero) for p in m.parameters()):
        raise RuntimeError('diagnostic gradients were not cleared')
    if any(objective.tensor_sha256(p)!=frozen[n] for n,p in aero.named_parameters() if n in frozen):
        raise RuntimeError('frozen bias invariant failed')


def dot(a,b):
    import torch
    if a.keys()!=b.keys(): raise ValueError('gradient names differ')
    total=0.0
    for name in a:
        x,y=a[name],b[name]
        if x.shape!=y.shape: raise ValueError('gradient shapes differ')
        dtype=torch.complex128 if x.is_complex() or y.is_complex() else torch.float64
        total+=float((x.to(dtype).conj()*y.to(dtype)).real.sum())
    if not __import__('math').isfinite(total): raise FloatingPointError('nonfinite dot')
    return total


def alignment(original,statistic):
    a,b=dot(original,original)**.5,dot(statistic,statistic)**.5
    cross=dot(statistic,original)
    return dict(original_norm=a,statistic_norm=b,real_inner_product=cross,
        negative_original_directional_derivative=-cross,
        unit_negative_original_directional_derivative=-cross/a if a else None,
        cosine=cross/(a*b) if a and b else None,
        direction='negative raw original gradient; NOT AdamW displacement')


def repeat_difference(a,b):
    difference={n:b[n]-a[n] for n in a}
    absolute=dot(difference,difference)**.5;norm=dot(a,a)**.5
    return dict(l2_difference=absolute,relative_l2_difference=absolute/norm if norm else None,
                interpretation='two-repeat observed spread, not rigorous numerical bound')


def add_mean(total,gradient,count):
    import torch
    for name,value in gradient.items():
        dtype=torch.complex128 if value.is_complex() else torch.float64
        v=value.to(dtype)/count
        if name not in total: total[name]=v.clone()
        else: total[name].add_(v)


def gradient_pass(objective,model,states,h1,batch,predict,*,cotangent=None,domain=None):
    """Replay immutable mixed20 schedule; cotangent VJP never changes its loss."""
    import torch
    model.zero_grad(set_to_none=True);captures=[];cast_error=0.0
    if cotangent is not None and (domain not in ('h1','ar') or cotangent.shape!=(100,) or not torch.isfinite(cotangent).all()):
        raise ValueError('finite H100 domain cotangent required')
    def recording(network,inputs,masks):
        nonlocal cast_error
        _,force=predict(network,inputs,masks)
        if force.shape!=(20,4) or not torch.isfinite(force).all(): raise ValueError('mixed20 forces differ')
        chunk=len(captures);captures.append(force.detach().cpu().clone())
        if cotangent is not None:
            upstream=torch.zeros_like(force);offset=0 if domain=='h1' else 10
            upstream[offset:offset+10,3]=cotangent[chunk*10:(chunk+1)*10].to(force)
            cast_error=max(cast_error,float((upstream[offset:offset+10,3].detach().cpu().double()-cotangent[chunk*10:(chunk+1)*10]).abs().max()))
            force.backward(upstream)
        return _,force
    metrics=objective.chunk_force_objective(model,states,h1,batch['mask'],batch['omega'],batch['target_force'],
        recording,chunk_size=10,backward=cotangent is None)
    if len(captures)!=10: raise ValueError('ten chunks required')
    gradient={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError('missing/nonfinite gradient')
            gradient[name]=p.grad.detach().cpu().clone()
        elif p.grad is not None: raise RuntimeError('frozen parameter received gradient')
    model.zero_grad(set_to_none=True)
    outputs={d:torch.cat([x[s:s+10] for x in captures]) for d,s in (('h1',0),('ar',10))}
    return gradient,outputs,{**metrics,'cotangent_cast_max_absolute_difference':cast_error if cotangent is not None else None}


def execute(args,diagnostic,objective):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model,predict,configured_force_indices
    started=time.monotonic();resources=[]
    def guard():
        if time.monotonic()-started>900: raise RuntimeError('15-minute diagnostic deadline')
        resources.append(diagnostic.check_memory())
    guard();cfg=OmegaConf.load(args.config)
    trainer=objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    input_sha=trainer.validate_data_contract(cfg)
    DistributedManager.initialize();dist=DistributedManager()
    if dist.distributed or not dist.cuda: raise RuntimeError('single GPU required, separately approved')
    torch.cuda.set_per_process_memory_fraction(.15,dist.device);precision=validate_runtime_precision()
    python_rng,numpy_rng=random.getstate(),np.random.get_state()
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
    flow,aero=build_model(cfg).to(dist.device),build_model(cfg).to(dist.device)
    for role,model,epoch in (('flow',flow,0),('aerodynamic',aero,1)):
        metadata={}
        if load_checkpoint(args.candidate/role,models=model,metadata_dict=metadata,device=dist.device)!=epoch:
            raise ValueError('official checkpoint epoch differs')
        if objective.tensor_state_sha256(model)!=TENSORS[role]: raise ValueError('official loaded tensor identity differs')
    for p in flow.parameters(): p.requires_grad_(False)
    flow.eval();aero.train()
    if sum(p.requires_grad for p in aero.parameters())!=28: raise ValueError('28 force parameter tensors required')
    frozen={n:objective.tensor_sha256(p) for n,p in aero.named_parameters() if not p.requires_grad}
    if set(frozen)!=set(objective.OFFICIAL_FROZEN_PARAMETER_NAMES): raise ValueError('two frozen bias identities differ')
    if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in aero.modules()):
        raise ValueError('stochastic/running-state model forbidden')
    indices=configured_force_indices(cfg)
    if tuple(indices)!=(0,1,2,3): raise ValueError('force order differs')
    base=TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),num_workers=cfg.training.workers,force_indices=indices)
    train=None;rows=[];aggregate=[{k:{} for k in (*KINDS,'original_six')} for _ in range(2)]
    unavailable=set()
    try:
        train,_=compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],rollout_steps=100,
            stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=indices)
        items=trainer.diagnostic_windows(train);diagnostic.validate_windows(items)
        for index,item in enumerate(items):
            guard();batch={k:v[None].to(dist.device) for k,v in item['sample'].items()}
            states=objective.frozen_flow_states(flow,batch['state'],batch['mask'],batch['omega'],predict)
            h1=objective.true_state_inputs(batch['state'],batch['target_state'])
            originals=[];outputs=[];losses=[]
            for repeat in range(2):
                guard();g,out,loss=gradient_pass(objective,aero,states,h1,batch,predict)
                originals.append(g);outputs.append(out);losses.append(loss)
                add_mean(aggregate[repeat]['original_six'],g,6)
                if index: add_mean(aggregate[repeat]['original'],g,5)
                print(json.dumps(dict(event='gradient_complete',global_index=item['global_index'],kind='original',repeat=repeat)),flush=True)
            record={k:item[k] for k in ('global_index','family','identity')}
            record.update(objective_repeats=losses,original_gradient_repeat=repeat_difference(*originals),statistics={})
            record['original_output_repeat_max_absolute_difference']={d:float((outputs[1][d]-outputs[0][d]).abs().max()) for d in ('h1','ar')}
            for name in KINDS[1:]:
                domain,kind=name.split('_');gradients=[];values=[];comparisons=[];replay=[];cast_errors=[]
                for repeat in range(2):
                    value,cot=statistic_cotangent(outputs[repeat][domain][:,3],batch['target_force'][0,:,3],float(base.force_std[3]),kind)
                    values.append(value)
                    if cot is None:
                        if index: unavailable.add(name)
                        break
                    guard();g,out,vjp_metrics=gradient_pass(objective,aero,states,h1,batch,predict,cotangent=cot,domain=domain)
                    cast_errors.append(vjp_metrics['cotangent_cast_max_absolute_difference'])
                    gradients.append(g);comparisons.append(alignment(originals[repeat],g))
                    replay.append(max(float((out[d]-outputs[repeat][d]).abs().max()) for d in ('h1','ar')))
                    if index: add_mean(aggregate[repeat][name],g,5)
                    print(json.dumps(dict(event='gradient_complete',global_index=item['global_index'],kind=name,repeat=repeat)),flush=True)
                record['statistics'][name]=dict(physical_squared_error_repeats=values,
                    available=len(gradients)==2,alignment_repeats=comparisons,
                    gradient_repeat=repeat_difference(*gradients) if len(gradients)==2 else None,
                    replay_output_max_absolute_differences=replay,
                    cotangent_cast_max_absolute_differences=cast_errors,
                    cotangent='CPU float64 analytic derivative cast to force dtype for VJP')
            rows.append(record)
            del originals,outputs,states,h1,batch
        summary={}
        for name in KINDS[1:]:
            summary[name]=dict(available=name not in unavailable)
            if name not in unavailable:
                summary[name].update(five_nonzero_alignment_repeats=[alignment(a['original'],a[name]) for a in aggregate],
                    against_six_window_original_repeats=[alignment(a['original_six'],a[name]) for a in aggregate],
                    gradient_repeat=repeat_difference(aggregate[0][name],aggregate[1][name]),
                    five_nonzero_physical_squared_error_repeats=[sum(row['statistics'][name]['physical_squared_error_repeats'][repeat] for row in rows[1:])/5 for repeat in range(2)])
        return dict(status='FC_P019_GRADIENT_ALIGNMENT_DIAGNOSTIC_COMPLETE_NOT_ADMISSION',
            rows=rows,aggregate=summary,source_sha256=diagnostic.SOURCE_SHA,p014_sha256=P014_SHA,
            script_sha256=sha(__file__),input_sha256=input_sha,candidate_files_sha256=FILES,
            candidate_audit_sha256=AUDIT_SHA,config_sha256=sha(args.config),precision=precision,resource_checks=resources,
            six_window_original_objective_repeats=[{key:sum(row['objective_repeats'][repeat][key] for row in rows)/6 for key in ('h1_balanced','ar_balanced','total')} for repeat in range(2)],
            five_nonzero_original_objective_repeats=[{key:sum(row['objective_repeats'][repeat][key] for row in rows[1:])/5 for key in ('h1_balanced','ar_balanced','total')} for repeat in range(2)],
            original_five_gradient_repeat=repeat_difference(aggregate[0]['original'],aggregate[1]['original']),
            original_six_gradient_repeat=repeat_difference(aggregate[0]['original_six'],aggregate[1]['original_six']),
            model_tensor_sha256=TENSORS,models_unchanged=True,force_training_mode=True,flow_training_mode=False,
            optimizer_created=False,optimizer_steps=0,candidate_saved=False,validation_accessed=False,
            frozen_test_accessed=False,scientific_admission=False,ppo_executed=False,
            hdf_verification='separate approved launcher must verify exact44 train bytes against pinned audit',
            uncertainty='two-repeat spread only; no rigorous bound, sign gate or Adam direction claim')
    finally:
        aero.zero_grad(set_to_none=True)
        random.setstate(python_rng);np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
        (train if train is not None else base).close()
        verify_unchanged(flow,aero,objective,frozen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','diagnostic-script','config','candidate','candidate-audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    diagnostic,objective=load_dependencies(args)
    if not args.execute: print('FC_P019_PREFLIGHT_ONLY_NO_GPU');return
    if args.output.exists(): raise FileExistsError(args.output)
    result=execute(args,diagnostic,objective);diagnostic.write_exclusive(args.output,result)


if __name__=='__main__': main()
