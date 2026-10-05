#!/usr/bin/env python3
"""FC-P017: one AdamW direction, fixed read-only trials, no saved candidate."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random

P016_SHA = '18b210077a93bbd21a327ae6578a73fb371bf0d2f48d0541f6ca8bd87aa42bbb'
FACTORS = (1.0, 1.0/64, -1.0/64)


def load_helpers(args):
    import hashlib
    if hashlib.sha256(args.panel_helper.read_bytes()).hexdigest() != P016_SHA:
        raise ValueError('immutable P016 helper differs')
    spec = importlib.util.spec_from_file_location('p017_pinned_p016',args.panel_helper)
    panel = importlib.util.module_from_spec(spec);spec.loader.exec_module(panel)
    diagnostic,objective = panel.load_dependencies(args.source_root.resolve(),args.diagnostic_script.resolve())
    return panel,diagnostic,objective


def real64(tensor):
    import torch
    value = tensor.detach().cpu()
    return (torch.view_as_real(value) if value.is_complex() else value).double().clone()


def restore(model, initial):
    model.load_state_dict(initial,strict=True)


def apply_displacement(model, initial, delta, factor):
    import torch
    restore(model,initial)
    norms, zeros = {}, {}
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if name not in delta: continue
            values = real64(initial[name]) + factor*delta[name]
            value = torch.view_as_complex(values.contiguous()) if parameter.is_complex() else values
            parameter.copy_(value.to(device=parameter.device,dtype=parameter.dtype))
            actual = real64(parameter)-real64(initial[name])
            norms[name] = float(actual.square().sum())
            zeros[name] = int(((factor*delta[name] != 0)&(actual == 0)).sum())
    return dict(actual_rounded_displacement_norm=sum(norms.values())**0.5,
                disappeared_real_components=sum(zeros.values()),per_parameter_squared_norm=norms)


def first_step(model, optimizer, initial, run_window, cached):
    import torch
    if len(cached) != 6: raise ValueError('exactly six windows required')
    if optimizer.state: raise ValueError('fresh AdamW required')
    if len(optimizer.param_groups) != 1: raise ValueError('one optimizer parameter group required')
    group = optimizer.param_groups[0]
    if (group['lr'],group['betas'],group['eps'],group['weight_decay']) != (1e-5,(.9,.999),1e-8,1e-4):
        raise ValueError('fixed AdamW configuration differs')
    optimizer.zero_grad(set_to_none=True)
    records = [run_window(item) for item in cached]
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for _,p in parameters):
        raise FloatingPointError('missing/nonfinite gradient')
    for _,p in parameters: p.grad.div_(6)
    raw = {n:real64(p.grad) for n,p in parameters}
    norm = torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.0,error_if_nonfinite=True)
    clipped = {n:real64(p.grad) for n,p in parameters}
    optimizer.step()
    if any(not torch.isfinite(p).all() for _,p in parameters): raise FloatingPointError('nonfinite AdamW output')
    if any(torch.is_tensor(v) and not torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values()):
        raise FloatingPointError('nonfinite AdamW state')
    delta,rows = {},{}
    for name,parameter in parameters:
        before,after,g,gc = real64(initial[name]),real64(parameter),raw[name],clipped[name]
        actual = after-before
        formula_gc = g*min(1.0,1.0/(float(norm)+1e-6))
        adaptive = -1e-5*formula_gc/(formula_gc.abs()+1e-8)
        decay = -1e-5*1e-4*before
        rounded_decay = real64(initial[name]*(1-1e-5*1e-4))-before
        ideal = adaptive+decay
        if not torch.isfinite(ideal).all(): raise FloatingPointError('nonfinite independent formula')
        delta[name] = actual
        rows[name] = dict(complex_parameter=parameter.is_complex(),dtype=str(parameter.dtype),real_components=actual.numel(),
            raw_gradient_squared_norm=float(g.square().sum()),delta_squared_norm=float(actual.square().sum()),
            directional_dot=float((g*actual).sum()),ideal_directional_dot=float((g*ideal).sum()),
            decay_directional_dot=float((g*decay).sum()),decay_squared_norm=float(decay.square().sum()),
            rounded_decay_directional_dot=float((g*rounded_decay).sum()),rounded_decay_squared_norm=float(rounded_decay.square().sum()),
            clip_formula_difference_max_abs=float((gc-formula_gc).abs().max()),
            formula_difference_squared_norm=float((actual-ideal).square().sum()),
            formula_difference_max_abs=float((actual-ideal).abs().max()),
            actual_zero_real_components=int((actual==0).sum()),
            nonzero_formula_disappeared_real_components=int(((ideal!=0)&(actual==0)).sum()))
    return delta,dict(window_records=records,optimizer_steps=1,raw_mean_gradient_norm_float64=sum(x['raw_gradient_squared_norm'] for x in rows.values())**.5,
        clip_returned_norm=float(norm),clip_factor=min(1.0,1.0/(float(norm)+1e-6)),
        actual_displacement_norm=sum(x['delta_squared_norm'] for x in rows.values())**.5,
        directional_dot=sum(x['directional_dot'] for x in rows.values()),
        decay_directional_dot=sum(x['decay_directional_dot'] for x in rows.values()),
        decay_displacement_norm=sum(x['decay_squared_norm'] for x in rows.values())**.5,
        formula_difference_norm=sum(x['formula_difference_squared_norm'] for x in rows.values())**.5,
        per_parameter=rows,formula_comparison='observational float64 real-component first-step formula; not a PASS tolerance',
        rounded_decay_method='isolated CPU original-dtype multiply; observational, not an isolated CUDA-kernel measurement',
        complex_semantics='real-view componentwise AdamW; dot equals Re(sum(conj(g)*delta))')


def summarize(evaluations, step):
    result = {}
    for domain in ('h1_balanced','ar_balanced','total'):
        values = {k:v['macro'][domain] for k,v in evaluations.items()}
        spread = max(values[k] for k in ('base1','base2','restored'))-min(values[k] for k in ('base1','base2','restored'))
        result[domain] = dict(values=values,observed_repeat_spread=spread,
            full_delta=values['full']-values['base1'],plus_delta=values['plus']-values['base1'],
            central_directional_difference=(values['plus']-values['minus'])/(2/64),
            small_decrease_exceeds_observed_repeat_spread=values['base1']-values['plus']>spread,
            full_increase_exceeds_observed_repeat_spread=values['full']-values['base1']>spread,
            small_step_sign_interpretation='inconclusive at observed repeat resolution' if abs(values['plus']-values['base1'])<=spread else 'sign exceeds observed repeat spread; TF32 accuracy still unbounded')
    return dict(domains=result,total_gradient_directional_dot=step['directional_dot'],
        numerical_caveat='Repeat spread is not a rigorous TF32/rounding bound; zero spread does not establish numerical accuracy.',
        scientific_admission=False,automatic_scientific_pass=False)


def execute(args,panel,diagnostic,objective):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model,configured_force_indices,predict
    trainer = objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    cfg = OmegaConf.load(args.config)
    input_sha = trainer.validate_data_contract(cfg)
    resources = [diagnostic.check_memory()]
    DistributedManager.initialize();dist=DistributedManager()
    if dist.distributed or not dist.cuda: raise RuntimeError('single CUDA device required')
    torch.cuda.set_per_process_memory_fraction(.15,dist.device)
    precision=validate_runtime_precision()
    random.seed(20261003);np.random.seed(20261003);torch.manual_seed(20261003);torch.cuda.manual_seed_all(20261003)
    indices=configured_force_indices(cfg)
    if tuple(indices)!=(0,1,2,3): raise ValueError('force channels differ')
    flow,aero=build_model(cfg).to(dist.device),build_model(cfg).to(dist.device)
    for model in (flow,aero):
        epoch=load_checkpoint(args.parent,models=model,metadata_dict={},device=dist.device)
        validate_calibrated_epoch_zero(args.parent,epoch,allow=True,expected_model_sha256=objective.PARENT_MODEL_SHA,
            expected_state_sha256=objective.PARENT_STATE_SHA,expected_kind=objective.PARENT_KIND)
    for p in flow.parameters(): p.requires_grad_(False)
    flow.eval();aero.train()
    if sum(p.requires_grad for p in aero.parameters())!=28: raise ValueError('force scope differs')
    frozen={n:objective.tensor_sha256(p) for n,p in aero.named_parameters() if not p.requires_grad}
    if set(frozen)!=set(objective.OFFICIAL_FROZEN_PARAMETER_NAMES): raise ValueError('frozen bias scope differs')
    flow_before=objective.tensor_state_sha256(flow);before=objective.tensor_state_sha256(aero)
    if flow_before!=before: raise ValueError('initial models differ')
    initial={n:v.detach().cpu().clone() for n,v in aero.state_dict().items()}
    base=TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),num_workers=cfg.training.workers,force_indices=indices)
    train=None
    try:
        train,_=compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],rollout_steps=100,
            stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=indices)
        items=trainer.diagnostic_windows(train);diagnostic.validate_windows(items)
        cached=[]
        for item in items:
            resources.append(diagnostic.check_memory())
            batch={k:v[None].to(dist.device) for k,v in item['sample'].items()}
            states=objective.frozen_flow_states(flow,batch['state'],batch['mask'],batch['omega'],predict)
            h1=objective.true_state_inputs(batch['state'],batch['target_state'])
            if any(not torch.isfinite(v).all() for v in [states,h1,*batch.values()]): raise FloatingPointError('nonfinite cache')
            cached.append(dict(identity={k:item[k] for k in ('global_index','family','identity')},states=states.detach().cpu(),
                h1=h1.detach().cpu(),batch={k:v.detach().cpu() for k,v in batch.items()}))
            del states,h1,batch
            print(json.dumps({'event':'cached_window','global_index':item['global_index']}),flush=True)
        def run(item,backward=False):
            resources.append(diagnostic.check_memory())
            batch={k:v.to(dist.device) for k,v in item['batch'].items()}
            metrics=objective.chunk_force_objective(aero,item['states'].to(dist.device),item['h1'].to(dist.device),
                batch['mask'],batch['omega'],batch['target_force'],predict,chunk_size=10,backward=backward)
            print(json.dumps({'event':'window','backward':backward,**item['identity'],'objective':metrics},allow_nan=False),flush=True)
            return {**item['identity'],'objective':metrics}
        def evaluate(label):
            with panel.readonly_panel(aero,objective,dist.device): rows=[run(x) for x in cached]
            return dict(label=label,rows=rows,macro={k:sum(r['objective'][k] for r in rows)/6 for k in ('h1_balanced','ar_balanced','total')},
                        tensor_sha256=objective.tensor_state_sha256(aero))
        evaluations={'base1':evaluate('base1'),'base2':evaluate('base2')}
        optimizer=torch.optim.AdamW(aero.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
        delta,step=first_step(aero,optimizer,initial,lambda x:run(x,True),cached)
        after_step=objective.tensor_state_sha256(aero)
        after_tensors={n:v.detach().cpu().clone() for n,v in aero.state_dict().items()}
        for name,p in aero.named_parameters():
            if name in frozen and (p.grad is not None or objective.tensor_sha256(p)!=frozen[name]): raise RuntimeError('frozen bias changed')
        trials={}
        for label,factor in zip(('full','plus','minus'),FACTORS):
            trials[label]=apply_displacement(aero,initial,delta,factor)
            evaluations[label]=evaluate(label)
            if label=='full':
                trials[label]['post_step_reconstruction_max_abs']=max(float((real64(v)-real64(after_tensors[n])).abs().max()) for n,v in aero.state_dict().items())
                trials[label]['matches_actual_post_step']=evaluations[label]['tensor_sha256']==after_step
        if not trials['full']['matches_actual_post_step']:
            restore(aero,after_tensors)
            evaluations['actual_post_step']=evaluate('actual_post_step')
        restore(aero,initial)
        if objective.tensor_state_sha256(aero)!=before: raise RuntimeError('initial tensors not restored exactly')
        evaluations['restored']=evaluate('restored')
        if objective.tensor_state_sha256(flow)!=flow_before or any(p.grad is not None for p in flow.parameters()): raise RuntimeError('flow changed')
        return dict(status='FC_P017_FIRST_STEP_DIAGNOSTIC_COMPLETE_NOT_ADMISSION',optimizer_steps=1,step=step,
            evaluations=evaluations,trials=trials,comparison=summarize(evaluations,step),precision=precision,
            tensor_sha256_initial=before,tensor_sha256_after_step=after_step,tensor_sha256_restored=objective.tensor_state_sha256(aero),
            flow_tensor_sha256_before=flow_before,flow_tensor_sha256_after=objective.tensor_state_sha256(flow),
            frozen_parameter_sha256=frozen,source_sha256=diagnostic.SOURCE_SHA,p016_helper_sha256=P016_SHA,p014_helper_sha256=panel.P014_SHA,
            config_sha256=panel.sha(args.config),probe_sha256=panel.sha(Path(__file__)),input_sha256=input_sha,
            parent_model_sha256=objective.PARENT_MODEL_SHA,parent_state_sha256=objective.PARENT_STATE_SHA,resource_checks=resources,
            candidate_saved=False,selection_performed=False,validation_accessed=False,frozen_test_accessed=False,ppo_executed=False,
            hdf_bytes_verification='external launch evidence required; manifests and normalization bound here')
    finally:
        restore(aero,initial)
        (train if train is not None else base).close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','panel-helper','diagnostic-script','config','parent','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    panel,diagnostic,objective=load_helpers(args)
    if panel.sha(args.config)!=objective.CONFIG_SHA: raise ValueError('config differs')
    objective.checkpoint_pair(args.parent)
    if args.output.exists(): raise FileExistsError(args.output)
    if not args.execute: print('FC_P017_IDENTITIES_VERIFIED_NO_GPU_NO_OPTIMIZATION');return
    diagnostic.write_exclusive(args.output,execute(args,panel,diagnostic,objective))


if __name__=='__main__': main()
