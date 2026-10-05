#!/usr/bin/env python3
"""Fixed two-arm train-only finite update probe; no saved candidate/admission."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

P019_SHA='03e0ba375bf131bcc30bbe0d96c8b44bcc5c18ef1ed99de5d9e563e1b6525a64'
PROTOCOL=dict(experiment='FC-P020',arms=['A_original','B_symmetric_tail'],updates_per_arm=16,
    windows_per_update=6,statistic_coefficient=5/16,learning_rate=1.5625e-7,weight_decay=1e-4,
    betas=[.9,.999],eps=1e-8,clip_norm=1.,seed=20261003,tail_start=38,tail_length=62,
    endpoint_diagnostics=[0,16],endpoint_repeats=2,allocator_fraction=.15,time_limit_seconds=1800)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_dependencies(args):
    if sha(args.gradient_helper)!=P019_SHA:raise ValueError('immutable P019 helper differs')
    spec=importlib.util.spec_from_file_location('p020_pinned_p019',args.gradient_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    diagnostic,objective=helper.load_dependencies(args)
    return helper,diagnostic,objective


def capture(objective,model,states,h1,batch,predict):
    import torch
    chunks=[]
    def record(network,inputs,masks):
        delta,force=predict(network,inputs,masks)
        if force.shape!=(20,4) or not torch.isfinite(force).all():raise ValueError('finite mixed20 outputs required')
        chunks.append(force.detach().cpu().clone());return delta,force
    with torch.no_grad():
        metrics=objective.chunk_force_objective(model,states,h1,batch['mask'],batch['omega'],batch['target_force'],record,chunk_size=10,backward=False)
    if len(chunks)!=10:raise ValueError('ten chunks required')
    return {d:torch.cat([f[start:start+10] for f in chunks]) for d,start in (('h1',0),('ar',10))},metrics


def window_backward(helper,objective,model,states,h1,batch,predict,*,statistics):
    """Add normalized statistic VJP ONCE to original chunkwise loss gradient."""
    import torch
    captures=None;upstream={};stat_values={};cast_error=0.;replay_error=0.;chunk=0
    if statistics:
        captures,_=capture(objective,model,states,h1,batch,predict)
        for domain in ('h1','ar'):
            cot=torch.zeros(100,dtype=torch.float64)
            for kind in ('mean','rms'):
                value,part=helper.statistic_cotangent(captures[domain][:,3],batch['target_force'][0,:,3],1.,kind)
                if part is None:raise ValueError('zero predicted RMS has undefined statistic gradient')
                stat_values[domain+'_'+kind]=value;cot.add_(part)
            upstream[domain]=PROTOCOL['statistic_coefficient']*cot
    def with_statistic(network,inputs,masks):
        nonlocal chunk,cast_error,replay_error
        delta,force=predict(network,inputs,masks)
        if force.shape!=(20,4) or not torch.isfinite(force).all():raise ValueError('mixed20 outputs differ')
        if statistics:
            extra=torch.zeros_like(force)
            for domain,start in (('h1',0),('ar',10)):
                cot=upstream[domain][chunk*10:(chunk+1)*10]
                extra[start:start+10,3]=cot.to(force)
                cast_error=max(cast_error,float((extra[start:start+10,3].detach().cpu().double()-cot).abs().max()))
                replay_error=max(replay_error,float((force[start:start+10].detach().cpu()-captures[domain][chunk*10:(chunk+1)*10]).abs().max()))
            force.register_hook(lambda original,extra=extra:original+extra)
        chunk+=1
        return delta,force
    original=objective.chunk_force_objective(model,states,h1,batch['mask'],batch['omega'],batch['target_force'],with_statistic,chunk_size=10,backward=True)
    if chunk!=10:raise ValueError('ten backward chunks required')
    return dict(original=original,normalized_statistic_errors=stat_values,
        training_total=original['total']+PROTOCOL['statistic_coefficient']*sum(stat_values.values()),
        replay_output_max_absolute_difference=replay_error,cotangent_cast_max_absolute_difference=cast_error)


def update(model,optimizer,items,run):
    import torch
    if len(items)!=6:raise ValueError('six windows required')
    optimizer.zero_grad(set_to_none=True);records=[run(item) for item in items]
    parameters=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('invalid raw gradient')
    for p in parameters:p.grad.div_(6)
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):raise FloatingPointError('nonfinite updated parameter')
    if any(torch.is_tensor(v) and not torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values()):raise FloatingPointError('nonfinite optimizer state')
    return dict(window_records=records,preclip_mean_gradient_norm=float(norm),
        mean_training_total=sum(r['training_total'] for r in records)/6)


def metrics(helper,outputs,target,force_std,objective):
    import torch
    result={}
    for domain,prediction in outputs.items():
        p,y=prediction.double(),target.detach().cpu().double();std=force_std.detach().cpu().double()
        residual=(p-y)*std;tail=residual[38:,3]
        bias,rms_error=[helper.statistic_cotangent(p[:,3],y[:,3],float(std[3]),kind)[0] for kind in ('mean','rms')]
        result[domain]=dict(bias_mse=bias,rms_error_mse=rms_error,absolute_rms_error=rms_error**.5,
            centered_residual_mse=float(((tail-tail.mean())**2).mean()),signed_mean_error=float(tail.mean()),
            predicted_tail_rms=float(p[38:,3].std(correction=0)*std[3]),truth_tail_rms=float(y[38:,3].std(correction=0)*std[3]),
            four_force_physical_mae=residual.abs().mean(0).tolist(),four_force_physical_mse=residual.square().mean(0).tolist())
    return dict(objective=objective,domains=result)


def aggregate(rows):
    if [r['global_index'] for r in rows]!=[160,816,923,975,1077,1233]:raise ValueError('fixed window ordering differs')
    return {d:dict(six_window_original_objective=sum(r['panel']['objective'][d+'_balanced'] for r in rows)/6,
        five_nonzero={k:sum(r['panel']['domains'][d][k] for r in rows[1:])/5 for k in
            ('bias_mse','rms_error_mse','absolute_rms_error','centered_residual_mse')}) for d in ('h1','ar')}


def compare(initial,control,intervention):
    checks={}
    for domain in ('h1','ar'):
        x,a,b=initial[domain],control[domain],intervention[domain]
        checks[domain]=dict(original_objective_lower_than_initial=b['six_window_original_objective']<x['six_window_original_objective'],
            centered_residual_not_increased=b['five_nonzero']['centered_residual_mse']<=x['five_nonzero']['centered_residual_mse'],
            statistics={k:dict(lower_than_initial=b['five_nonzero'][k]<x['five_nonzero'][k],
                lower_than_control=b['five_nonzero'][k]<a['five_nonzero'][k],
                delta_initial=b['five_nonzero'][k]-x['five_nonzero'][k],delta_control=b['five_nonzero'][k]-a['five_nonzero'][k])
                for k in ('bias_mse','rms_error_mse')})
    support=all(v['original_objective_lower_than_initial'] and v['centered_residual_not_increased']
        and all(s['lower_than_initial'] and s['lower_than_control'] for s in v['statistics'].values()) for v in checks.values())
    return dict(checks=checks,strict_local_conditions=support,scientific_admission=False,
        interpretation='finite train-panel comparison only; near-zero numeric differences require review; no automatic full training')


def numeric_delta(first,second):
    """Signed repeat differences; observations, not rigorous error bounds."""
    if isinstance(first,dict):return {k:numeric_delta(v,second[k]) for k,v in first.items()}
    if isinstance(first,list):return [numeric_delta(v,second[i]) for i,v in enumerate(first)]
    return second-first


def repeated_panel(evaluate):
    first,second=evaluate(),evaluate()
    if [r['global_index'] for r in first['rows']]!=[r['global_index'] for r in second['rows']]:
        raise ValueError('endpoint repeat window identity differs')
    return {**first,'repeat':second,
        'repeat_aggregate_signed_delta':numeric_delta(first['aggregate'],second['aggregate']),
        'repeat_per_window_signed_delta':[dict(global_index=a['global_index'],panel=numeric_delta(a['panel'],b['panel']))
            for a,b in zip(first['rows'],second['rows'])],
        'repeat_uncertainty':'absolute paired difference is observed spread only, not a rigorous bound'}


def assess_repeat_resolution(a_initial,a_terminal,b_initial,b_terminal):
    def within_spread(a,b,ar,br):
        if isinstance(a,dict):return a.keys()==b.keys()==ar.keys()==br.keys() and all(within_spread(a[k],b[k],ar[k],br[k]) for k in a)
        if isinstance(a,list):return len(a)==len(b)==len(ar)==len(br) and all(within_spread(*x) for x in zip(a,b,ar,br))
        return abs(a-b)<=abs(ar-a)+abs(br-b)
    def value(panel,domain,key):
        row=panel['aggregate'][domain]
        return row['six_window_original_objective'] if key=='objective' else row['five_nonzero'][key]
    def spread(panel,domain,key):return abs(value(panel['repeat'],domain,key)-value(panel,domain,key))
    rows={};initial_agreement=True;resolved=True;window_agreement=[]
    for a,b,ar,br in zip(a_initial['rows'],b_initial['rows'],a_initial['repeat']['rows'],b_initial['repeat']['rows']):
        agrees=(a['global_index']==b['global_index']==ar['global_index']==br['global_index']
                and within_spread(a['panel'],b['panel'],ar['panel'],br['panel']))
        window_agreement.append(dict(global_index=a['global_index'],agrees_at_observed_repeat_resolution=agrees,
            signed_difference=numeric_delta(a['panel'],b['panel'])))
        initial_agreement=initial_agreement and agrees
    for domain in ('h1','ar'):
        initial_checks={}
        for key in ('objective','bias_mse','rms_error_mse','centered_residual_mse'):
            difference=abs(value(a_initial,domain,key)-value(b_initial,domain,key))
            noise=spread(a_initial,domain,key)+spread(b_initial,domain,key)
            agrees=difference<=noise
            initial_checks[key]=dict(absolute_difference=difference,combined_observed_repeat_spread=noise,
                                     agrees_at_observed_repeat_resolution=agrees)
            initial_agreement=initial_agreement and agrees
        comparisons={}
        pairs=[('objective','initial',b_initial),('centered_residual_mse','initial',b_initial)]
        pairs += [(key,label,reference) for key in ('bias_mse','rms_error_mse')
                  for label,reference in (('initial',b_initial),('control',a_terminal))]
        for key,label,reference in pairs:
            margin=value(reference,domain,key)-value(b_terminal,domain,key)
            noise=spread(reference,domain,key)+spread(b_terminal,domain,key)
            # Exact unchanged centered error is permitted by the preregistered
            # non-increase rule; strict improvements must exceed observed spread.
            distinguishes=abs(margin)>noise or (key=='centered_residual_mse' and margin==0 and noise==0)
            comparisons[key+'_vs_'+label]=dict(improvement_delta=margin,combined_observed_repeat_spread=noise,
                numerically_resolved_at_observed_repeats=distinguishes)
            resolved=resolved and distinguishes
        rows[domain]=dict(initial_agreement=initial_checks,comparisons=comparisons)
    return dict(domains=rows,per_window_initial_agreement=window_agreement,
        initials_agree_at_observed_repeat_resolution=initial_agreement,
        numerically_resolved=initial_agreement and resolved,
        caveat='two-repeat spread is not a rigorous numerical bound; no scientific tolerance or admission threshold')


def execute(args,helper,diagnostic,objective):
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
        if time.monotonic()-started>1800:raise RuntimeError('30-minute whole-probe deadline')
        resources.append(diagnostic.check_memory())
    guard();cfg=OmegaConf.load(args.config)
    trainer=objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    inputs=trainer.validate_data_contract(cfg)
    DistributedManager.initialize();dist=DistributedManager()
    if dist.distributed or not dist.cuda:raise RuntimeError('single approved CUDA device required')
    torch.cuda.set_per_process_memory_fraction(.15,dist.device);precision=validate_runtime_precision()
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    flow,aero=build_model(cfg).to(dist.device),build_model(cfg).to(dist.device)
    for role,model,epoch in (('flow',flow,0),('aerodynamic',aero,1)):
        if load_checkpoint(args.candidate/role,models=model,metadata_dict={},device=dist.device)!=epoch:raise ValueError('epoch differs')
        if objective.tensor_state_sha256(model)!=helper.TENSORS[role]:raise ValueError('P018 tensor identity differs')
    for p in flow.parameters():p.requires_grad_(False)
    flow.eval();aero.train()
    if sum(p.requires_grad for p in aero.parameters())!=28:raise ValueError('trainable scope differs')
    frozen={n:objective.tensor_sha256(p) for n,p in aero.named_parameters() if not p.requires_grad}
    if set(frozen)!=set(objective.OFFICIAL_FROZEN_PARAMETER_NAMES):raise ValueError('frozen bias names differ')
    if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in aero.modules()):raise ValueError('stateful/stochastic force mode forbidden')
    original={n:v.detach().cpu().clone() for n,v in aero.state_dict().items()}
    indices=configured_force_indices(cfg)
    if tuple(indices)!=(0,1,2,3):raise ValueError('force order differs')
    base=TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),num_workers=cfg.training.workers,force_indices=indices)
    train=None;arms=[]
    try:
        train,_=compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],rollout_steps=100,
            stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=indices)
        items=trainer.diagnostic_windows(train);diagnostic.validate_windows(items);cached=[]
        for item in items:
            guard();batch={k:v[None].to(dist.device) for k,v in item['sample'].items()}
            states=objective.frozen_flow_states(flow,batch['state'],batch['mask'],batch['omega'],predict)
            h1=objective.true_state_inputs(batch['state'],batch['target_state'])
            cached.append(dict(identity={k:item[k] for k in ('global_index','family','identity')},states=states.cpu(),h1=h1.cpu(),
                batch={k:batch[k].cpu() for k in ('mask','omega','target_force')}))
            del states,h1,batch
            print(json.dumps(dict(event='cached_window',global_index=item['global_index'])),flush=True)
        def transfer(item):
            guard();return item['states'].to(dist.device),item['h1'].to(dist.device),{k:v.to(dist.device) for k,v in item['batch'].items()}
        def single_panel(step):
            rows=[]
            for item in cached:
                states,h1,batch=transfer(item);outputs,loss=capture(objective,aero,states,h1,batch,predict)
                rows.append({**item['identity'],'panel':metrics(helper,outputs,batch['target_force'][0],base.force_std,loss)})
            return dict(update=step,rows=rows,aggregate=aggregate(rows))
        def panel(step):
            before=objective.tensor_state_sha256(aero)
            gradients={n:None if p.grad is None else objective.tensor_sha256(p.grad) for n,p in aero.named_parameters()}
            result=repeated_panel(lambda:single_panel(step))
            if objective.tensor_state_sha256(aero)!=before or gradients!={n:None if p.grad is None else objective.tensor_sha256(p.grad) for n,p in aero.named_parameters()}:
                raise RuntimeError('endpoint repeats mutated force model or gradients')
            return result
        for arm,statistics in (('A_original',False),('B_symmetric_tail',True)):
            aero.load_state_dict(original);aero.zero_grad(set_to_none=True);helper.verify_unchanged(flow,aero,objective,frozen)
            random.seed(PROTOCOL['seed']);np.random.seed(PROTOCOL['seed']);torch.manual_seed(PROTOCOL['seed']);torch.cuda.manual_seed_all(PROTOCOL['seed'])
            optimizer=torch.optim.AdamW([p for p in aero.parameters() if p.requires_grad],lr=PROTOCOL['learning_rate'],
                betas=tuple(PROTOCOL['betas']),eps=PROTOCOL['eps'],weight_decay=PROTOCOL['weight_decay'])
            initial=panel(0);records=[]
            def run(item):
                states,h1,batch=transfer(item)
                value=window_backward(helper,objective,aero,states,h1,batch,predict,statistics=statistics)
                print(json.dumps(dict(event='window_backward_complete',arm=arm,update=step,**item['identity'])),flush=True)
                return {**item['identity'],**value}
            for step in range(1,17):
                record=update(aero,optimizer,cached,run)
                if len(optimizer.state)!=28 or any(float(s['step'])!=step for s in optimizer.state.values()):raise ValueError('optimizer step/state count differs')
                if any(p.grad is not None or objective.tensor_sha256(p)!=frozen[n] for n,p in aero.named_parameters() if n in frozen):raise RuntimeError('frozen force bias changed')
                if objective.tensor_state_sha256(flow)!=helper.TENSORS['flow'] or any(p.grad is not None for p in flow.parameters()):raise RuntimeError('flow changed')
                records.append(dict(update=step,**record))
                print(json.dumps(dict(event='arm_update_complete',arm=arm,update=step)),flush=True)
            terminal=panel(16)
            arms.append(dict(arm=arm,initial=initial,terminal=terminal,records=records,optimizer_steps=16,
                terminal_in_memory_tensor_sha256=objective.tensor_state_sha256(aero)))
            del optimizer
        comparison=compare(arms[1]['initial']['aggregate'],arms[0]['terminal']['aggregate'],arms[1]['terminal']['aggregate'])
        repeat_assessment=assess_repeat_resolution(arms[0]['initial'],arms[0]['terminal'],arms[1]['initial'],arms[1]['terminal'])
        comparison.update(repeat_assessment=repeat_assessment,
            local_support=bool(comparison['strict_local_conditions'] and repeat_assessment['numerically_resolved']),
            conclusion=('INCONCLUSIVE_NUMERICAL_REPEATS' if not repeat_assessment['numerically_resolved'] else
                        'LOCAL_CONDITIONS_MET_NOT_ADMISSION' if comparison['strict_local_conditions'] else 'LOCAL_CONDITIONS_NOT_MET'))
        return dict(status='FC_P020_SYMMETRIC_TAIL_STAT_DIAGNOSTIC_COMPLETE_NOT_ADMISSION',arms=arms,
            comparison=comparison,
            protocol=PROTOCOL,protocol_sha256=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            script_sha256=sha(__file__),p019_helper_sha256=P019_SHA,p014_sha256=helper.P014_SHA,
            source_sha256=diagnostic.SOURCE_SHA,input_sha256=inputs,candidate_files_sha256=helper.FILES,candidate_audit_sha256=helper.AUDIT_SHA,
            precision=precision,resource_checks=resources,restored_model_tensor_sha256=helper.TENSORS,
            flow_training_mode=False,force_training_mode=True,config_sha256=sha(args.config),
            candidate_saved=False,validation_accessed=False,frozen_test_accessed=False,scientific_admission=False,ppo_executed=False,
            optimizer_steps_total=32,hdf_verification='external approved launcher must verify44trainbytes against pinned audit')
    finally:
        aero.load_state_dict(original);aero.zero_grad(set_to_none=True)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        (train if train is not None else base).close();helper.verify_unchanged(flow,aero,objective,frozen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','diagnostic-script','gradient-helper','config','candidate','candidate-audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    helper,diagnostic,objective=load_dependencies(args)
    if not args.execute:print('FC_P020_PREFLIGHT_ONLY_NO_GPU');return
    if args.output.exists():raise FileExistsError(args.output)
    diagnostic.write_exclusive(args.output,execute(args,helper,diagnostic,objective))


if __name__=='__main__':main()
