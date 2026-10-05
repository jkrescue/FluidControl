"""Fixed LOW/HIGH96-block study; project adapter, no saved candidate."""
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
PROTOCOL=dict(experiment='FC-P023',arms=['LOW','HIGH'],updates_per_arm=16,windows_per_update=6,
    learning_rates={'LOW':1.5625e-7,'HIGH':1e-5},trainable_scalars=96,trainable_tensors=1,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,gradient_clip_norm=1.,seed=20261003,
    objective='original J0, half H1/half AR, half equal-four/half rearCl',horizon=100,checkpoint_block=10,
    endpoint_updates=[0,16],endpoint_repeats=2,terminal_zero_input_ablation_repeats=2,allocator_fraction=.06,time_limit_seconds=1800,
    candidate_saved=False,heldout_accessed=False)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def protocol_sha():return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load(path,digest,name):
    if sha(path)!=digest:raise ValueError(name+' SHA differs')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def limits(memory,elapsed,startup=False):
    if elapsed>1800:raise RuntimeError('whole-comparison30minute deadline')
    if memory.get('MemFree',0)<(30 if startup else 20) or memory.get('MemAvailable',0)<(50 if startup else 20):raise RuntimeError('physical memory floor')

def check_optimizer(adapter,optimizer,step,rate,block_module):
    import torch
    block_module.assert_optimizer_scope(adapter,optimizer,step=step)
    for group in optimizer.param_groups:
        if group['lr']!=rate or group['weight_decay']!=PROTOCOL['weight_decay'] or tuple(group['betas'])!=tuple(PROTOCOL['betas']) or group['eps']!=PROTOCOL['eps']:raise ValueError('optimizer protocol differs')
    if any(torch.is_tensor(v) and not torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values()):raise FloatingPointError('nonfinite optimizer state')

def comparison(p020,initial,a,b):
    result=p020.compare(initial['aggregate'],a['aggregate'],b['aggregate'])
    # Stronger predeclared objective condition: B must also beat matched A.
    for d in ('h1','ar'):
        result['checks'][d]['original_objective_lower_than_control']=b['aggregate'][d]['six_window_original_objective']<a['aggregate'][d]['six_window_original_objective']
    result['strict_local_conditions']=result['strict_local_conditions'] and all(result['checks'][d]['original_objective_lower_than_control'] for d in ('h1','ar'))
    return result

def accumulation_step(model,optimizer,items,run,snapshot):
    import torch
    if len(items)!=6:raise ValueError('exactly six windows required')
    optimizer.zero_grad(set_to_none=True);records=[run(item) for item in items]
    parameters=[p for p in model.parameters() if p.requires_grad]
    if len(parameters)!=1 or parameters[0].numel()!=96 or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('one96 finite raw gradient required')
    for p in parameters:p.grad.div_(6)
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    before=snapshot();optimizer.step();after=snapshot()
    if any(not torch.isfinite(p).all() for p in parameters):raise FloatingPointError('updated parameter nonfinite')
    return dict(window_records=records,preclip_mean_gradient_norm=float(norm),mean_training_total=sum(r['training_total'] for r in records)/6,
        before_optimizer_step_memory=before,after_optimizer_step_memory=after,
        actual_adam_moment_bytes=sum(v.numel()*v.element_size() for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')))

def repeat_resolution(p020,ai,at,bi,bt):
    result=p020.assess_repeat_resolution(ai,at,bi,bt)
    for domain in ('h1','ar'):
        def value(panel):return panel['aggregate'][domain]['six_window_original_objective']
        margin=value(at)-value(bt)
        spread=abs(value(at['repeat'])-value(at))+abs(value(bt['repeat'])-value(bt))
        resolved=abs(margin)>spread
        result['domains'][domain]['comparisons']['objective_vs_control']=dict(improvement_delta=margin,combined_observed_repeat_spread=spread,numerically_resolved_at_observed_repeats=resolved)
        result['numerically_resolved']=result['numerically_resolved'] and resolved
    return result

def optimizer_identity(optimizer,objective):
    import torch
    return dict(groups=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups],
        parameter_shapes=[[list(p.shape) for p in group['params']] for group in optimizer.param_groups],
        states=[{k:objective.tensor_sha256(v) if torch.is_tensor(v) else v for k,v in sorted(state.items())} for state in optimizer.state.values()])

def ablation_difference(conditioned,ablated,std,p020):
    import torch
    rows=[]
    for c,a,cr,ar in zip(conditioned['rows'],ablated['rows'],conditioned['repeat']['rows'],ablated['repeat']['rows']):
        if len({x['global_index'] for x in (c,a,cr,ar)})!=1:raise ValueError('ablation window identities differ')
        domains={}
        for domain in ('h1','ar'):
            cp,ap,crp,arp=[torch.tensor(x['normalized_predictions'][domain],dtype=torch.float64) for x in (c,a,cr,ar)]
            difference=(cp-ap)*std.double();spread=(cp-crp).abs()*std.double()+(ap-arp).abs()*std.double()
            domains[domain]=dict(physical_output_difference_l2_per_force=difference.square().sum(0).sqrt().tolist(),
                physical_output_difference_max_abs_per_force=difference.abs().amax(0).tolist(),
                observed_combined_repeat_max_abs_per_force=spread.amax(0).tolist())
        rows.append(dict(global_index=c['global_index'],domains=domains,conditioned_minus_ablated_metrics=p020.numeric_delta(a['panel'],c['panel'])))
    return dict(rows=rows,conditioned_minus_ablated_aggregate=p020.numeric_delta(ablated['aggregate'],conditioned['aggregate']),
        interpretation='same terminal weights; zero input is diagnostic ablation, not a training control or admission criterion; observed repeat spread is not a rigorous bound')

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
    resources=[];phase='preflight';calls={'flow':0,'training_forward':0,'training_recompute':0,'panel':0,'ablation':0}
    def guard():
        memory={line.split(':')[0]:int(line.split()[1])/2**20 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemFree:','MemAvailable:'))}
        elapsed=time.monotonic()-started;resources.append(dict(elapsed_seconds=elapsed,phase=phase,**memory));limits(memory,elapsed)
        return memory
    cfg=OmegaConf.load(args.config);trainer=objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    data_identity=trainer.validate_data_contract(cfg)
    causal_items,mean,std=causal.load_causal_inputs(args.causal_audit,Path(cfg.data.root)/'normalization.json',args.raw_source_view)
    limits(guard(),time.monotonic()-started,startup=True)
    DistributedManager.initialize();dist=DistributedManager()
    if dist.distributed or not dist.cuda or torch.cuda.device_count()!=1:raise RuntimeError('one approved GPU required')
    torch.cuda.set_per_process_memory_fraction(.06,dist.device);precision=validate_runtime_precision()
    flow,legacy=build_model(cfg).to(dist.device),build_model(cfg).to(dist.device)
    for role,model,epoch in [('flow',flow,0),('aerodynamic',legacy,1)]:
        if load_checkpoint(args.candidate/role,models=model,metadata_dict={},device=dist.device)!=epoch or objective.tensor_state_sha256(model)!=helper.TENSORS[role]:raise ValueError('P018 parent differs')
    legacy.cpu();flow.eval()
    for p in flow.parameters():p.requires_grad_(False)
    if tuple(configured_force_indices(cfg))!=(0,1,2,3):raise ValueError('force order differs')
    base=TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),num_workers=cfg.training.workers,force_indices=(0,1,2,3))
    train=None;model=None;optimizer=None;arms=[];initial_identity=None
    try:
        train,_=compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],rollout_steps=100,stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=(0,1,2,3))
        cached=[];phase='cache_six_cpu_histories'
        for entry in causal_items:
            guard();sample,metadata=train[entry['global_index']];identity=trainer.training_identity([metadata])
            if identity!=entry['identity']:raise ValueError('fixed six identity differs')
            batch={k:v[None].to(dist.device) for k,v in sample.items()}
            def flow_predict(*a):
                guard();calls['flow']+=1;return predict(*a)
            states=objective.frozen_flow_states(flow,batch['state'],batch['mask'],batch['omega'],flow_predict)
            h1=objective.true_state_inputs(batch['state'],batch['target_state']);mask=batch['mask'];omega=batch['omega'][0]
            cached.append(dict(global_index=entry['global_index'],identity=identity,
                h1=objective.make_inputs(h1[0],mask.expand(100,-1,-1,-1),omega[:-1],omega[1:]).detach().cpu(),
                ar=objective.make_inputs(states[0],mask.expand(100,-1,-1,-1),omega[:-1],omega[1:]).detach().cpu(),
                mask=mask.cpu(),current=entry['normalized'],target=batch['target_force'][0].cpu(),hdf_sha256=entry['hdf_sha256'],
                persistence=causal.persistence(entry['physical'],batch['target_force'][0].detach().cpu()*std+mean)))
            del sample,batch,states,h1,mask,omega
            print(json.dumps(dict(event='cached_window_cpu',global_index=entry['global_index'])),flush=True)
        flow.cpu();torch.cuda.empty_cache()
        expanded_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));expanded_cfg.model.in_channels=10
        expanded=build_model(expanded_cfg);causal.warmstart(legacy,expanded,inspect.getfile(type(expanded)))
        official_sources=block_module.verify_official_sources(expanded)
        expanded.to(dist.device);expanded.train()
        model=block_module.FrozenForceBlock(expanded);model.train()
        initial_identity=model.diagnostic_identity()
        trainable=[p for p in model.parameters() if p.requires_grad]
        if len(trainable)!=1 or trainable[0] is not model.block:raise ValueError('only independent96leaf may train')
        def scope():
            model.verify_frozen()
            if any(p.grad is not None for parent in (flow,legacy) for p in parent.parameters()):raise RuntimeError('parent gradients changed')
        def run_window(item,zero,backward,ablation=False):
            guard();data=[item[k].to(dist.device) for k in ('h1','ar','mask','current','target')];recompute=False
            def counted(*a):
                guard();calls['training_recompute' if recompute else 'training_forward' if backward else 'ablation' if ablation else 'panel']+=1
                return predict(*a)
            if backward:
                out=causal.recurrent_objective(model,*data,counted,objective.balanced_force_objective,zero_conditioning=zero)
                recompute=True;out['total'].backward()
                loss={key:float(out[field]['balanced'].detach()) for key,field in [('h1_balanced','h1_loss'),('ar_balanced','ar_loss')]};loss['total']=float(out['total'].detach())
                result=dict(global_index=item['global_index'],original=loss,training_total=loss['total'])
            else:
                with torch.no_grad():out=causal.recurrent_objective(model,*data,counted,objective.balanced_force_objective,zero_conditioning=zero,checkpointed=False)
                loss=dict(h1_balanced=float(out['h1_loss']['balanced']),ar_balanced=float(out['ar_loss']['balanced']),total=float(out['total']),
                    h1_channel_mse=out['h1_loss']['channel_mse'].cpu().tolist(),ar_channel_mse=out['ar_loss']['channel_mse'].cpu().tolist())
                panel=p020.metrics(helper,{d:out[d].detach().cpu() for d in ('h1','ar')},data[-1],std,loss)
                result=dict(global_index=item['global_index'],identity=item['identity'],panel=panel,
                    normalized_predictions={d:out[d].detach().cpu().tolist() for d in ('h1','ar')})
            del out,data;guard();return result
        for arm,rate in PROTOCOL['learning_rates'].items():
            phase=arm;model.restore_zero()
            if model.diagnostic_identity()!=initial_identity:raise RuntimeError('arm initial restoration differs')
            random.seed(PROTOCOL['seed']);np.random.seed(PROTOCOL['seed']);torch.manual_seed(PROTOCOL['seed']);torch.cuda.manual_seed_all(PROTOCOL['seed'])
            optimizer=torch.optim.AdamW([model.block],lr=rate,weight_decay=PROTOCOL['weight_decay'],betas=tuple(PROTOCOL['betas']),eps=PROTOCOL['eps'])
            block_module.assert_optimizer_scope(model,optimizer)
            def single_panel(ablation=False):
                before=model.diagnostic_identity();gradients={n:None if p.grad is None else objective.tensor_sha256(p.grad) for n,p in model.named_parameters()}
                optimizer_before=optimizer_identity(optimizer,objective)
                mode=model.training;rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all();rng_python=random.getstate();rng_numpy=np.random.get_state()
                rows=[run_window(item,ablation,False,ablation=ablation) for item in cached]
                if model.diagnostic_identity()!=before or gradients!={n:None if p.grad is None else objective.tensor_sha256(p.grad) for n,p in model.named_parameters()} or optimizer_identity(optimizer,objective)!=optimizer_before:raise RuntimeError('panel/ablation mutated model/gradients/optimizer')
                if model.training!=mode or not torch.equal(rng_cpu,torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise RuntimeError('panel changed mode/RNG')
                after_numpy=np.random.get_state()
                if random.getstate()!=rng_python or rng_numpy[0]!=after_numpy[0] or not np.array_equal(rng_numpy[1],after_numpy[1]) or rng_numpy[2:]!=after_numpy[2:]:raise RuntimeError('panel changed Python/NumPy RNG')
                return dict(rows=rows,aggregate=p020.aggregate(rows))
            initial=p020.repeated_panel(single_panel);records=[];torch.cuda.reset_peak_memory_stats()
            for step in range(1,17):
                def train_window(item):
                    row=run_window(item,False,True)
                    print(json.dumps(dict(event='window_backward_complete',arm=arm,update=step,global_index=item['global_index'])),flush=True);return row
                def memory_snapshot():
                    torch.cuda.synchronize();return dict(**guard(),cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved())
                before_block=model.block.detach().cpu().clone()
                record=accumulation_step(model,optimizer,cached,train_window,memory_snapshot);check_optimizer(model,optimizer,step,rate,block_module);scope()
                record.update(update=step,actual_learning_rate=rate,postclip_block_gradient_norm=float(model.block.grad.double().norm()),
                    block_update=block_module.block_update_metrics(before_block,model.block),effective_identity=model.diagnostic_identity(),optimizer_identity=optimizer_identity(optimizer,objective))
                records.append(record);guard();print(json.dumps(dict(event='arm_update_complete',arm=arm,update=step)),flush=True)
            terminal=p020.repeated_panel(single_panel)
            ablation=p020.repeated_panel(lambda:single_panel(ablation=True))
            arms.append(dict(arm=arm,learning_rate=rate,initial=initial,terminal=terminal,terminal_zero_input_ablation=ablation,
                ablation_sensitivity=ablation_difference(terminal,ablation,std,p020),ablation_minus_initial_aggregate=p020.numeric_delta(initial['aggregate'],ablation['aggregate']),records=records,optimizer_steps=16,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),terminal_effective_identity=model.diagnostic_identity()))
            optimizer=None;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        interpretation=comparison(p020,arms[1]['initial'],arms[0]['terminal'],arms[1]['terminal'])
        repeats=repeat_resolution(p020,arms[0]['initial'],arms[0]['terminal'],arms[1]['initial'],arms[1]['terminal'])
        interpretation.update(repeat_assessment=repeats,local_support=bool(interpretation['strict_local_conditions'] and repeats['numerically_resolved']),scientific_admission=False,control_arm='LOW',experimental_arm='HIGH',both_arms_receive_causal_inputs=True)
        interpretation['per_window_terminal_deltas']=[dict(global_index=bi['global_index'],versus_initial=p020.numeric_delta(bi['panel'],bt['panel']),versus_control=p020.numeric_delta(at['panel'],bt['panel'])) for bi,at,bt in zip(arms[1]['initial']['rows'],arms[0]['terminal']['rows'],arms[1]['terminal']['rows'])]
        if calls!=dict(flow=600,training_forward=19200,training_recompute=19200,panel=4800,ablation=2400):raise RuntimeError('actual forward/backward/panel counts differ')
        return dict(status='FC_P023_INPUT_BLOCK_COMPARISON_COMPLETE_NOT_ADMISSION',protocol=PROTOCOL,protocol_sha256=protocol_sha(),arms=arms,comparison=interpretation,
            precision=precision,resources=resources,calls=calls,optimizer_steps_total=32,training_window_backwards=192,endpoint_window_forwards=48,ablation_window_forwards=24,
            expanded_initial_identity=initial_identity,official_sources=official_sources,parent_tensors=helper.TENSORS,data_identity=data_identity,hdf_identities=[dict(global_index=x['global_index'],sha256=x['hdf_sha256']) for x in cached],
            strict_causal_persistence=[dict(global_index=x['global_index'],metrics=x['persistence']) for x in cached],persistence_target_note='unchanged normalized training targets de-normalized in float32; no bitwise claim against original physical HDF receipt',
            candidate_saved=False,validation_accessed=False,frozen_test_accessed=False,scientific_admission=False,memory_strategy='six CPU histories, one window GPU, original parents CPU')
    finally:
        optimizer=None
        if model is not None and initial_identity is not None:
            model.restore_zero()
            if model.diagnostic_identity()!=initial_identity:raise RuntimeError('final expanded restore failed')
        for role,parent in [('flow',flow),('aerodynamic',legacy)]:
            if objective.tensor_state_sha256(parent)!=helper.TENSORS[role] or any(p.grad is not None for p in parent.parameters()):raise RuntimeError('original parent changed')
        (train if train is not None else base).close()

def main():
    parser=argparse.ArgumentParser()
    for key in ('source-root','diagnostic-script','gradient-helper','comparison-helper','config','candidate','candidate-audit','causal-module','block-adapter','causal-audit','raw-source-view','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();started=time.monotonic()
    if not args.execute:raise RuntimeError('separate scientific execution approval required')
    if args.output.exists():raise FileExistsError(args.output)
    try:
        causal=load(args.causal_module,CAUSAL_SHA,'p023_causal');block_module=load(args.block_adapter,BLOCK_SHA,'p023_block');p020=load(args.comparison_helper,P020_SHA,'p023_p020');helper=load(args.gradient_helper,P019_SHA,'p023_p019')
        diagnostic,objective=helper.load_dependencies(args)
        result=execute(args,causal,block_module,p020,helper,diagnostic,objective,started)
        result['source_sha256']={key:sha(getattr(args,key)) for key in ('diagnostic_script','gradient_helper','comparison_helper','causal_module','block_adapter','causal_audit','config','candidate_audit')}
        result['harness_sha256']=sha(Path(__file__));result['elapsed_seconds']=time.monotonic()-started
        with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    except Exception as error:
        with args.output.with_name(args.output.name+'.failure.json').open('x') as stream:json.dump(dict(status='FC_P023_FAILED_NOT_ADMISSION',error=repr(error),elapsed_seconds=time.monotonic()-started),stream,indent=2)
        raise

if __name__=='__main__':main()
