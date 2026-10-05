#!/usr/bin/env python3
"""FC-P018 project trainer: P015 accumulation with one fixed learning-rate override."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil

P015_SHA = '2f5de1a946b040c42c21f8164da41a5afc642fd6174e7bb35372bb7ba98eb996'
EXPERIMENT = 'FC-P018'
MANIFEST_STATUS = 'FC_P018_DUAL_FNO_MANIFEST_VERIFIED'
SYSTEM_KIND = 'FC_P018_REDUCED_RATE_FORCE_FNO'
AERO_KIND = 'FC_P018_REDUCED_RATE_FORCE_FNO_AERODYNAMIC_CHECKPOINT'
LEARNING_RATE = 1.5625e-7
PROTOCOL_SHA = '310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def load_helpers(args):
    if sha(args.accumulation_helper) != P015_SHA: raise ValueError('immutable P015 helper differs')
    spec=importlib.util.spec_from_file_location('p018_pinned_p015',args.accumulation_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    diagnostic,objective=helper.load_dependencies(args.source_root.resolve(),args.diagnostic_script.resolve())
    return helper,diagnostic,objective


def validate_protocol(path, expected_sha):
    if expected_sha!=PROTOCOL_SHA or sha(path)!=expected_sha: raise ValueError('training protocol SHA differs')
    protocol=json.loads(Path(path).read_text())
    if protocol.get('training_experiment')!=EXPERIMENT or protocol.get('sole_optimizer_override')!={'learning_rate':LEARNING_RATE}:
        raise ValueError('fixed P018 experiment/learning rate differs')
    return protocol


def identity(protocol_sha):
    return dict(training_experiment=EXPERIMENT,accumulation_windows=8,training_windows=1368,optimizer_steps=171,
        actual_learning_rate=LEARNING_RATE,training_protocol_sha256=protocol_sha,
        training_protocol_file='training_protocol.json',
        config_role='base_model_data_configuration_not_full_effective_training_configuration')


def check_optimizer(optimizer, protocol, expected_steps=None):
    import torch
    for group in optimizer.param_groups:
        if (group['lr'],group['betas'],group['eps'],group['weight_decay']) != (protocol['sole_optimizer_override']['learning_rate'],(.9,.999),1e-8,1e-4):
            raise ValueError('actual optimizer differs from fixed P018 protocol')
    if any(torch.is_tensor(v) and not torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values()):
        raise FloatingPointError('nonfinite optimizer state')
    if expected_steps is not None and (len(optimizer.state)!=sum(len(g['params']) for g in optimizer.param_groups)
            or any(float(state.get('step',-1))!=expected_steps for state in optimizer.state.values())):
        raise ValueError('actual AdamW step count differs')


def check_frozen(aero, frozen, objective):
    if any(p.grad is not None or objective.tensor_sha256(p)!=frozen[n] for n,p in aero.named_parameters() if n in frozen):
        raise RuntimeError('frozen aerodynamic biases changed or received gradients')


def manifest_payload(helper,objective,output,aero_model,aero_state,input_sha,protocol_sha):
    result=helper.manifest_payload(objective,output,aero_model,aero_state,input_sha)
    result.update(status=MANIFEST_STATUS,kind=SYSTEM_KIND,**identity(protocol_sha))
    result['training_semantics'].update(**identity(protocol_sha),learning_rate=LEARNING_RATE)
    result['aerodynamic']['metadata_kind']=AERO_KIND
    result['base_config_sha256']=objective.CONFIG_SHA
    result['accumulation_helper_sha256']=P015_SHA
    return result


def execute(args,helper,diagnostic,objective,protocol):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint,save_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model,configured_force_indices,predict
    diagnostic.check_memory()
    trainer=objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    cfg=OmegaConf.load(args.config)
    input_sha=trainer.validate_data_contract(cfg)
    DistributedManager.initialize();dist=DistributedManager()
    if dist.distributed or not dist.cuda: raise RuntimeError('exactly one CUDA device required')
    torch.cuda.set_per_process_memory_fraction(.15,dist.device)
    precision=validate_runtime_precision()
    random.seed(20261003);np.random.seed(20261003);torch.manual_seed(20261003);torch.cuda.manual_seed_all(20261003)
    indices=configured_force_indices(cfg)
    if tuple(indices)!=(0,1,2,3): raise ValueError('force channel order differs')
    base=TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),num_workers=cfg.training.workers,force_indices=indices)
    train=None
    try:
        train,_=compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],rollout_steps=100,
            stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=indices)
        loader=DataLoader(train,batch_size=1,shuffle=True,collate_metadata=True,prefetch_factor=int(cfg.data.prefetch_factor),
            num_streams=int(cfg.data.num_streams),use_streams=True,seed=20261003)
        audit_loader=DataLoader(train,batch_size=1,shuffle=True,prefetch_factor=0,use_streams=False,seed=20261003)
        expected_order=list(iter(audit_loader.sampler))
        if len(loader)!=1368 or trainer.sequence_sha(expected_order)!=trainer.EXPECTED_ORDER_SHA: raise ValueError('fixed1368 order differs')
        flow,aero=build_model(cfg).to(dist.device),build_model(cfg).to(dist.device)
        for model in (flow,aero):
            epoch=load_checkpoint(args.parent,models=model,metadata_dict={},device=dist.device)
            validate_calibrated_epoch_zero(args.parent,epoch,allow=True,expected_model_sha256=objective.PARENT_MODEL_SHA,
                expected_state_sha256=objective.PARENT_STATE_SHA,expected_kind=objective.PARENT_KIND)
        for p in flow.parameters(): p.requires_grad_(False)
        flow.eval();aero.train()
        if sum(p.requires_grad for p in aero.parameters())!=28: raise ValueError('trainable scope differs')
        frozen={n:objective.tensor_sha256(p) for n,p in aero.named_parameters() if not p.requires_grad}
        if set(frozen)!=set(objective.OFFICIAL_FROZEN_PARAMETER_NAMES): raise ValueError('frozen bias scope differs')
        flow_before,aero_before=objective.tensor_state_sha256(flow),objective.tensor_state_sha256(aero)
        if flow_before!=aero_before: raise ValueError('official initial model tensors differ')
        optimizer=torch.optim.AdamW([p for p in aero.parameters() if p.requires_grad],lr=protocol['sole_optimizer_override']['learning_rate'],
            betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
        check_optimizer(optimizer,protocol)
        items=trainer.diagnostic_windows(train)
        panels=[dict(consumed_windows=0,optimizer_steps=0,**helper.diagnostic_panel(diagnostic,objective,flow,aero,items,base.force_std,predict,dist.device))]
        identity_map=trainer.identity_index(train)
        observed,records=[],[]
        iterator=iter(loader)
        def windows():
            for _ in range(8):
                batch,metadata=next(iterator)
                ident=trainer.training_identity(metadata)
                observed.append(identity_map[(ident['case'],ident['start'],ident['dataset_index'])])
                yield {k:v.to(dist.device,non_blocking=True) for k,v in batch.items()},ident
        def run(item):
            batch,ident=item
            return dict(identity=ident,**objective.run_window(flow,aero,batch,predict,backward=True))
        for step in range(1,172):
            diagnostic.check_memory();check_optimizer(optimizer,protocol)
            record=helper.accumulation_step(aero,optimizer,windows(),run,objective.audit_aerodynamic_gradients)
            check_optimizer(optimizer,protocol,expected_steps=step);check_frozen(aero,frozen,objective)
            if objective.tensor_state_sha256(flow)!=flow_before or any(p.grad is not None for p in flow.parameters()):
                raise RuntimeError('frozen flow changed or received gradients')
            record.update(update=step,consumed_windows=step*8,actual_learning_rate=optimizer.param_groups[0]['lr'],
                training_protocol_sha256=args.training_protocol_sha256)
            records.append(record)
            print(json.dumps({'event':'accumulation_update',**record},allow_nan=False),flush=True)
            if step*8 in (456,912,1368):
                panels.append(dict(consumed_windows=step*8,optimizer_steps=step,
                    **helper.diagnostic_panel(diagnostic,objective,flow,aero,items,base.force_std,predict,dist.device)))
        if observed!=expected_order or next(iterator,None) is not None: raise RuntimeError('consumed order differs')
        aero_after=objective.tensor_state_sha256(aero)
        if aero_after==aero_before: raise RuntimeError('aerodynamic model did not change')
        validate_protocol(args.training_protocol,args.training_protocol_sha256)
        args.output.mkdir(parents=True)
        shutil.copy2(args.training_protocol,args.output/'training_protocol.json')
        validate_protocol(args.output/'training_protocol.json',args.training_protocol_sha256)
        flow_dir,aero_dir=args.output/'flow',args.output/'aerodynamic';flow_dir.mkdir()
        parent_model,parent_state=objective.checkpoint_pair(args.parent)
        shutil.copy2(parent_model,flow_dir/parent_model.name);shutil.copy2(parent_state,flow_dir/parent_state.name)
        metadata=dict(status=AERO_KIND,checkpoint_epoch=1,**identity(args.training_protocol_sha256),
            base_config_sha256=objective.CONFIG_SHA,accumulation_helper_sha256=P015_SHA,
            flow_parent_model_sha256=objective.PARENT_MODEL_SHA,flow_parent_state_sha256=objective.PARENT_STATE_SHA,
            aerodynamic_initial_model_sha256=objective.PARENT_MODEL_SHA,aerodynamic_initial_state_sha256=objective.PARENT_STATE_SHA,
            training_semantics='independent_force_fno_h1_ar_equal_mix_balanced_force_only',sampler_order_sha256=trainer.EXPECTED_ORDER_SHA,
            input_sha256=input_sha,selection_performed=False,validation_accessed=False,frozen_test_accessed=False,ppo_executed=False)
        save_checkpoint(aero_dir,models=aero,optimizer=optimizer,epoch=1,metadata=metadata)
        for directory,expected,epoch in ((flow_dir,flow_before,0),(aero_dir,aero_after,1)):
            fresh=build_model(cfg).to(dist.device);fresh_metadata={}
            loaded=load_checkpoint(directory,models=fresh,metadata_dict=fresh_metadata,device=dist.device)
            if loaded!=epoch or objective.tensor_state_sha256(fresh)!=expected: raise RuntimeError('official fresh reload differs')
            if directory==aero_dir and fresh_metadata!=metadata: raise RuntimeError('official metadata differs')
            del fresh
        manifest=manifest_payload(helper,objective,args.output,aero_dir/'FNO.0.1.mdlus',aero_dir/'checkpoint.0.1.pt',input_sha,args.training_protocol_sha256)
        diagnostic.write_exclusive(args.output/'dual_model_manifest.json',manifest)
        diagnostic.write_exclusive(args.output/'result.json',dict(status='FC_P018_REDUCED_RATE_TRAINING_COMPLETE_NOT_ADMISSION',
            **identity(args.training_protocol_sha256),records=records,fixed_train_panels=panels,
            official_pair_fresh_reload_verified=True,dual_adapter_fresh_reload_verified=False,
            dual_adapter_reload_status='SEPARATE_P018_PROFILE_VERIFICATION_REQUIRED',
            flow_tensor_sha256_before=flow_before,flow_tensor_sha256_after=objective.tensor_state_sha256(flow),
            aerodynamic_tensor_sha256_before=aero_before,aerodynamic_tensor_sha256_after=aero_after,frozen_parameter_sha256=frozen,
            dual_model_manifest_sha256=sha(args.output/'dual_model_manifest.json'),input_sha256=input_sha,
            source_sha256=diagnostic.SOURCE_SHA,p014_diagnostic_sha256=helper.P014_SHA,p015_helper_sha256=P015_SHA,
            trainer_sha256=sha(Path(__file__)),config_sha256=objective.CONFIG_SHA,base_config_sha256=objective.CONFIG_SHA,
            precision=precision,sampler_order_sha256=trainer.sequence_sha(observed),selection_performed=False,
            validation_accessed=False,frozen_test_accessed=False,ppo_executed=False))
    finally:
        (train if train is not None else base).close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','diagnostic-script','accumulation-helper','config','parent','training-protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--training-protocol-sha256',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    helper,diagnostic,objective=load_helpers(args)
    protocol=validate_protocol(args.training_protocol,args.training_protocol_sha256)
    if sha(args.config)!=objective.CONFIG_SHA: raise ValueError('base configuration SHA differs')
    objective.checkpoint_pair(args.parent)
    if args.output.exists(): raise FileExistsError(args.output)
    if not args.execute: print('FC_P018_IDENTITIES_VERIFIED_NO_GPU_NO_TRAINING');return
    execute(args,helper,diagnostic,objective,protocol)


if __name__=='__main__': main()
