#!/usr/bin/env python3
"""FC-P015 project trainer: fixed eight-window raw-gradient accumulation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys

EXPERIMENT = 'FC-P015'
MANIFEST_STATUS = 'FC_P015_DUAL_FNO_MANIFEST_VERIFIED'
SYSTEM_KIND = 'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO'
AERO_KIND = 'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO_AERODYNAMIC_CHECKPOINT'
P014_SHA = '849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d'
GROUP = 8
WINDOW_COUNT = 1368
UPDATE_COUNT = 171
PANEL_COUNTS = (0, 456, 912, 1368)


def experiment_fields():
    return dict(training_experiment=EXPERIMENT, accumulation_windows=GROUP,
                training_windows=WINDOW_COUNT, optimizer_steps=UPDATE_COUNT)


def load_dependencies(source_root, diagnostic_path):
    import hashlib
    with diagnostic_path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != P014_SHA:
            raise ValueError('pinned P014 diagnostic differs')
    spec = importlib.util.spec_from_file_location('p015_pinned_p014', diagnostic_path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    for name, expected in diagnostic.SOURCE_SHA.items():
        if diagnostic.sha256(source_root / name) != expected:
            raise ValueError('immutable numerical source differs: ' + name)
    sys.path[:0] = [str(source_root / 'src'), str(source_root / 'scripts')]
    return diagnostic, diagnostic.load_objective(source_root)


def accumulation_step(model, optimizer, windows, run_window, audit):
    """Eight raw window gradients /8, then ONE clip and AdamW step."""
    import torch
    optimizer.zero_grad(set_to_none=True)
    records = []
    for window in windows:
        if len(records) == GROUP:
            raise ValueError('too many windows in group')
        records.append(run_window(window))
    if len(records) != GROUP:
        raise ValueError('incomplete group; no update allowed')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('missing/nonfinite accumulated gradient')
    for parameter in parameters:
        parameter.grad.div_(GROUP)
    gradient_audit = audit(model)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('nonfinite updated parameter')
    return dict(records=records, preclip_mean_gradient_norm=float(norm),
                gradient_audit=gradient_audit, windows=GROUP, optimizer_steps=1,
                mean_objective={k: sum(r[k] for r in records)/GROUP
                                for k in ('h1_balanced', 'ar_balanced', 'total')})


def diagnostic_panel(diagnostic, objective, flow, aero, items, force_std, predict, device):
    import numpy as np
    import torch
    diagnostic.validate_windows(items)
    before = (objective.tensor_state_sha256(flow), objective.tensor_state_sha256(aero))
    grads = {n: None if p.grad is None else objective.tensor_sha256(p.grad)
             for n, p in aero.named_parameters()}
    rng, numpy_rng = random.getstate(), np.random.get_state()
    rows = []
    try:
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []), torch.no_grad():
            for item in items:
                diagnostic.check_memory()
                batch = {k: v[None].to(device) for k, v in item['sample'].items()}
                states = objective.frozen_flow_states(flow, batch['state'], batch['mask'], batch['omega'], predict)
                h1 = objective.true_state_inputs(batch['state'], batch['target_state'])
                if not torch.isfinite(states).all() or not torch.isfinite(h1).all():
                    raise FloatingPointError('nonfinite diagnostic input')
                panel = diagnostic.capture_force_panel(objective, aero, states, h1, batch,
                                                        force_std.to(device), predict)
                rows.append({**{k: item[k] for k in ('global_index','family','identity')}, 'panel': panel})
                print(json.dumps({'event':'diagnostic_window_complete','global_index':item['global_index']}),flush=True)
    finally:
        random.setstate(rng)
        np.random.set_state(numpy_rng)
    after = (objective.tensor_state_sha256(flow), objective.tensor_state_sha256(aero))
    after_grads = {n: None if p.grad is None else objective.tensor_sha256(p.grad)
                   for n, p in aero.named_parameters()}
    if before != after or grads != after_grads:
        raise RuntimeError('diagnostic changed model tensors/gradients')
    return dict(rows=rows, tensor_sha256_before=before, tensor_sha256_after=after,
                no_grad=True, model_selection=False)


def manifest_payload(objective, output, aero_model, aero_state, input_sha):
    """Same dual schema; explicitly NOT the legacy P013 experiment profile."""
    flow_model, flow_state = output/'flow/FNO.0.0.mdlus', output/'flow/checkpoint.0.0.pt'
    return {
        'schema_version': 1, 'status': MANIFEST_STATUS, 'kind': SYSTEM_KIND,
        **experiment_fields(), 'config_sha256': objective.CONFIG_SHA,
        'normalization_sha256': objective.NORMALIZATION_SHA,
        'precision_protocol': {'float32_matmul_precision':'high', 'cuda_matmul_allow_tf32':True, 'cudnn_allow_tf32':True},
        'flow_parent_model_sha256': objective.PARENT_MODEL_SHA,
        'flow_parent_state_sha256': objective.PARENT_STATE_SHA,
        'aerodynamic_initial_model_sha256': objective.PARENT_MODEL_SHA,
        'aerodynamic_initial_state_sha256': objective.PARENT_STATE_SHA,
        'architecture': {'in_channels':6,'out_channels':7,'latent_channels':48,'num_fno_layers':5,
                         'num_fno_modes':[32,32],'decoder_layers':2,'decoder_layer_size':128,
                         'padding':8,'coord_features':True,'force_channels':list(objective.FORCE_CHANNELS)},
        'training_semantics': {**experiment_fields(), 'window_count':WINDOW_COUNT,'rollout_steps':100,
            'batch_size':1,'seed':20261003,'sampler_order_sha256':'177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f',
            'h1_force_weight':0.5,'frozen_flow_ar_force_weight':0.5,
            'force_objective':'0.5_equal_four_normalized_mse_plus_0.5_rear_cl_normalized_mse',
            'field_loss_used':False,'chunk_size':10,'optimizer':'AdamW','learning_rate':1e-5,
            'weight_decay':1e-4,'gradient_clip_norm':1.0,'validation_accessed':False,'frozen_test_accessed':False},
        'input_sha256':input_sha,
        'flow': {'role':'flow','frozen':True,'checkpoint_relative_directory':'flow',
                 'model_file':flow_model.name,'state_file':flow_state.name,'checkpoint_epoch':0,
                 'model_sha256':objective.sha256(flow_model),'state_sha256':objective.sha256(flow_state),
                 'metadata_kind':objective.PARENT_KIND},
        'aerodynamic': {'role':'aerodynamic','frozen':False,'checkpoint_relative_directory':'aerodynamic',
                 'model_file':aero_model.name,'state_file':aero_state.name,'checkpoint_epoch':1,
                 'model_sha256':objective.sha256(aero_model),'state_sha256':objective.sha256(aero_state),
                 'metadata_kind':AERO_KIND}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','diagnostic-script','config','parent','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    diagnostic, objective = load_dependencies(args.source_root.resolve(), args.diagnostic_script.resolve())
    if args.output.exists(): raise FileExistsError(args.output)
    if diagnostic.sha256(args.config) != objective.CONFIG_SHA: raise ValueError('config SHA differs')
    parent_model, parent_state = objective.checkpoint_pair(args.parent)
    if not args.execute:
        print('FC_P015_IDENTITIES_VERIFIED_NO_GPU_NO_TRAINING')
        return
    diagnostic.check_memory()
    trainer = objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model, configured_force_indices, predict
    cfg = OmegaConf.load(args.config)
    input_sha = trainer.validate_data_contract(cfg)
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda: raise RuntimeError('exactly one CUDA device required')
    fraction = float(cfg.training.gpu_memory_fraction)
    if not 0 < fraction <= 0.45: raise ValueError('allocator fraction differs')
    torch.cuda.set_per_process_memory_fraction(fraction, dist.device)
    precision = validate_runtime_precision()
    random.seed(20261003); np.random.seed(20261003)
    torch.manual_seed(20261003); torch.cuda.manual_seed_all(20261003)
    indices = configured_force_indices(cfg)
    if tuple(indices) != (0,1,2,3): raise ValueError('force channels differ')
    base = TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),
                               num_workers=cfg.training.workers,force_indices=indices)
    train = None
    try:
        train, _ = compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],
            rollout_steps=100,stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=indices)
        loader = DataLoader(train,batch_size=1,shuffle=True,collate_metadata=True,
            prefetch_factor=int(cfg.data.prefetch_factor),num_streams=int(cfg.data.num_streams),use_streams=True,seed=20261003)
        audit_loader = DataLoader(train,batch_size=1,shuffle=True,prefetch_factor=0,use_streams=False,seed=20261003)
        expected_order = list(iter(audit_loader.sampler))
        if len(loader) != WINDOW_COUNT or trainer.sequence_sha(expected_order) != trainer.EXPECTED_ORDER_SHA:
            raise ValueError('fixed 1368-window order differs')
        flow, aero = build_model(cfg).to(dist.device), build_model(cfg).to(dist.device)
        for model in (flow,aero):
            metadata = {}
            epoch = load_checkpoint(args.parent,models=model,metadata_dict=metadata,device=dist.device)
            validate_calibrated_epoch_zero(args.parent,epoch,allow=True,
                expected_model_sha256=objective.PARENT_MODEL_SHA,expected_state_sha256=objective.PARENT_STATE_SHA,
                expected_kind=objective.PARENT_KIND)
        for parameter in flow.parameters(): parameter.requires_grad_(False)
        flow.eval(); aero.train()
        if sum(p.requires_grad for p in aero.parameters()) != 28: raise ValueError('trainable scope differs')
        flow_before, aero_before = objective.tensor_state_sha256(flow), objective.tensor_state_sha256(aero)
        optimizer = torch.optim.AdamW([p for p in aero.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4)
        items = trainer.diagnostic_windows(train)
        panels = [{'consumed_windows':0,'optimizer_steps':0,
                   **diagnostic_panel(diagnostic,objective,flow,aero,items,base.force_std,predict,dist.device)}]
        identity_map = trainer.identity_index(train)
        observed, records = [], []
        iterator = iter(loader)
        def windows():
            for _ in range(GROUP):
                batch, metadata = next(iterator)
                identity = trainer.training_identity(metadata)
                observed.append(identity_map[(identity['case'],identity['start'],identity['dataset_index'])])
                yield {k:v.to(dist.device,non_blocking=True) for k,v in batch.items()}, identity
        def run(item):
            batch, identity = item
            return {'identity':identity, **objective.run_window(flow,aero,batch,predict,backward=True)}
        for update in range(1,UPDATE_COUNT+1):
            diagnostic.check_memory()
            record = accumulation_step(aero,optimizer,windows(),run,objective.audit_aerodynamic_gradients)
            record.update(update=update,consumed_windows=update*GROUP)
            records.append(record)
            if objective.tensor_state_sha256(flow) != flow_before: raise RuntimeError('frozen flow changed')
            print(json.dumps({'event':'accumulation_update',**record},allow_nan=False),flush=True)
            if update*GROUP in PANEL_COUNTS:
                panels.append({'consumed_windows':update*GROUP,'optimizer_steps':update,
                    **diagnostic_panel(diagnostic,objective,flow,aero,items,base.force_std,predict,dist.device)})
        if observed != expected_order or next(iterator,None) is not None: raise RuntimeError('consumed order differs')
        aero_after = objective.tensor_state_sha256(aero)
        if aero_after == aero_before: raise RuntimeError('aerodynamic model did not change')
        args.output.mkdir(parents=True)
        flow_dir, aero_dir = args.output/'flow', args.output/'aerodynamic'
        flow_dir.mkdir()
        shutil.copy2(parent_model,flow_dir/parent_model.name)
        shutil.copy2(parent_state,flow_dir/parent_state.name)
        metadata = {'status':AERO_KIND,'checkpoint_epoch':1,**experiment_fields(),
            'flow_parent_model_sha256':objective.PARENT_MODEL_SHA,'flow_parent_state_sha256':objective.PARENT_STATE_SHA,
            'aerodynamic_initial_model_sha256':objective.PARENT_MODEL_SHA,'aerodynamic_initial_state_sha256':objective.PARENT_STATE_SHA,
            'training_semantics':'independent_force_fno_h1_ar_equal_mix_balanced_force_only',
            'sampler_order_sha256':trainer.EXPECTED_ORDER_SHA,'input_sha256':input_sha,
            'selection_performed':False,'validation_accessed':False,'frozen_test_accessed':False,'ppo_executed':False}
        save_checkpoint(aero_dir,models=aero,optimizer=optimizer,epoch=1,metadata=metadata)
        for directory, expected_hash, expected_epoch in ((flow_dir,flow_before,0),(aero_dir,aero_after,1)):
            fresh = build_model(cfg).to(dist.device)
            loaded_metadata = {}
            if load_checkpoint(directory,models=fresh,metadata_dict=loaded_metadata,device=dist.device) != expected_epoch:
                raise RuntimeError('official fresh reload epoch differs')
            if objective.tensor_state_sha256(fresh) != expected_hash: raise RuntimeError('fresh reload tensor differs')
            if directory == aero_dir and loaded_metadata != metadata: raise RuntimeError('fresh metadata differs')
            del fresh
        manifest = manifest_payload(objective,args.output,aero_dir/'FNO.0.1.mdlus',aero_dir/'checkpoint.0.1.pt',input_sha)
        diagnostic.write_exclusive(args.output/'dual_model_manifest.json',manifest)
        result = {'status':'FC_P015_WINDOW_ACCUMULATION_TRAINING_COMPLETE_NOT_ADMISSION',**experiment_fields(),
            'records':records,'fixed_train_panels':panels,'official_pair_fresh_reload_verified':True,
            'dual_adapter_fresh_reload_verified':False,'dual_adapter_reload_status':'SEPARATE_P015_PROFILE_VERIFICATION_REQUIRED',
            'flow_tensor_sha256_before':flow_before,'flow_tensor_sha256_after':objective.tensor_state_sha256(flow),
            'aerodynamic_tensor_sha256_before':aero_before,'aerodynamic_tensor_sha256_after':aero_after,
            'dual_model_manifest_sha256':diagnostic.sha256(args.output/'dual_model_manifest.json'),
            'input_sha256':input_sha,'source_sha256':diagnostic.SOURCE_SHA,'p014_diagnostic_sha256':P014_SHA,
            'trainer_sha256':diagnostic.sha256(Path(__file__)),'config_sha256':objective.CONFIG_SHA,'precision':precision,
            'sampler_order_sha256':trainer.sequence_sha(observed),'selection_performed':False,
            'validation_accessed':False,'frozen_test_accessed':False,'ppo_executed':False}
        diagnostic.write_exclusive(args.output/'result.json',result)
    finally:
        (train if train is not None else base).close()


if __name__ == '__main__': main()
