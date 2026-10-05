#!/usr/bin/env python3
"""Project-owned FC-P016 train-only local fit probe; no saved model/admission."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

P014_SHA = '849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d'
UPDATES = 32
WINDOWS = 6
PANELS = (0, 8, 16, 32)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_dependencies(source_root, diagnostic_script):
    if sha(diagnostic_script) != P014_SHA:
        raise ValueError('pinned P014 diagnostic differs')
    spec = importlib.util.spec_from_file_location('p016_pinned_p014', diagnostic_script)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    for name, expected in diagnostic.SOURCE_SHA.items():
        if sha(source_root / name) != expected:
            raise ValueError('immutable source differs: ' + name)
    sys.path[:0] = [str(source_root / 'src'), str(source_root / 'scripts')]
    return diagnostic, diagnostic.load_objective(source_root)


def rms_statistics(predicted, target, std):
    """Amplitude statistics in float64 from normalized values; mean cancels."""
    import numpy as np
    predicted, target = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if predicted.shape != (100,) or target.shape != (100,) or not (
            np.isfinite(predicted).all() and np.isfinite(target).all() and np.isfinite(std) and std > 0):
        raise ValueError('finite H100 rear-Cl vectors and positive std required')
    p, t = predicted[38:], target[38:]
    p_rms = float(np.sqrt(np.mean((p-p.mean())**2))*std)
    t_rms = float(np.sqrt(np.mean((t-t.mean())**2))*std)
    return dict(predicted_tail62_rms=p_rms, target_tail62_rms=t_rms,
                signed_tail62_rms_error=p_rms-t_rms, absolute_tail62_rms_error=abs(p_rms-t_rms))


def capture_panel(diagnostic, objective, model, states, h1, batch, force_std, predict):
    import torch
    captured = []
    def recording(network, inputs, mask):
        delta, forces = predict(network, inputs, mask)
        captured.append(forces.detach().cpu())
        return delta, forces
    panel = diagnostic.capture_force_panel(objective, model, states, h1, batch, force_std, recording)
    if len(captured) != 10: raise ValueError('mixed20 capture count differs')
    for domain, selection in (('h1', slice(0,10)), ('ar', slice(10,20))):
        predicted = torch.cat([x[selection] for x in captured])
        target = batch['target_force'][0].detach().cpu()
        physical = (predicted-target)*force_std.detach().cpu()
        panel['domains'][domain]['four_force_physical_mae'] = physical.abs().mean(dim=0).tolist()
        panel['domains'][domain]['four_force_physical_rmse'] = physical.square().mean(dim=0).sqrt().tolist()
        panel['domains'][domain]['tail62'].update(rms_statistics(predicted[:,3].numpy(), target[:,3].numpy(), float(force_std[3])))
    panel['amplitude_formula'] = 'float64 std(norm_tail62, ddof=0) * force_std; not centered residual RMSE'
    return panel


def aggregate(rows):
    if len(rows) != WINDOWS or rows[0]['global_index'] != 160:
        raise ValueError('fixed six panel required, zero window first')
    result = {}
    for domain in ('h1','ar'):
        result[domain] = {
            'six_window_macro_objective':sum(r['panel']['objective'][domain+'_balanced'] for r in rows)/WINDOWS,
            'five_nonzero_macro':{key:sum(r['panel']['domains'][domain]['tail62'][key] for r in rows[1:])/5
                for key in ('bias_mse','centered_residual_mse','absolute_tail62_rms_error')},
        }
    return result


def update(model, optimizer, cached, run_window):
    import torch
    if len(cached) != WINDOWS: raise ValueError('exactly six windows required')
    optimizer.zero_grad(set_to_none=True)
    records = [run_window(item) for item in cached]
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('missing/nonfinite raw gradient')
    for p in parameters: p.grad.div_(WINDOWS)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters): raise FloatingPointError('nonfinite updated parameter')
    if any(torch.is_tensor(v) and not torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values()):
        raise FloatingPointError('nonfinite optimizer state')
    return dict(window_records=records, preclip_mean_gradient_norm=float(norm), optimizer_steps=1)


@contextmanager
def readonly_panel(model, objective, device):
    import numpy as np
    import torch
    before = objective.tensor_state_sha256(model)
    gradients = {n:None if p.grad is None else objective.tensor_sha256(p.grad) for n,p in model.named_parameters()}
    rng, numpy_rng, mode = random.getstate(), np.random.get_state(), model.training
    try:
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []), torch.no_grad():
            yield
    finally:
        random.setstate(rng); np.random.set_state(numpy_rng); model.train(mode)
    after_gradients = {n:None if p.grad is None else objective.tensor_sha256(p.grad) for n,p in model.named_parameters()}
    if before != objective.tensor_state_sha256(model) or gradients != after_gradients:
        raise RuntimeError('panel mutated model or gradients')


def terminal_comparison(initial, terminal):
    result = {}
    for domain in ('h1','ar'):
        a, b = initial[domain], terminal[domain]
        pairs = {'six_window_macro_objective':(a['six_window_macro_objective'],b['six_window_macro_objective'])}
        pairs.update({key:(a['five_nonzero_macro'][key],b['five_nonzero_macro'][key]) for key in a['five_nonzero_macro']})
        result[domain] = {key:{'initial':x,'terminal':y,'absolute_delta':y-x,
            'relative_delta':(y-x)/x if x != 0 else None,'strictly_lower':y<x} for key,(x,y) in pairs.items()}
    return dict(domains=result,local_support=all(row['strictly_lower'] for rows in result.values() for row in rows.values()),
                scientific_admission=False,interpretation='local fixed-panel fitting only; terminal32 vs initial0')


def execute(args, diagnostic, objective):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from fluid_control.augmented_datapipe import compose_training_data
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control.calibrated_checkpoint import validate_calibrated_epoch_zero
    from fluid_control.dual_fno import validate_runtime_precision
    from train_tandem_fno import build_model, configured_force_indices, predict

    trainer = objective.load_frozen_trainer(args.source_root/'scripts/train_fcp011_decoder_scope.py')
    cfg = OmegaConf.load(args.config)
    input_sha = trainer.validate_data_contract(cfg)
    resource_checks = [diagnostic.check_memory()]
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.distributed or not dist.cuda: raise RuntimeError('single CUDA device required')
    torch.cuda.set_per_process_memory_fraction(0.15, dist.device)
    precision = validate_runtime_precision()
    random.seed(20261003); np.random.seed(20261003)
    torch.manual_seed(20261003); torch.cuda.manual_seed_all(20261003)
    indices = configured_force_indices(cfg)
    if tuple(indices) != (0,1,2,3): raise ValueError('force channels differ')
    flow, aero = build_model(cfg).to(dist.device), build_model(cfg).to(dist.device)
    for model in (flow,aero):
        epoch = load_checkpoint(args.parent, models=model, metadata_dict={}, device=dist.device)
        validate_calibrated_epoch_zero(args.parent, epoch, allow=True,
            expected_model_sha256=objective.PARENT_MODEL_SHA, expected_state_sha256=objective.PARENT_STATE_SHA,
            expected_kind=objective.PARENT_KIND)
    for p in flow.parameters(): p.requires_grad_(False)
    flow.eval(); aero.train()
    if sum(p.requires_grad for p in aero.parameters()) != 28: raise ValueError('trainable scope differs')
    frozen_aero = {n:objective.tensor_sha256(p) for n,p in aero.named_parameters() if not p.requires_grad}
    if len(frozen_aero) != 2: raise ValueError('two frozen lifting biases required')
    flow_before, aero_before = objective.tensor_state_sha256(flow), objective.tensor_state_sha256(aero)
    if flow_before != aero_before: raise ValueError('official initial model tensors differ')
    optimizer = torch.optim.AdamW(aero.parameters(), lr=1e-5, weight_decay=1e-4)
    base = TandemRolloutDataset(cfg.data.root,'train',100,stride=int(cfg.training.train_stride),
                               num_workers=cfg.training.workers,force_indices=indices)
    train = None
    try:
        train, _ = compose_training_data(base,[Path(x) for x in cfg.data.additional_train_roots],
            rollout_steps=100,stride=int(cfg.training.additional_train_stride),workers=cfg.training.workers,force_indices=indices)
        items = trainer.diagnostic_windows(train)
        diagnostic.validate_windows(items)
        cached = []
        for item in items:
            resource_checks.append(diagnostic.check_memory())
            batch = {k:v[None].to(dist.device) for k,v in item['sample'].items()}
            states = objective.frozen_flow_states(flow,batch['state'],batch['mask'],batch['omega'],predict)
            h1 = objective.true_state_inputs(batch['state'],batch['target_state'])
            if any(not torch.isfinite(v).all() for v in [states,h1,*batch.values()]):
                raise FloatingPointError('nonfinite cached input')
            cached.append(dict(identity={k:item[k] for k in ('global_index','family','identity')},
                states=states.detach().cpu(),h1=h1.detach().cpu(),batch={k:v.detach().cpu() for k,v in batch.items()}))
            del states, h1, batch
            print(json.dumps({'event':'cached_window','global_index':item['global_index']}),flush=True)
        force_std = base.force_std.to(dist.device)
        def transfer(item):
            resource_checks.append(diagnostic.check_memory())
            return item['states'].to(dist.device), item['h1'].to(dist.device), {k:v.to(dist.device) for k,v in item['batch'].items()}
        def panel(step):
            rows = []
            with readonly_panel(aero,objective,dist.device):
                for item in cached:
                    states,h1,batch = transfer(item)
                    rows.append({**item['identity'],'panel':capture_panel(diagnostic,objective,aero,states,h1,batch,force_std,predict)})
                    print(json.dumps({'event':'panel_window','update':step,**item['identity']}),flush=True)
            return dict(update=step,rows=rows,aggregate=aggregate(rows),selection_performed=False)
        def run(item):
            states,h1,batch = transfer(item)
            metrics = objective.chunk_force_objective(aero,states,h1,batch['mask'],batch['omega'],batch['target_force'],
                                                      predict,chunk_size=10,backward=True)
            print(json.dumps({'event':'training_window',**item['identity'],'objective':metrics},allow_nan=False),flush=True)
            return {**item['identity'],'objective':metrics}
        panels, records = [panel(0)], []
        for step in range(1,UPDATES+1):
            records.append(dict(update=step,**update(aero,optimizer,cached,run)))
            if objective.tensor_state_sha256(flow) != flow_before: raise RuntimeError('frozen flow changed')
            if any(p.grad is not None or objective.tensor_sha256(p) != frozen_aero[n]
                   for n,p in aero.named_parameters() if n in frozen_aero):
                raise RuntimeError('frozen aerodynamic bias changed or received gradients')
            if step in PANELS: panels.append(panel(step))
        if any(p.grad is not None for p in flow.parameters()): raise RuntimeError('frozen flow received gradients')
        return dict(status='FC_P016_FIXED_PANEL_FIT_COMPLETE_NOT_ADMISSION',training_experiment='FC-P016',
            optimizer_steps=UPDATES,training_window_exposures=UPDATES*WINDOWS,records=records,panels=panels,
            terminal_comparison=terminal_comparison(panels[0]['aggregate'],panels[-1]['aggregate']),
            scientific_terminal_update=32,source_sha256=diagnostic.SOURCE_SHA,input_sha256=input_sha,
            diagnostic_sha256=P014_SHA,probe_sha256=sha(Path(__file__)),config_sha256=sha(args.config),
            parent_model_sha256=objective.PARENT_MODEL_SHA,parent_state_sha256=objective.PARENT_STATE_SHA,
            flow_tensor_sha256_before=flow_before,flow_tensor_sha256_after=objective.tensor_state_sha256(flow),
            aerodynamic_tensor_sha256_before=aero_before,aerodynamic_tensor_sha256_after=objective.tensor_state_sha256(aero),
            precision=precision,force_training_mode=True,flow_training_mode=False,
            resource_checks=resource_checks,frozen_aerodynamic_parameter_sha256=frozen_aero,
            candidate_saved=False,selection_performed=False,validation_accessed=False,frozen_test_accessed=False,ppo_executed=False,
            hdf_bytes_verification='required external launch evidence; input_sha256 binds manifests/normalization')
    finally:
        (train if train is not None else base).close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','diagnostic-script','config','parent','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    diagnostic, objective = load_dependencies(args.source_root.resolve(),args.diagnostic_script.resolve())
    if sha(args.config) != objective.CONFIG_SHA: raise ValueError('config differs')
    objective.checkpoint_pair(args.parent)
    if args.output.exists(): raise FileExistsError(args.output)
    if not args.execute:
        print('FC_P016_IDENTITIES_VERIFIED_NO_GPU_NO_OPTIMIZATION');return
    diagnostic.write_exclusive(args.output,execute(args,diagnostic,objective))


if __name__ == '__main__': main()
