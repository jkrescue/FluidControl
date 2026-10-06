"""Project offline HDF error decomposition; not admission or control evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

ORIGIN = 51
HORIZON = 10
FAMILIES = {
    'base': ('tandem_cylinders_matched_start_full40_dev30_v1', 20),
    'train8': ('tandem_cylinders_dynamic_train8_v1', 8),
    'train16': ('tandem_cylinders_directppo_train16_v1', 16),
}


def finite(value):
    ok = torch.isfinite(value).all() if torch.is_tensor(value) else np.isfinite(value).all()
    if not bool(ok):
        raise ValueError('nonfinite diagnostic input/output')
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'source/input SHA mismatch: {path}')


def read_origin(dataset, file_index):
    """Existing official reader; retain only fourteen field frames, no target edits."""
    reader = dataset._reader(file_index)
    states, actions, physical_actions, forces, times = [], [], [], [], []
    mask = None
    for index in range(62):
        frame, _ = reader[index]
        current_mask = frame['mask'].float()
        if mask is None:
            mask = current_mask.clone()
        if not torch.equal(mask, current_mask):
            raise ValueError('mask changed within trajectory')
        forces.append(frame['force'].float().reshape(4).clone())
        physical_actions.append(float(frame['omega'].float().reshape(())))
        times.append(float(frame['time'].reshape(-1)[0]))
        if index >= 48:
            states.append(((frame['state'].float() - dataset.state_mean) / dataset.state_std * mask).clone())
            actions.append(frame['omega'].float().reshape(()) / dataset.action_scale)
    result = dict(states=torch.stack(states), actions=torch.stack(actions), mask=mask,
                  forces=torch.stack(forces), times=np.asarray(times),
                  physical_actions=np.asarray(physical_actions), origin=ORIGIN)
    for key in ('states', 'actions', 'mask', 'forces', 'times', 'physical_actions'):
        finite(result[key])
    if not np.all(np.diff(result['times']) > 0):
        raise ValueError('nonmonotone stored times')
    expected_times = result['times'][0] + .1*np.arange(62)
    tolerance = 2*float(np.max(np.spacing(np.abs(result['times']).astype(np.float32))))
    if np.max(np.abs(result['times']-expected_times)) > max(tolerance, 1e-7):
        raise ValueError('stored0.1 grid differs beyond float32 timestamp quantization')
    result['time_grid_tolerance'] = max(tolerance, 1e-7)
    return result


def flow_input(q, mask, now, nxt):
    return torch.cat((q, mask, torch.ones_like(mask)*now, torch.ones_like(mask)*nxt), dim=1)


@torch.no_grad()
def predict_origin(flow, aerodynamic, sample, force_mean, force_std, guard=lambda: None):
    """Share one frozen flow rollout; aero outputs never feed state recurrence."""
    from p026_history_inference import trajectory_history, pack_history_input, advance_history
    states, actions, mask = sample['states'], sample['actions'], sample['mask'][None]
    if states.shape[0:2] != (14, 3) or actions.shape != (14,) or set(aerodynamic) != {1, 4}:
        raise ValueError('fixed origin51/H10/K1K4 contract')
    for model in (flow, *aerodynamic.values()):
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError('models must be frozen eval with no gradients')
    q = states[3:4].clone()
    trajectory = [q.clone()]
    for j in range(10):
        guard()
        raw = finite(flow(flow_input(q, mask, actions[3+j], actions[4+j])))
        if raw.shape != (1, 7, *q.shape[-2:]):
            raise ValueError('official flow raw output shape')
        q = finite((q + raw[:, :3]) * mask)
        trajectory.append(q.clone())
    predictions = {}
    for k, model in aerodynamic.items():
        ar = trajectory_history(states, actions, torch.tensor([3], device=states.device), k=k)
        rows = {'h1': [], 'ar': []}
        for j in range(10):
            h1 = trajectory_history(states, actions, torch.tensor([3+j], device=states.device), k=k)
            for label, buffer in (('h1', h1), ('ar', ar)):
                guard()
                raw = finite(model(pack_history_input(buffer, mask, actions[4+j:5+j])))
                if raw.shape != (1, 7, *q.shape[-2:]):
                    raise ValueError('official aerodynamic raw output shape')
                normalized = (raw[:, 3:] * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
                rows[label].append(finite(normalized * force_std + force_mean)[0].cpu().numpy().copy())
            ar = advance_history(ar, trajectory[j+1], actions[4+j:5+j])
        predictions[f'k{k}'] = {label: np.stack(values) for label, values in rows.items()}
    return predictions


def metrics(prediction, truth, prefix, baseline, *, omega=None, delta_omega=None):
    """All ten endpoints plus separate canonical mixed62 statistics; no PASS output."""
    from fluid_control.canonical_joint_v1 import canonical_force_ledger, canonical_joint_cost_components
    pred, target, past = [np.asarray(finite(v), dtype=np.float64) for v in (prediction, truth, prefix)]
    if pred.shape != (10, 4) or target.shape != (10, 4) or past.shape != (52, 4):
        raise ValueError('exact10 predicted and52 stored endpoints required')
    residual = pred-target
    cl = residual[:, 3]
    total = residual[:, 0]+residual[:, 2]
    mix = np.concatenate((past, pred))
    reference = np.concatenate((past, target))
    p_ledger = canonical_force_ledger(mix, baseline, window_ready=True)
    t_ledger = canonical_force_ledger(reference, baseline, window_ready=True)
    # Do not export physical gate booleans as a diagnostic admission verdict.
    fields = ('mean_total_drag', 'mean_rear_cl', 'rear_cl_fluctuation_rms',
              'total_drag_reduction', 'rear_cl_fluctuation_ratio',
              'abs_mean_rear_cl_over_baseline_clprime_rms')
    costs = {'available': False, 'reason': 'physical endpoint actions not supplied'}
    if omega is not None and delta_omega is not None:
        try:
            costs = dict(available=True,
                predicted=canonical_joint_cost_components(p_ledger, omega=omega, delta_omega=delta_omega),
                truth=canonical_joint_cost_components(t_ledger, omega=omega, delta_omega=delta_omega))
        except ValueError as error:
            costs = dict(available=False, reason=str(error), omega=omega, delta_omega=delta_omega)
    costs.update(scope='terminal_endpoint61_only_not_all_ten_actions', omega=omega, delta_omega=delta_omega)
    return dict(diagnostic_only=True, terminal_endpoint_cost=costs,
        prediction=pred.tolist(), target=target.tolist(), signed_error=residual.tolist(),
        absolute_error=np.abs(residual).tolist(), squared_error=(residual**2).tolist(),
        four_mae=np.abs(residual).mean(0).tolist(), four_rmse=np.sqrt((residual**2).mean(0)).tolist(),
        four_bias=residual.mean(0).tolist(), total_drag_error=total.tolist(),
        ten_point=dict(rear_cl_signed_mean_error=float(cl.mean()),
            rear_cl_centered_residual_mse=float(np.mean((cl-cl.mean())**2)),
            predicted_rms=float(pred[:, 3].std()), truth_rms=float(target[:, 3].std()),
            absolute_rms_error=float(abs(pred[:, 3].std()-target[:, 3].std()))),
        mixed62={key: dict(predicted=p_ledger[key], truth=t_ledger[key],
                          error=p_ledger[key]-t_ledger[key]) for key in fields},
        baseline=baseline, reference_label='offline_stored_HDF_interpolated_not_verified_online_causal')


def summaries(rows):
    groups = {'overall44': rows}
    for row in rows:
        for key in ('family', 'phase'):
            groups.setdefault(f'{key}:{row[key]}', []).append(row)
        groups.setdefault(f"family_phase:{row['family']}:{row['phase']}", []).append(row)
    result = {}
    for group, members in groups.items():
        result[group] = {'count': len(members), 'metrics': {}}
        for path in members[0]['metrics']:
            result[group]['metrics'][path] = {
                key: np.mean([r['metrics'][path][key] for r in members], axis=0).tolist()
                for key in ('absolute_error', 'squared_error', 'signed_error', 'four_mae', 'four_rmse', 'four_bias')}
            agg = result[group]['metrics'][path]
            agg['per_lead_four_rmse'] = np.sqrt(agg['squared_error']).tolist()
            agg['four_rmse_mean_case'] = agg.pop('four_rmse')
            agg['four_rmse_pooled'] = np.sqrt(np.mean(agg['squared_error'], axis=0)).tolist()
            drag = np.asarray([r['metrics'][path]['total_drag_error'] for r in members])
            agg['per_lead_total_drag'] = dict(mae=np.abs(drag).mean(0).tolist(),
                rmse=np.sqrt((drag**2).mean(0)).tolist(), bias=drag.mean(0).tolist())
            result[group]['metrics'][path]['ten_point'] = {
                key: float(np.mean([r['metrics'][path]['ten_point'][key] for r in members]))
                for key in members[0]['metrics'][path]['ten_point']}
            result[group]['metrics'][path]['mixed62_error'] = {
                key: float(np.mean([r['metrics'][path]['mixed62'][key]['error'] for r in members]))
                for key in members[0]['metrics'][path]['mixed62']}
            valid = [r['metrics'][path]['terminal_endpoint_cost'] for r in members
                     if r['metrics'][path]['terminal_endpoint_cost']['available']]
            result[group]['metrics'][path]['cost_subset'] = {
                'valid_count': len(valid), 'invalid_count': len(members)-len(valid),
                'force_statistics_count': len(members),
                'mean': {side: {key: float(np.mean([v[side][key] for v in valid]))
                                for key in valid[0][side]} for side in ('predicted', 'truth')} if valid else None}
        result[group]['paired_ar_minus_h1'] = {}
        for k in (1, 4):
            ar, h1 = f'k{k}_ar', f'k{k}_h1'
            if ar not in members[0]['metrics'] or h1 not in members[0]['metrics']:
                continue
            result[group]['paired_ar_minus_h1'][f'k{k}'] = {
                key: np.mean([np.asarray(r['metrics'][ar][key])-np.asarray(r['metrics'][h1][key])
                              for r in members], axis=0).tolist()
                for key in ('absolute_error', 'squared_error', 'four_mae')}
    return result


def phase_metadata(config, source_config=None):
    """Use bound source restart metadata, never a phase guessed from a filename."""
    if config.get('split') != 'train':
        raise ValueError('case metadata is not train')
    authoritative = config
    if 'source_case' in config:
        if not isinstance(source_config, dict) or source_config.get('case') != config['source_case']:
            raise ValueError('bound source-case metadata required')
        if source_config.get('split') != 'train':
            raise ValueError('source-case metadata is not train')
        for key in ('source_restart_case', 'source_restart_time'):
            if source_config.get(key) != config.get(key):
                raise ValueError('episode/source restart identity differs')
        authoritative = source_config
    phase = authoritative.get('phase_bin')
    if type(phase) is not int or phase not in range(8):
        raise ValueError('authoritative source phase_bin missing')
    return dict(phase=f'b{phase:02d}', declared_episode_phase=config.get('phase'),
                source_case=config.get('source_case', config.get('case')),
                source_restart_case=config.get('source_restart_case'),
                source_restart_time=config.get('source_restart_time'))


def train16_metadata(case, predeclaration):
    entry = predeclaration['cases'][case]
    if entry['split'] != 'train' or entry['phase'] not in ('b00', 'b02'):
        raise ValueError('train16 declared phase/split differs')
    if entry['env_index'] not in (0, 1) or entry['expected_frames'] != 129 or entry['control_dt'] != .1:
        raise ValueError('train16 declaration differs')
    return dict(phase=entry['phase'], episode=entry['episode'], env_index=entry['env_index'],
                run_window=entry['run_window'],
                phase_scope='within_train16_not_cross_family_physical_phase_equivalence')


def load_baselines(path):
    from fluid_control.canonical_joint_v1 import validate_baseline
    checked(path, 'b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7')
    document = json.loads(Path(path).read_text())
    scope = document.get('scope', {})
    if (document.get('status') != 'FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY'
            or scope.get('split') != 'train' or scope.get('validation_or_frozen_results_read') is not False
            or scope.get('phase_bins') != [0,2,4,6] or set(document['phases']) != {'b00','b02','b04','b06'}):
        raise ValueError('existing train20 baseline scope differs')
    result = {}
    for phase, entry in document['phases'].items():
        row = entry['same_phase_zero']
        if row['case'] != f'matched_start_acquisition_train_{phase}_zero':
            raise ValueError('zero baseline case differs')
        result[phase] = validate_baseline(dict(total_drag=row['metrics']['mean_cd_total'],
            rear_cl_fluctuation_rms=row['metrics']['rms_cl_rear_fluctuation'],
            source=f'{path}:{phase}:long-term final60D/U zero reference; not origin51 same-window baseline'))
    return result


def runtime_dependencies():
    from train_tandem_fno import build_model
    from evaluate_tandem_fno import load_composed_config
    from fluid_control.dual_fno import load_dual_fno
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    return build_model, load_composed_config, load_dual_fno, TandemRolloutDataset


def validate_phase_mapping(mapping, data, predeclaration):
    if mapping.get('status') != 'FCP003C_FULL_TRAIN_SOURCE_PHASE_MAPPING_COMPLETE':
        raise ValueError('reviewed phase mapping status differs')
    result = {}
    for row in mapping['trajectories']:
        family = 'base' if row['family'] == 'base20' else row['family']
        key = (family, row['file'])
        if family not in FAMILIES or key in result:
            raise ValueError('duplicate/unknown phase mapping identity')
        if row['canonical_physical_phase'] not in ('b00', 'b02', 'b04', 'b06'):
            raise ValueError('canonical physical phase invalid')
        extra = train16_metadata(Path(row['file']).stem, predeclaration) if family == 'train16' else {}
        if extra and extra['run_window'][0] != row['physical_start_time']:
            raise ValueError('episode run window/source phase time differs')
        extra.pop('phase', None)
        result[key] = dict(**extra, phase=row['canonical_physical_phase'],
            family_phase_label=row['family_phase_label'], physical_start_time=row['physical_start_time'],
            source_case=row['source_case'])
    expected = {(family, name) for family in FAMILIES for name in data[family]['train_files']}
    if set(result) != expected or len(result) != 44:
        raise ValueError('phase mapping must cover exact44 with no extras')
    for family, (_, count) in FAMILIES.items():
        label = 'base20' if family == 'base' else family
        if (len(data[family]['train_files']) != count
                or mapping['manifest_sha256'][label] != data[family]['manifest_sha256']):
            raise ValueError('phase mapping family/manifest differs')
    return result


def execute(spec, output):
    """Future approved entry; not invoked by software tests or default CLI."""
    if spec.get('status') != 'P027_OFFLINE_DIAGNOSTIC_EXECUTION_APPROVED':
        raise ValueError('separate execution approval required')
    build_model, load_composed_config, load_dual_fno, TandemRolloutDataset = runtime_dependencies()
    started = time.monotonic()
    def guard(startup=False):
        values = {line.split(':')[0]: int(line.split()[1])*1024
                  for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
        if values['MemFree'] < (30 if startup else 20)*2**30 or values['MemAvailable'] < (50 if startup else 20)*2**30:
            raise RuntimeError('host memory guard')
        if time.monotonic()-started > 900:
            raise TimeoutError('whole diagnostic900s bound')
        if torch.cuda.is_initialized() and torch.cuda.mem_get_info()[0] < 20*2**30:
            raise RuntimeError('CUDA20GiB floor')
    guard(True)
    if not spec['source_files'] or len({v['path'] for v in spec['source_files']}) != len(spec['source_files']):
        raise ValueError('nonempty unique source closure required')
    for entry in spec['source_files']:
        checked(entry['path'], entry['sha256'])
    checked(spec['baselines_file']['path'], spec['baselines_file']['sha256'])
    baselines = load_baselines(spec['baselines_file']['path'])
    checked(spec['train_audit']['path'], '03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9')
    original_files = json.loads(Path(spec['train_audit']['path']).read_text())['train_hdf_sha256']
    planned_files = {f'data/curated/{dirname}/train/{name}': value
                     for family, (dirname, _) in FAMILIES.items()
                     for name, value in spec['data'][family]['train_files'].items()}
    if planned_files != original_files or len(planned_files) != 44:
        raise ValueError('pinned original44 file map differs')
    phase_entry = spec['source_phase_mapping']
    if phase_entry['sha256'] != '57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92':
        raise ValueError('phase map declared SHA differs')
    checked(phase_entry['path'], '57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92')
    declared = spec['train16_predeclaration']
    checked(declared['path'], declared['sha256'])
    phases = validate_phase_mapping(json.loads(Path(phase_entry['path']).read_text()), spec['data'],
                                    json.loads(Path(declared['path']).read_text()))
    for family in FAMILIES:
        entry = spec['data'][family]
        checked(Path(entry['root'])/'manifest.json', entry['manifest_sha256'])
        checked(Path(entry['root'])/'normalization.json', entry['normalization_sha256'])
        if entry['normalization_sha256'] != 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1':
            raise ValueError('original normalization required')
    checked(spec['config']['path'], spec['config']['sha256'])
    cfg = load_composed_config(Path(spec['config']['path']))
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(.06)
    device = torch.device('cuda')
    models = {}
    for k in (1, 4):
        entry = spec['candidates'][str(k)]
        model, identity = load_dual_fno(Path(entry['manifest']), cfg, device,
            build_model=build_model, expected_manifest_sha256=entry['manifest_sha256'])
        if identity.payload['kind'] != f'FC_P026_K{k}_HISTORY_FORCE_FNO':
            raise ValueError('candidate arm mismatch')
        if identity.payload['config_sha256'] != spec['config']['sha256']:
            raise ValueError('actual config bytes differ from candidate training config')
        if identity.payload['normalization_sha256'] != 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1':
            raise ValueError('candidate normalization identity differs')
        models[k] = model
    for a, b in zip(models[1].flow_model.state_dict().values(), models[4].flow_model.state_dict().values(), strict=True):
        if not torch.equal(a, b):
            raise ValueError('flow tensors differ across arms')
    models[4].flow_model.cpu()  # One shared device flow, no duplicate flow calls.
    snapshots = [(value, value._version) for model in models.values()
                 for value in list(model.parameters()) + list(model.buffers())]
    rows = []
    for family, (dirname, count) in FAMILIES.items():
        entry = spec['data'][family]
        root = Path(entry['root'])
        checked(root/'manifest.json', entry['manifest_sha256'])
        checked(root/'normalization.json', entry['normalization_sha256'])
        if entry['normalization_sha256'] != 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1':
            raise ValueError('original train normalization required')
        dataset = TandemRolloutDataset(root, 'train', 10, force_indices=(0, 1, 2, 3))
        try:
            if len(dataset.paths) != count or set(p.name for p in dataset.paths) != set(entry['train_files']):
                raise ValueError('exact44 train inventory differs')
            for index, path in enumerate(dataset.paths):
                guard()
                checked(path, entry['train_files'][path.name])
                sample = read_origin(dataset, index)
                for key in ('states', 'actions', 'mask'):
                    sample[key] = sample[key].to(device)
                predictions = predict_origin(models[1].flow_model,
                    {k: model.aerodynamic_model for k, model in models.items()}, sample,
                    dataset.force_mean.to(device), dataset.force_std.to(device), guard)
                forces = sample['forces'].numpy()
                provenance = phases[(family, path.name)]
                baseline = baselines[provenance['phase']]
                omega = float(sample['physical_actions'][61])
                delta = omega-float(sample['physical_actions'][60])
                values = {f'{arm}_{domain}': metrics(pred, forces[52:62], forces[:52], baseline, omega=omega, delta_omega=delta)
                          for arm, domains in predictions.items() for domain, pred in domains.items()}
                values['persistence'] = metrics(np.repeat(forces[51:52], 10, axis=0), forces[52:62], forces[:52], baseline, omega=omega, delta_omega=delta)
                rows.append(dict(case=path.stem, family=family,
                    **provenance, source_phase_mapping_sha256=phase_entry['sha256'], origin=51,
                    warm=True, times=sample['times'].tolist(), actions_normalized=sample['actions'].cpu().tolist(),
                    stored_physical_actions=sample['physical_actions'].tolist(),
                    metrics=values))
                print(json.dumps(dict(event='origin_complete', count=len(rows), case=path.stem)), flush=True)
        finally:
            dataset.close()
    if len(rows) != 44:
        raise ValueError('all44 required')
    if any(value._version != version or (value.is_leaf and value.grad is not None)
           for value, version in snapshots):
        raise ValueError('model tensor mutation or gradient detected')
    result = dict(status='P027_OFFLINE_DIAGNOSTIC_COMPLETE_NOT_ADMISSION', rows=rows,
        summaries=summaries(rows), scientific_admission=False, optimizer_created=False,
        model_saved=False, validation_accessed=False, frozen_test_accessed=False,
        source_spec=spec, flow_transitions=440, aerodynamic_state_evaluations=1760)
    with Path(output).open('x') as stream:
        json.dump(result, stream, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PREPARATION_ONLY_NO_DATA_OR_MODEL_ACCESS')
        return
    checked(args.spec, args.spec_sha256)
    execute(json.loads(args.spec.read_text()), args.output)


if __name__ == '__main__':
    main()
