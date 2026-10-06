"""Project-only fixed same-q0 response loss; official model/predict/reader supplied by caller."""
import hashlib
import json
from pathlib import Path

import torch

RECEIPT_SHA = '6269562b1c4cc92f32853b9c7260e21f702f7ff8249125071fee27365a2be4ce'
PHASES = (0, 2, 4, 6)
ROLES = ('m075', 'zero', 'p075')  # m0375/p0375 are exact first-pair duplicates.
LAMBDA = 1.0


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def protocol():
    return dict(profile='p064_same_q0_response_aux_v1', coefficient=LAMBDA,
                objective='mean_8_contrasts_4_channels_squared_normalized_force_delta_error',
                phases=list(PHASES), roles=list(ROLES), current_frame=0, target_frame=1,
                updates=32, auxiliary_batches_per_update=1, batch_size=12,
                extra_aerodynamic_forward_calls=32, extra_aerodynamic_samples=384,
                extra_flow_forward_calls=0, pre_accumulator_divide_compensation=8,
                receipt_sha256=RECEIPT_SHA, state_count=4, nonzero_contrasts=8,
                batch_precision_caveat='batch12 differs from original mixed20 force batches; not identical-compute intervention',
                validation_accessed=False, frozen_test_accessed=False)


def load_pairs(base, receipt_path, device):
    """Reuse the original train Dataset's official HDF5Reader; never read q1 as model input."""
    if sha(receipt_path) != RECEIPT_SHA or base.split != 'train' or base.force_indices != (0, 1, 2, 3):
        raise ValueError('fixed train first-pair identity required')
    proof = json.loads(Path(receipt_path).read_text())
    expected = {r['case']: r for r in proof['rows']}
    if len(expected) != 20 or not proof['all_clocks_and_action_endpoints_exact']:
        raise ValueError('first-pair proof differs')
    inputs, masks, targets, records = [], [], [], []
    for phase in PHASES:
        phase_frames = []
        for role in ROLES:
            case = f'matched_start_acquisition_train_b{phase:02d}_{role}'
            matches = [i for i, p in enumerate(base.paths) if p.stem == case]
            if len(matches) != 1 or sha(base.paths[matches[0]]) != expected[case]['hdf_sha256']:
                raise ValueError('train HDF identity differs')
            reader = base._reader(matches[0])
            q0, _ = reader[0]
            q1, _ = reader[1]
            # Verify official-reader values against independently audited decoded arrays.
            for key in ('state', 'mask', 'time', 'omega', 'force'):
                for j, frame in enumerate((q0, q1)):
                    digest = hashlib.sha256(frame[key].cpu().numpy().tobytes()).hexdigest()
                    if digest != expected[case]['frame_hashes'][key][j]:
                        raise ValueError('official-reader first-pair values differ')
            phase_frames.append(q0)
            mask = q0['mask'].float()
            state = ((q0['state'].float() - base.state_mean) / base.state_std) * mask
            height, width = state.shape[-2:]
            actions = torch.stack((q0['omega'].float().reshape(()), q1['omega'].float().reshape(()))) / base.action_scale
            planes = actions[:, None, None].expand(-1, height, width)
            inputs.append(torch.cat((state, mask, planes), dim=0))
            masks.append(mask)
            targets.append((q1['force'].float() - base.force_mean) / base.force_std)
            records.append(dict(case=case, frames=[0, 1], hdf_sha256=expected[case]['hdf_sha256'],
                                omega=[float(q0['omega'].reshape(())), float(q1['omega'].reshape(()))]))
        for current in phase_frames:
            if any(not torch.equal(current[k], phase_frames[1][k]) for k in ('state', 'mask', 'omega', 'time', 'force')):
                raise ValueError('same phase must share q0/current omega/force')
    return tuple(torch.stack(v).to(device) for v in (inputs, masks, targets)) + (records,)


def response_loss(predicted, target):
    """Normalized force differences cancel the mean; shared zero retains gradient."""
    if predicted.shape != (12, 4) or target.shape != (12, 4) or target.requires_grad:
        raise ValueError('fixed 4 phases x 3 actions x 4 force channels required')
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise FloatingPointError('nonfinite response')
    p, y = predicted.reshape(4, 3, 4), target.reshape(4, 3, 4)
    pd = p[:, (0, 2)] - p[:, 1:2]
    yd = y[:, (0, 2)] - y[:, 1:2]
    return ((pd - yd) ** 2).mean()


def backward_once(model, packed, predict):
    inputs, masks, target, _ = packed
    if any(t.requires_grad for t in (inputs, masks, target)):
        raise ValueError('auxiliary data must be frozen')
    _, forces = predict(model, inputs, masks)
    loss = response_loss(forces, target)
    # Existing accumulator divides ALL gradients by 8 exactly once.
    (8 * LAMBDA * loss).backward()
    return dict(loss=float(loss.detach()), coefficient=LAMBDA, backward_multiplier=8,
                aerodynamic_calls=1, sample_count=12, flow_calls=0)
