"""Read-only extraction of the original B K1 six-window diagnostic.

Caller owns exact source/data/manifest binding and load_dual_fno loading.
This helper accepts already-loaded models; it never loads, optimizes or saves.
Pass the pinned original trainer, B runner, P013 objective, P026 chunk and
P020 metric modules. Both B and candidate must use this same function.
"""
import random
import numpy as np
import torch

WINDOWS = (
    (160, 'base20', 'matched_start_acquisition_train_b00_zero', 320),
    (816, 'train8', 'dynamic_train8_b00_prbs', 90),
    (923, 'train8', 'dynamic_train8_b02_prbs', 100),
    (975, 'train8', 'dynamic_train8_b04_prbs', 0),
    (1077, 'train8', 'dynamic_train8_b06_prbs', 0),
    (1233, 'train16', 'direct_cfd_directppo2048_v1_env0_ep0009_b00', 0),
)


def evaluate_fixed_six(*, flow, aero, dataset, base, device, trainer,
                       original_runner, objective, chunk, history, p020,
                       helper, predict, guard):
    """Original continuous AR100 and teacher-forced H1, highest/noTF32.

    `dataset` is the unchanged composed original44 dataset, with `_datasets`
    in the original [base20, train8, train16] order and original strides
    [20, 2, 2]. It is not the selected 256-point panel or ScheduledABDataset.
    `base` supplies original force std. No auxiliary loss.
    Model train/eval mode is preserved exactly as in the original panel.
    """
    items = trainer.diagnostic_windows(dataset)
    actual = tuple((x['global_index'], x['family'], x['identity']['case'],
                    x['identity']['start']) for x in items)
    if actual != WINDOWS:
        raise ValueError('original fixed-six identity/order differs')
    if any(x['identity']['split'] != 'train' for x in items):
        raise ValueError('fixed-six must remain train-only')
    device = torch.device(device)
    models = (flow, aero)
    def signatures():
        return [(objective.tensor_state_sha256(m),
                 {n: None if p.grad is None else objective.tensor_sha256(p.grad)
                  for n,p in m.named_parameters()},
                 {n:s.training for n,s in m.named_modules()}) for m in models]
    before = signatures()
    py, npr = random.getstate(), np.random.get_state()
    old_precision = (torch.get_float32_matmul_precision(),
                     torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32)
    rows = []
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def counted(*args):
        guard()
        if (torch.get_float32_matmul_precision() != 'highest'
                or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32):
            raise RuntimeError('diagnostic precision changed')
        return predict(*args)
    try:
        devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for item in items:
                guard()
                sample, ident = item['sample'], item['identity']
                pre, actions, padding = original_runner.preceding(
                    dataset, ident, sample, 1, history)
                batch = {k:v[None].to(device) for k,v in sample.items()}
                states = objective.frozen_flow_states(
                    flow, batch['state'], batch['mask'], batch['omega'], counted)
                state_sha = objective.tensor_sha256(states)
                h1 = objective.true_state_inputs(batch['state'], batch['target_state'])
                result = chunk.chunk_force_objective(
                    aero, states, h1, batch['mask'], batch['omega'],
                    batch['target_force'], pre.to(device), actions.to(device),
                    counted, objective, backward=False)
                result.update(identity=ident, history=padding,
                              flow_history_sha256=state_sha)
                preds = {k:v[0].cpu() for k,v in result.pop('normalized_predictions').items()}
                if set(preds) != {'h1', 'ar'}:
                    raise ValueError('original lowercase h1/ar prediction domains required')
                metrics = p020.metrics(helper, preds, sample['target_force'],
                                       base.force_std, result)
                arrays = {k: v.tolist() for k, v in preds.items()}
                arrays.update(target_force=sample['target_force'].cpu().tolist(),
                              force_std=base.force_std.cpu().tolist())
                rows.append(dict(global_index=item['global_index'], identity=ident,
                                 history=padding, flow_history_sha256=state_sha,
                                 panel=metrics, arrays=arrays))
        if signatures() != before:
            raise RuntimeError('fixed-six mutated tensors/gradients/mode')
        return dict(rows=rows, aggregate=p020.aggregate(rows),
                    history_subgroups=original_runner.grouped_panel(rows),
                    inference_precision=dict(matmul='highest', cuda_tf32=False,
                                             cudnn_tf32=False),
                    optimizer_steps=0, candidate_saved=False)
    finally:
        random.setstate(py)
        np.random.set_state(npr)
        torch.set_float32_matmul_precision(old_precision[0])
        torch.backends.cuda.matmul.allow_tf32 = old_precision[1]
        torch.backends.cudnn.allow_tf32 = old_precision[2]
