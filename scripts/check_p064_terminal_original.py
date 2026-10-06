"""Read-only terminal engineering checks; never imports models or reads HDF5.

Run only after terminal approval. Prints evidence, does not write artifacts.
Checkpoint inspection is optional CPU weights_only deserialization, not a forward.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
import zipfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite JSON value')
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def terminal(properties, invocation):
    require(properties.get('InvocationID') == invocation, 'wrong invocation')
    require(properties.get('MainPID') == '0', 'still running; candidate must not be read')
    require(properties.get('ExecMainCode') == '1' and properties.get('ExecMainStatus') == '0', 'not normal exit zero')
    require(properties.get('Result') == 'success', 'unit failed')


def schedule(order, arm):
    require(arm in ('A', 'B'), 'invalid arm')
    require(len(order) == 1368 and all(type(x) is int for x in order)
            and set(order) == set(range(1368)), 'invalid parent order')
    return [dict(consumed=i + 1, original_global_index=order[i],
                 source='controlled_b00' if arm == 'B' and i % 4 == 0 else 'original44',
                 b00_start=(i // 4) * 700 // 63 if arm == 'B' and i % 4 == 0 else None)
            for i in range(256)]


def check_records(result, events, order, arm):
    finite(result)
    require(result['status'] == f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION', 'wrong result status')
    require(result['arm'] == arm and result['history_k'] == 1, 'wrong arm/history')
    require(result['optimizer_steps'] == 32 and result['training_windows'] == 256, 'wrong budget')
    require(result['scientific_admission'] is False and result['official_fresh_reload_verified'] is True,
            'missing producer reload/nonadmission')
    require(len(result['records']) == 32, 'wrong record count')
    for i, record in enumerate(result['records'], 1):
        require(record['update'] == i and record['consumed_windows'] == 8 * i, 'update order')
        require(record['windows'] == 8 and record['optimizer_steps'] == 1 and len(record['records']) == 8, 'accumulation count')
        norm = record['preclip_mean_gradient_norm']
        require(norm >= 0 and math.isclose(record['applied_clip_scale'], min(1., 1. / (norm + 1e-6)), abs_tol=1e-15), 'clip report')
        for key in ('h1_balanced', 'ar_balanced', 'total'):
            mean = sum(x[key] for x in record['records']) / 8
            require(math.isclose(mean, record['mean_objective'][key], rel_tol=1e-12, abs_tol=1e-14), 'objective mean')
        for item in record['records']:
            require(item['identity']['split'] == 'train' and item['identity']['rollout_steps'] == 100, 'nontrain/wrong horizon')
    expected = schedule(order, arm)
    windows = [e for e in events if e.get('event') == 'training_window_complete']
    require([{k: e[k] for k in expected[0]} for e in windows] == expected, 'actual journal schedule differs')
    updates = [e['update'] for e in events if e.get('event') == 'accumulation_update_complete']
    require(updates == list(range(1, 33)), 'journal updates differ')
    ordered = [e for e in events if e.get('event') in ('training_window_complete', 'accumulation_update_complete')]
    require(len(ordered) == 288 and all(ordered[i * 9 + 8]['event'] == 'accumulation_update_complete' for i in range(32)), 'event interleaving')
    require([p['consumed_windows'] for p in result['fixed_train_panels']] == [0, 256], 'panel counts')
    for item, row in zip((x for r in result['records'] for x in r['records']), expected):
        if row['source'] == 'controlled_b00':
            require(item['identity']['dataset_index'] == 3 and item['identity']['start'] == row['b00_start'], 'b00 record identity')
        else:
            require(item['identity']['dataset_index'] in (0, 1, 2), 'original family identity')


def checkpoint_cpu(path, arm, protocol_sha):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    require(checkpoint['epoch'] == 1, 'checkpoint epoch')
    metadata = checkpoint['metadata']
    require(metadata['status'] == f'FC_P064_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'
            and metadata['training_protocol_sha256'] == protocol_sha, 'checkpoint metadata')
    optimizer = checkpoint['optimizer_state_dict']
    require(len(optimizer['param_groups']) == 1, 'optimizer groups')
    group = optimizer['param_groups'][0]
    require(len(group['params']) == 28 and set(group['params']) == set(optimizer['state']), '28 Adam parameter states')
    require(group['lr'] == 1.5625e-7 and tuple(group['betas']) == (.9, .999)
            and group['eps'] == 1e-8 and group['weight_decay'] == 1e-4, 'optimizer hyperparameters')
    for state in optimizer['state'].values():
        require(float(state['step']) == 32, 'Adam step not 32')
        for value in state.values():
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), 'nonfinite optimizer state')
    return {'adam_states': 28, 'all_steps': 32, 'weights_only_cpu': True}


def model_state_cpu(path):
    import torch
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            payload = archive.read('model.pt')
    else:
        with tarfile.open(path) as archive:
            members = [m for m in archive.getmembers() if m.name in ('model.pt', './model.pt') and m.isfile()]
            require(len(members) == 1, 'ambiguous model.pt archive')
            payload = archive.extractfile(members[0]).read()
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)


def frozen_bias_cpu(parent_path, candidate_path):
    import torch
    parent = model_state_cpu(parent_path)
    candidate = model_state_cpu(candidate_path)
    require(set(parent) == set(candidate), 'state dictionary names changed')
    for tensor in candidate.values():
        require(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()), 'nonfinite model tensor')
    names = ['spec_encoder.lift_network.0.conv.bias', 'spec_encoder.lift_network.2.conv.bias']
    for name in names:
        require(torch.equal(parent[name], candidate[name]), 'frozen bias changed: ' + name)
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--approval-sha256', required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--inspect-checkpoint-cpu', action='store_true')
    args = parser.parse_args()
    require(sha(args.approval) == args.approval_sha256, 'approval SHA')
    approval = read(args.approval)
    require(approval['execution_authorized'] is True, 'unapproved')
    unit = approval['planned_unit']
    properties = dict(line.split('=', 1) for line in subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '--property=InvocationID,MainPID,ExecMainCode,ExecMainStatus,Result,MemoryPeak,MemoryMax,MemorySwapMax'], text=True).splitlines())
    terminal(properties, args.invocation)  # Must precede ANY candidate access.
    argv = approval['argv']
    cli = {argv[i][2:].replace('-', '_'): argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith('--')}
    root = Path(approval['source_root'])
    require(sha(root / 'source_manifest.json') == approval['source_manifest_sha256'], 'source manifest')
    source_map = read(root / 'source_manifest.json')
    require(isinstance(source_map, dict) and all(isinstance(v, str) for v in source_map.values()), 'source map schema')
    for name, digest in source_map.items():
        path = root / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(), 'unsafe source path')
        require(sha(path) == digest, 'source SHA: ' + name)
    output = Path(approval['planned_output'])
    result = read(output / 'result.json')
    manifest = read(output / 'dual_model_manifest.json')
    protocol = read(output / 'training_protocol.json')
    order = read(cli['parent_order'])
    require(canonical(order) == approval['parent_order_canonical_sha256'], 'order hash')
    journal = subprocess.check_output(['journalctl', '--user', '_SYSTEMD_INVOCATION_ID=' + args.invocation, '-o', 'cat', '--no-pager'], text=True)
    events = []
    for line in journal.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    check_records(result, events, order, approval['arm'])
    require(sha(output / 'training_protocol.json') == result['protocol_sha256'] == manifest['training_protocol_sha256'], 'protocol digest')
    require(manifest['training_semantics'] == protocol and manifest['source_sha256'] == result['source_sha256'], 'manifest/result source protocol')
    require(result['trainer_sha256'] == approval['runner_sha256'], 'trainer identity')
    require(result['flow_tensor_sha256'] == '89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb', 'flow tensor identity')
    require(result['aerodynamic_initial_tensor_sha256'] == 'b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb', 'aero parent tensor identity')
    require(manifest['parent_manifest_sha256'] == approval['parent_manifest']['sha256']
            == sha(approval['parent_manifest']['path']), 'parent manifest binding')
    require(manifest['status'] == f"FC_P064_ARM_{approval['arm']}_DUAL_FNO_MANIFEST_VERIFIED", 'manifest arm status')
    require(result['schedule_sha256'] == protocol['schedule_sha256'], 'schedule digest binding')
    require(result['effective_training_data'] == manifest['effective_training_data'], 'effective roots binding')
    require(result['precision'] == dict(float32_matmul_precision='high', cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True), 'precision profile')
    for name, digest in result['source_sha256'].items():
        require(sha(cli[name]) == digest, 'direct source/input: ' + name)
    for family in ('base', 'train8', 'train16'):
        view = approval['training_views'][family]
        require(sha(Path(view['root']) / 'manifest.json') == view['manifest_sha256'], 'data manifest')
        require(sha(Path(view['root']) / 'normalization.json') == view['normalization_sha256'], 'normalization')
    parent = read(approval['parent_manifest']['path'])
    for role in ('flow', 'aerodynamic'):
        entry = manifest[role]
        for kind in ('model', 'state'):
            file = output / entry['checkpoint_relative_directory'] / entry[kind + '_file']
            require(sha(file) == entry[kind + '_sha256'], 'saved role file')
            if role == 'flow':
                require(entry[kind + '_sha256'] == parent[role][kind + '_sha256'], 'frozen flow bytes')
    resources = result['resources']
    require(bool(resources), 'no resource observations')
    evidence = dict(status='P064_TERMINAL_ENGINEERING_REVIEW_NOT_ADMISSION', unit=properties,
                    approval_sha256=args.approval_sha256, result_sha256=sha(output / 'result.json'),
                    source_files=len(source_map), records=32, consumed=256,
                    observed_min_available_gib=min(x['MemAvailable'] for x in resources),
                    producer_official_reload=True, independent_model_reload=False,
                    frozen_bias_independent_check='pending explicit tensor inspection; producer checks only',
                    data_payload_rehashed=False)
    if args.inspect_checkpoint_cpu:
        evidence['checkpoint'] = checkpoint_cpu(output / 'aerodynamic/checkpoint.0.1.pt', approval['arm'], result['protocol_sha256'])
        parent_role = parent['aerodynamic']
        parent_file = Path(approval['parent_manifest']['path']).parent / parent_role['checkpoint_relative_directory'] / parent_role['model_file']
        require(sha(parent_file) == parent_role['model_sha256'], 'parent aero archive SHA')
        evidence['frozen_bias_independent_check'] = frozen_bias_cpu(parent_file, output / 'aerodynamic/FNO.0.1.mdlus')
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
