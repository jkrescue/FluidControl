"""Read-only terminal check; saved loss arithmetic, never FNO forward/retraining."""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import subprocess

BASE = Path(__file__).with_name('check_p064_terminal_original.py')
BASE_SHA = '8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
spec = importlib.util.spec_from_file_location('reflection_terminal_base', BASE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
for name, old, new in (
    ('check_records', "f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'", "'FC_P064_Y_REFLECTION_PAIRED_TRAINING_COMPLETE_NOT_ADMISSION'"),
    ('checkpoint_cpu', "f'FC_P064_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'", "'FC_P064_Y_REFLECTION_PAIRED_AERO_CHECKPOINT'"),
):
    source = inspect.getsource(getattr(b, name))
    assert source.count(old) == 1
    exec(source.replace(old, new), b.__dict__)


def paired_records(result, events):
    projected = copy.deepcopy(result)
    expected_events = []
    index = 0
    for group, projected_group in zip(result['records'], projected['records']):
        for row, projected_row in zip(group['records'], projected_group['records']):
            index += 1
            b.require(set(row['branches']) == {'original', 'reflected'}, 'branch labels')
            original, reflected = (row['branches'][x] for x in ('original', 'reflected'))
            b.require(original['identity'] == reflected['identity'], 'paired source identity')
            b.require(original['history'] == reflected['history'], 'paired history identity')
            b.require(original['branch_input_sha256']['mask'] == reflected['branch_input_sha256']['mask'], 'mask symmetry')
            for key in ('h1_balanced', 'ar_balanced', 'total'):
                b.require(math.isclose(row[key], .5 * (original[key] + reflected[key]), rel_tol=1e-12, abs_tol=1e-14), 'half paired objective')
            for offset, (label, branch) in enumerate((('original', original), ('reflected', reflected)), 1):
                b.require(branch['reflection_branch'] == label, 'branch record label')
                b.require(branch['flow_calls'] == 100 and branch['aerodynamic_calls'] == 10, 'branch calls')
                b.require(math.isclose(branch['total'], .5 * (branch['h1_balanced'] + branch['ar_balanced']), rel_tol=2e-6, abs_tol=1e-9), 'H1 AR mixture')
                hashes = branch['branch_input_sha256']
                b.require(set(hashes) == {'state', 'target_state', 'omega', 'target_force', 'mask'}, 'input hash keys')
                b.require(all(len(x) == 64 and all(c in '0123456789abcdef' for c in x) for x in [*hashes.values(), branch['flow_history_sha256']]), 'invalid SHA')
                expected_events.append(dict(event='training_reflection_branch_complete', branch=label,
                    consumed_original_windows=index, completed_transformed_branches=2*(index-1)+offset,
                    flow_calls=100, aerodynamic_calls=10, flow_history_sha256=branch['flow_history_sha256'],
                    branch_input_sha256=hashes))
            projected_row['identity'] = original['identity']
    actual = [e for e in events if e.get('event') == 'training_reflection_branch_complete']
    b.require(index == 256 and actual == expected_events, '512 actual branch events/record binding')
    windows = [e for e in events if e.get('event') == 'training_window_complete']
    b.require([e['completed_transformed_branches'] for e in windows] == list(range(2, 513, 2)), 'window branch counts')
    ordered = [e['event'] for e in events if e.get('event') in ('training_reflection_branch_complete', 'training_window_complete', 'accumulation_update_complete')]
    expected_order = []
    for i in range(256):
        expected_order += ['training_reflection_branch_complete'] * 2 + ['training_window_complete']
        if (i+1) % 8 == 0:
            expected_order.append('accumulation_update_complete')
    b.require(ordered == expected_order, 'actual event order')
    return projected


def reclaimed_terminal(properties, invocation, unit, receipt, journal):
    """Do not invent retained systemd fields after a transient is collected."""
    b.require(properties.get('LoadState') == 'not-found' and properties.get('MainPID') == '0', 'unit is not collected/inactive')
    b.require(receipt.get('status') == 'P064_Y_REFLECTION_PAIRED_SUPERVISOR_COMPLETE'
              and receipt.get('invocation_id') == invocation and receipt.get('unit') == unit,
              'bound successful supervisor receipt missing')
    worker = [x for x in journal if x.get('_SYSTEMD_INVOCATION_ID') == invocation]
    b.require(bool(worker), 'original invocation journal missing')
    last_worker = max(int(x['__REALTIME_TIMESTAMP']) for x in worker)
    ending = [x for x in journal if x.get('USER_UNIT') == unit
              and x.get('MESSAGE_ID') == 'ae8f7b866b0347b9af31fe1c80b127c0'
              and x.get('MESSAGE', '').startswith(unit + ': Consumed ')
              and int(x['__REALTIME_TIMESTAMP']) >= last_worker]
    b.require(len(ending) == 1, 'unambiguous manager completion missing')
    b.require(not any('Failed with result' in x.get('MESSAGE', '') or 'Main process exited, code=' in x.get('MESSAGE', '') for x in journal
                      if int(x['__REALTIME_TIMESTAMP']) >= min(int(w['__REALTIME_TIMESTAMP']) for w in worker)), 'manager failure evidence')
    return dict(mode='collected_transient_supervisor_and_journal',
                observed_systemctl=properties, invocation=invocation,
                manager_completion=ending[0]['MESSAGE'],
                manager_completion_timestamp=ending[0]['__REALTIME_TIMESTAMP'],
                retained_unit_limits_available=False,
                resource_contract_verified_by_bound_supervisor=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--approval-sha256', required=True)
    p.add_argument('--invocation', required=True)
    args = p.parse_args()
    b.require(b.sha(args.approval) == args.approval_sha256, 'approval SHA')
    approval = b.read(args.approval)
    b.require(approval['execution_authorized'] is True, 'not authorized')
    props = dict(line.split('=', 1) for line in subprocess.check_output(['systemctl', '--user', 'show', approval['planned_unit'], '--property=LoadState,InvocationID,MainPID,ExecMainCode,ExecMainStatus,Result,MemoryPeak,MemoryMax,MemorySwapMax'], text=True).splitlines())
    if props.get('LoadState') == 'not-found':
        # Explicitly permitted fallback: read only lifecycle evidence before any model/result.
        raw = subprocess.check_output(['journalctl', '--user', '-u', approval['planned_unit'], '-o', 'json', '--no-pager'], text=True)
        lifecycle = b.read(Path(approval['planned_output'])/'supervisor_receipt.json')
        terminal_evidence = reclaimed_terminal(props, args.invocation, approval['planned_unit'], lifecycle,
                                               [json.loads(line) for line in raw.splitlines()])
    else:
        b.terminal(props, args.invocation)  # Before all candidate reads.
        b.require(props['MemoryMax'] == str(24*2**30) and props['MemorySwapMax'] == '0', 'unit memory contract')
        terminal_evidence = dict(mode='retained_unit', observed_systemctl=props)
    for item in [approval[x] for x in ('worker', 'reflection', 'supervisor', 'source_manifest')]:
        b.require(b.sha(item['path']) == item['sha256'], 'overlay source SHA')
    root = Path(approval['source_root'])
    source_map = b.read(approval['source_manifest']['path'])
    for rel, digest in source_map.items():
        b.require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'source path')
        b.require(b.sha(root/rel) == digest, 'source closure SHA')
    for name, item in approval['inputs'].items():
        if name != 'b00_source_hdf':  # Large payload was hashed by bound supervisor; not re-read here.
            b.require(b.sha(item['path']) == item['sha256'], 'input SHA: '+name)
    argv = approval['worker_argv']
    cli = {argv[i][2:].replace('-', '_'): argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
    output = Path(approval['planned_output'])
    result = b.read(output/'result.json')
    manifest = b.read(output/'dual_model_manifest.json')
    protocol = b.read(output/'training_protocol.json')
    receipt = b.read(output/'supervisor_receipt.json')
    b.require(receipt['invocation_id'] == args.invocation and receipt['result_sha256'] == b.sha(output/'result.json'), 'supervisor result binding')
    journal = subprocess.check_output(['journalctl', '--user', '_SYSTEMD_INVOCATION_ID='+args.invocation, '-o', 'cat', '--no-pager'], text=True)
    events = []
    for line in journal.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    b.check_records(paired_records(result, events), events, b.read(cli['parent_order']), 'B')
    b.require(b.sha(output/'training_protocol.json') == result['protocol_sha256'] == manifest['training_protocol_sha256'], 'protocol SHA')
    b.require(manifest['training_semantics'] == protocol, 'manifest protocol')
    for key, expected in dict(y_reflection_pairing=True, original_window_loss_weight=.5, reflected_window_loss_weight=.5, original_training_windows=256, transformed_window_equivalents=512, expected_training_flow_calls=51200, expected_training_aerodynamic_calls=5120).items():
        b.require(protocol[key] == expected, 'reflection protocol: '+key)
    b.require(manifest['kind'] == 'FC_P064_Y_REFLECTION_PAIRED_K1_FRESH_FORCE_FNO' and manifest['status'] == 'FC_P064_Y_REFLECTION_PAIRED_DUAL_FNO_MANIFEST_VERIFIED', 'candidate identity')
    b.require(result['trainer_sha256'] == approval['worker']['sha256'], 'actual trainer SHA')
    b.require(result['source_sha256'] == manifest['source_sha256'], 'source mapping')
    for name, digest in result['source_sha256'].items():
        b.require(b.sha(cli[name]) == digest, 'actual direct source: '+name)
    parent = b.read(cli['parent_manifest'])
    b.require(manifest['parent_manifest_sha256'] == b.sha(cli['parent_manifest']), 'parent manifest')
    for role in ('flow', 'aerodynamic'):
        for kind in ('model', 'state'):
            entry = manifest[role]
            b.require(b.sha(output/entry['checkpoint_relative_directory']/entry[kind+'_file']) == entry[kind+'_sha256'], 'checkpoint file SHA')
            if role == 'flow':
                b.require(entry[kind+'_sha256'] == parent[role][kind+'_sha256'], 'frozen flow exact bytes')
    b.require(result['flow_tensor_sha256'] == '89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb', 'flow digest')
    b.require(result['aerodynamic_initial_tensor_sha256'] == 'b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb', 'K1 parent digest')
    b.require(result['precision'] == dict(float32_matmul_precision='high', cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True), 'precision')
    b.require(result['allocator_bytes'] == 16*2**30, 'allocator')
    minimum = min(x['MemAvailable'] for x in result['resources'])
    b.require(minimum >= 22 and min(x['mem_available_gib'] for x in receipt['resource_samples']) >= 22, 'runtime reserve')
    checkpoint = b.checkpoint_cpu(output/'aerodynamic/checkpoint.0.1.pt', 'B', result['protocol_sha256'])
    entry = parent['aerodynamic']
    biases = b.frozen_bias_cpu(Path(cli['parent_manifest']).parent/entry['checkpoint_relative_directory']/entry['model_file'], output/'aerodynamic/FNO.0.1.mdlus')
    print(json.dumps(dict(status='Y_REFLECTION_TERMINAL_ENGINEERING_ACCEPT_NOT_ADMISSION', unit=props, terminal_evidence=terminal_evidence,
        approval_sha256=args.approval_sha256, result_sha256=b.sha(output/'result.json'), manifest_sha256=b.sha(output/'dual_model_manifest.json'),
        original_windows=256, branches=512, updates=32, flow_calls=51200, aero_calls=5120, checkpoint=checkpoint,
        frozen_bias_names=biases, min_available_gib=minimum, supervisor_receipt_sha256=b.sha(output/'supervisor_receipt.json'),
        independent_training_loss_recomputed=False, independent_model_forward=False, b00_payload_rehashed=False), indent=2))


if __name__ == '__main__':
    main()
