"""Generate PENDING A/B dev evaluation only from actual successful training.

Metadata/source/checkpoint byte verification only; never deserialize a model,
read HDF arrays, authorize execution, or launch a process doing inference.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/workspace/fluid_control')
SOURCE = ROOT / 'artifacts/fcp064_training_source_20261006_immutable'
SOURCE_SHA = '05c00539ad03cc9059dc101cf9c9cab47d523a8b5bdae29b4af1cea88c66a9c0'
DUAL_SHA = '83ac4e4169bc8179cba00d450742f3c4c37a4a3c291ced323ba6c717424b3d7b'
WORKER_SHA = 'e86ef8ea3c55be63287b0f9d5e9e0cf0034e7fc6959df25e53e49c0fe23d90a8'
CANDIDATE_WORKER_SHA = '0d6f1ac8981d85d092c0e970dcbf1597fa2074d749a75b56f3b82d0cb3bf5af0'
FLOW_SHA = '89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb'
TRAINER_SHA = '8066f4a1e092c566e1b84f706ba56737afa06e13998446fee2f79dc2590920dd'
BASE_SHA = '05891a360ffaa17d61e6854a8dce632d818a3ab1ec5b4a616244d4abb6a85b79'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def item(path):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT) and not path.is_symlink(), 'project path')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_terminal(properties, invocation):
    require(properties.get('InvocationID') == invocation and len(invocation) == 32,
            'training invocation')
    require(properties.get('MainPID') == '0' and properties.get('ExecMainStatus') == '0'
            and properties.get('Result') == 'success'
            and properties.get('SubState') in ('exited', 'dead'), 'successful terminal required')


def validate_result(result, arm):
    require(result.get('status') == f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'
            and result.get('arm') == arm and result.get('history_k') == 1, 'training scope')
    require(type(result.get('optimizer_steps')) is int and result['optimizer_steps'] == 32
            and type(result.get('training_windows')) is int and result['training_windows'] == 256,
            'training counts')
    records = result.get('records', [])
    require(len(records) == 32 and all(r.get('update') == i and r.get('consumed_windows') == 8*i
            for i, r in enumerate(records, 1)), 'complete accumulation records')
    require(result.get('flow_tensor_sha256') == FLOW_SHA, 'frozen flow tensor')
    require(result.get('official_fresh_reload_verified') is True
            and result.get('scientific_admission') is False
            and result.get('engineering_fixture_not_candidate') is not True, 'real candidate proof')
    require(result.get('trainer_sha256') == TRAINER_SHA, 'actual trainer source')


def prepare(args):
    require(args.arm in ('A', 'B'), 'arm')
    approval_path = Path(args.training_approval)
    require(sha(approval_path) == args.training_approval_sha256, 'training approval SHA')
    approval = json.loads(approval_path.read_text())
    require(approval.get('status') == f'FC_P064_ARM_{args.arm}_TRAINING_EXECUTION_APPROVED'
            and approval.get('execution_authorized') is True and approval.get('arm') == args.arm,
            'actual arm training approval')
    output = ROOT / f'artifacts/fcp064_controlled_aero_arm_{args.arm.lower()}_20261006'
    require(approval['planned_unit'] == args.training_unit, 'approved training unit')
    require((ROOT / approval['planned_output']).resolve() == output.resolve()
            and approval['runner_sha256'] == TRAINER_SHA, 'approved output/source')
    keys = ['InvocationID', 'MainPID', 'ExecMainStatus', 'Result', 'ActiveState', 'SubState']
    raw = subprocess.check_output(['systemctl', '--user', 'show', args.training_unit,
        *[v for key in keys for v in ('-p', key)]], text=True, timeout=10)
    properties = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    validate_terminal(properties, args.training_invocation)
    result_path = output / 'result.json'
    result = json.loads(result_path.read_text())
    validate_result(result, args.arm)
    source_manifest = SOURCE / 'source_manifest.json'
    require(sha(source_manifest) == SOURCE_SHA, '434 source manifest')
    inventory = json.loads(source_manifest.read_text())
    require(len(inventory) == 434, 'source count')
    sources = {}
    for name, digest in inventory.items():
        bound = item(SOURCE / name)
        require(bound['sha256'] == digest, 'frozen source bytes')
        sources['training_source:' + name] = bound
    dual_path = SOURCE / 'src/fluid_control/dual_fno.py'
    require(sha(dual_path) == DUAL_SHA, 'production loader')
    dual = load_module(dual_path, 'p064_candidate_metadata_loader')
    manifest_path = output / 'dual_model_manifest.json'
    identity = dual.validate_dual_fno_manifest(manifest_path, expected_sha256=sha(manifest_path))
    manifest = identity.payload
    require(manifest['kind'] == f'FC_P064_ARM_{args.arm}_CONTROLLED_AERO_FORCE_FNO'
            and manifest.get('arm') == args.arm, 'candidate arm')
    require(manifest['training_protocol_sha256'] == result['protocol_sha256']
            and manifest['training_semantics']['schedule_sha256'] == result['schedule_sha256'],
            'result/manifest protocol')
    base_path = ROOT / 'docs/P064_K1_DEVELOPMENT_APPROVAL_20261006.json'
    require(sha(base_path) == BASE_SHA, 'same protocol baseline')
    spec = json.loads(base_path.read_text())
    require(len(spec['runtime_sources']) == 192 and spec['driver']['sha256'] == WORKER_SHA,
            'same worker/runtime')
    for key in ('core', 'reader', 'selector'):
        sources[key] = spec['sources'][key]
    for key, name in {'dual':'src/fluid_control/dual_fno.py',
                      'trainer':'scripts/train_tandem_fno.py',
                      'history':'scripts/p026_state_history.py'}.items():
        sources[key] = item(SOURCE / name)
    spec['sources'] = sources
    spec['driver'] = item(Path(args.worker))
    require(spec['driver']['sha256'] == CANDIDATE_WORKER_SHA, 'selector import-only derived worker')
    spec['pythonpath'] = [str(SOURCE / 'scripts'), str(SOURCE / 'src')]
    spec.update(status='P064_DEVELOPMENT_H1_H5_PENDING_NOT_AUTHORIZED', execution_authorized=False,
        candidate_label=args.arm, output=f'artifacts/p064_arm_{args.arm.lower()}_development_h1_h5_20261006',
        unit=f'fluid-control-p064-arm-{args.arm.lower()}-development-h1-h5-20261006.service')
    spec['inputs']['manifest'] = item(manifest_path)
    spec['inputs']['training_result'] = item(result_path)
    spec['inputs']['training_approval'] = item(approval_path)
    spec['inputs']['training_source_manifest'] = item(source_manifest)
    spec['training_terminal_observation'] = {'unit':args.training_unit, **properties}
    destination = Path(args.pending)
    require(destination.resolve().is_relative_to(ROOT) and not destination.exists(), 'exclusive pending')
    worker = load_module(ROOT / spec['driver']['path'], 'p064_pending_worker')
    test = copy.deepcopy(spec); test.update(status=worker.STATUS, execution_authorized=True)
    inputs, evaluation_output = worker.validate_spec(test)
    require(not evaluation_output.exists(), 'evaluation output already exists')
    records, hdfs = worker.conversion_records(inputs)
    require(len(records) == len(hdfs) == 16, 'same fixed development panel')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as f:
        json.dump(spec, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    persisted = json.loads(destination.read_text())
    persisted.update(status=worker.STATUS, execution_authorized=True)
    worker.validate_spec(persisted)
    print(json.dumps({'pending':str(destination), 'sha256':sha(destination),
                      'execution_authorized':False, 'model_loaded':False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['A', 'B'], required=True)
    for key in ('training-unit', 'training-invocation', 'training-approval',
                'training-approval-sha256', 'pending', 'worker'):
        parser.add_argument('--' + key, required=True)
    prepare(parser.parse_args())
