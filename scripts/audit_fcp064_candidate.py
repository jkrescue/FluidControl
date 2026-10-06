#!/usr/bin/env python3
"""P064 seven-file formal proof adapter over independently verified CPU evidence.

No model construction, forward, optimizer, or scientific admission. Historical
P026 tools remain unchanged. The prior weights-only audit remains explicit.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

LOADER_SHA = '83ac4e4169bc8179cba00d450742f3c4c37a4a3c291ced323ba6c717424b3d7b'
PRIOR = {
    'A': ('2d51c4f84b53fa9b29748c77200cac279ab5a3a4b8ebb5aa6309b4ae9a45f52a', '50de1d8b43ce42ac923752fad76ca4d9', 'fluid-control-fcp064-aero-arm-a-r2-20261006.service'),
    'B': ('d78f87d041fd907c50ad6b2ca8880498bf8f5ad80e5b6916270ec585105b2915', '450ef57c25c14ec38e722cbd597ffb50', 'fluid-control-fcp064-aero-arm-b-20261006.service'),
}
FILES = ('result.json', 'training_protocol.json', 'dual_model_manifest.json',
         'flow/FNO.0.0.mdlus', 'flow/checkpoint.0.0.pt',
         'aerodynamic/FNO.0.1.mdlus', 'aerodynamic/checkpoint.0.1.pt')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def equal(actual, expected, label):
    require(json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False), label)

def fields(actual, expected, label):
    require(isinstance(actual, dict), label)
    equal({k: actual.get(k) for k in expected}, expected, label)

def candidate_files(root):
    root = Path(root).resolve()
    result = {}
    for name in FILES:
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'candidate file escaped/missing')
        result['candidate/' + name] = sha(path)
    return result

def validate_prior(receipt, result, arm, files):
    fields(receipt, dict(status='P064_TERMINAL_ENGINEERING_REVIEW_NOT_ADMISSION',
        result_sha256=files['candidate/result.json'], records=32, consumed=256,
        producer_official_reload=True, independent_model_reload=False), 'prior independent review')
    fields(result, dict(status=f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION',
        arm=arm, history_k=1, optimizer_steps=32, training_windows=256,
        scientific_admission=False, official_fresh_reload_verified=True), 'candidate result')
    fields(receipt['checkpoint'], dict(adam_states=28, all_steps=32, weights_only_cpu=True), 'CPU optimizer audit')
    fields(receipt['unit'], dict(InvocationID=PRIOR[arm][1], MainPID='0', Result='success', ExecMainStatus='0'), 'prior terminal')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'prior-review', 'loader', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output already exists')
    require(sha(args.prior_review) == PRIOR[args.arm][0], 'independent receipt bytes differ')
    require(sha(args.loader) == LOADER_SHA, 'production loader differs')
    files = candidate_files(args.root)
    result = read(args.root/'result.json')
    receipt = read(args.prior_review)
    validate_prior(receipt, result, args.arm, files)
    keys = ('LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainCode', 'ExecMainStatus', 'MainPID', 'InvocationID')
    raw = subprocess.check_output(['systemctl', '--user', 'show', PRIOR[args.arm][2], *['-p'+k for k in keys]], text=True, timeout=15)
    terminal = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    fields(terminal, dict(LoadState='loaded', ActiveState='active', SubState='exited', Result='success', ExecMainCode='1', ExecMainStatus='0', MainPID='0', InvocationID=PRIOR[args.arm][1]), 'actual retained terminal')
    spec = importlib.util.spec_from_file_location('p064_proof_dual', args.loader)
    dual = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dual
    spec.loader.exec_module(dual)
    identity = dual.validate_dual_fno_manifest(args.root/'dual_model_manifest.json')
    require(identity.payload['kind'] == f'FC_P064_ARM_{args.arm}_CONTROLLED_AERO_FORCE_FNO', 'candidate kind')
    require(files['candidate/training_protocol.json'] == result['protocol_sha256'] == identity.payload['training_protocol_sha256'], 'protocol identity')
    tensors = dict(flow=result['flow_tensor_sha256'], aerodynamic=result['aerodynamic_terminal_tensor_sha256'])
    payload = dict(status=f'FC_P064_ARM_{args.arm}_CANDIDATE_INTEGRITY_VERIFIED_NOT_ADMISSION',
        arm=args.arm, history_k=1, actual_optimizer_steps=32, actual_training_windows=256,
        accumulation_windows=8, training_unit=PRIOR[args.arm][2], training_invocation=PRIOR[args.arm][1],
        terminal_evidence=terminal, candidate_sha256=files,
        training_protocol_sha256=files['candidate/training_protocol.json'],
        dual_manifest_sha256=files['candidate/dual_model_manifest.json'], candidate_result_sha256=files['candidate/result.json'],
        tensor_sha256=tensors, role_loader_sha256=LOADER_SHA, independent_cpu_review_sha256=sha(args.prior_review),
        flow_trained=False, aerodynamic_trained=True, aerodynamic_frozen=False,
        dual_adapter_fresh_reload_verified=False, scientific_admission=False, ppo_authorized=False)
    require(candidate_files(args.root) == files, 'candidate changed during audit')
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=payload['status'], receipt_sha256=sha(args.output))))

if __name__ == '__main__':
    main()
