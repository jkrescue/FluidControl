#!/usr/bin/env python3
"""Read-only official CPU dual reload; no forward, optimizer or admission.

The receipt declares the required image, not an observed Docker image identity.
The approved caller must separately capture actual image/command and bind this receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

IMAGE = 'sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e'
STATUS = 'FC_P015_OFFICIAL_CPU_DUAL_RELOAD_VERIFIED_NOT_ADMISSION'
EXPERIMENT = dict(training_experiment='FC-P015', accumulation_windows=8, training_windows=1368, optimizer_steps=171)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def bindings(manifest, config, training_result, numerical_source):
    payload, result = load(manifest), load(training_result)
    if payload.get('kind') != 'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO':
        raise ValueError('P015 manifest kind required')
    if result.get('status') != 'FC_P015_WINDOW_ACCUMULATION_TRAINING_COMPLETE_NOT_ADMISSION':
        raise ValueError('P015 terminal result required')
    for item in (payload, result):
        if any(item.get(k) != v for k,v in EXPERIMENT.items()):
            raise ValueError('P015 training experiment differs')
    if payload.get('config_sha256') != sha(config) or result.get('dual_model_manifest_sha256') != sha(manifest):
        raise ValueError('manifest/config/result binding differs')
    if result.get('flow_tensor_sha256_before') != result.get('flow_tensor_sha256_after'):
        raise ValueError('training flow changed')
    checkpoint_hashes = {}
    for role in ('flow','aerodynamic'):
        part = payload[role]
        if part['checkpoint_relative_directory'] != role:
            raise ValueError('checkpoint directory differs')
        for field in ('model','state'):
            path = manifest.parent/role/part[field+'_file']
            if not path.resolve().is_relative_to((manifest.parent/role).resolve()):
                raise ValueError('checkpoint path escapes role directory')
            digest = sha(path)
            if digest != part[field+'_sha256']:
                raise ValueError('checkpoint SHA differs')
            checkpoint_hashes[role+'_'+field+'_sha256'] = digest
    source_paths = {name:numerical_source/name for name in (
        'src/fluid_control/dual_fno.py','src/fluid_control/calibrated_checkpoint.py','scripts/train_tandem_fno.py')}
    source_paths.update({name:numerical_source.parent/name for name in (
        'scripts/train_fcp013_independent_force_fno.py','scripts/verify_fcp015_dual_reload.py')})
    tensors = {role:result[role+'_tensor_sha256_after'] for role in ('flow','aerodynamic')}
    if any(not isinstance(v,str) or len(v) != 64 for v in tensors.values()):
        raise ValueError('terminal tensor hashes absent')
    return {'status':STATUS, **EXPERIMENT, 'required_official_image_id':IMAGE, 'device':'cpu',
            'dual_manifest_sha256':sha(manifest),'training_result_sha256':sha(training_result),
            'config_sha256':sha(config),'posteval_chain_receipt_sha256':sha(numerical_source.parent/'receipt.json'),
            'runtime_source_sha256':{k:sha(v) for k,v in source_paths.items()},
            **checkpoint_hashes, 'tensor_sha256':tensors, 'forward_performed':False,
            'optimizer_created':False,'model_saved':False,'gpu_used':False,
            'scientific_admission':False,'ppo_authorized':False}


def validate_receipt(receipt, manifest, config, training_result, numerical_source):
    expected = bindings(manifest,config,training_result,numerical_source)
    if load(receipt) != expected:
        raise ValueError('P015 CPU reload receipt differs from current bound artifacts')
    return expected


def verify_loaded_tensors(adapter, expected, tensor_sha):
    observed = {role:tensor_sha(getattr(adapter,role+'_model')) for role in ('flow','aerodynamic')}
    if observed != expected:
        raise ValueError('official dual-loaded tensors differ from saved training result')
    if any(p.device.type != 'cpu' or p.requires_grad or p.grad is not None for p in adapter.parameters()):
        raise ValueError('dual reload must be frozen, gradient-free and CPU-only')


def execute_cpu(args, expected):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('explicit CUDA_VISIBLE_DEVICES= required for CPU verification')
    import torch
    from omegaconf import OmegaConf
    from fluid_control.dual_fno import load_dual_fno
    from train_tandem_fno import build_model
    from train_fcp013_independent_force_fno import tensor_state_sha256
    if torch.cuda.is_initialized(): raise RuntimeError('CUDA was initialized')
    cfg = OmegaConf.load(args.config)
    with torch.no_grad():
        adapter, identity = load_dual_fno(args.manifest,cfg,torch.device('cpu'),build_model=build_model,
                                         expected_manifest_sha256=expected['dual_manifest_sha256'])
    verify_loaded_tensors(adapter,expected['tensor_sha256'],tensor_state_sha256)
    if identity.payload['kind'] != 'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO' or torch.cuda.is_initialized():
        raise RuntimeError('wrong experiment or CUDA initialized')
    if bindings(args.manifest,args.config,args.training_result,args.numerical_source) != expected:
        raise RuntimeError('bound files changed during CPU reload')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','config','training-result','numerical-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--execute-cpu',action='store_true')
    modes.add_argument('--check-receipt',action='store_true')
    args = parser.parse_args()
    source = args.numerical_source.resolve()
    sys.path[:0] = [str(source/'src'),str(source/'scripts'),str(source.parent/'scripts')]
    validator_path = source.parent/'scripts/validate_fcp008_posteval.py'
    spec=importlib.util.spec_from_file_location('p015_reload_chain_validator',validator_path)
    validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    validator.configure_profile('p015')
    validator.validate_chain_receipt(source.parent/'receipt.json',source)
    if sha(Path(__file__)) != sha(source.parent/'scripts/verify_fcp015_dual_reload.py'):
        raise ValueError('executed verifier differs from frozen chain')
    if args.check_receipt:
        validate_receipt(args.output,args.manifest,args.config,args.training_result,source)
        print('FC_P015_CPU_DUAL_RELOAD_RECEIPT_VALIDATED_NO_MODEL_LOAD')
        return
    if args.output.exists(): raise FileExistsError(args.output)
    expected=bindings(args.manifest,args.config,args.training_result,source)
    execute_cpu(args,expected)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=args.output.parent,delete=False) as stream:
        tmp=Path(stream.name)
        json.dump(expected,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    try: os.link(tmp,args.output)
    finally: tmp.unlink(missing_ok=True)
    print(json.dumps({'status':STATUS,'receipt_sha256':sha(args.output)}))


if __name__ == '__main__': main()
