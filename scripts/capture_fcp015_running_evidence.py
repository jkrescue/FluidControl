#!/usr/bin/env python3
"""Capture actual P015 execution identity; never approve a launch retrospectively."""
import hashlib
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/fluid_control')
OUTPUT = ROOT / 'artifacts/fcp015_window_accumulation_training_20261005'
SOURCE = ROOT / 'artifacts/fcp015_window_accumulation_source_20261005_immutable'
NUMERICAL_SOURCE = ROOT / 'artifacts/fcp013_training_source_1634c05_immutable'
IMAGE = 'sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-approval-sha256', required=True)
    args = parser.parse_args()
    approval_sha = args.expected_approval_sha256
    launcher = 'run_fcp015_window_accumulation_spark.sh'
    inspect = json.loads(subprocess.check_output(
        ['docker', 'inspect', 'fcp015-window-accumulation-training-20261005'], text=True, timeout=10))[0]
    unit = subprocess.check_output(['systemctl', '--user', 'show',
        'fluid-control-fcp015-window-accumulation-20261005.service', '-p', 'MainPID', '-p', 'ActiveState',
        '-p', 'SubState', '-p', 'InvocationID', '-p', 'ExecStart'], text=True, timeout=10)
    fields = dict(line.split('=', 1) for line in unit.splitlines() if '=' in line)
    if inspect['Image'] != IMAGE or inspect['State']['Running'] is not True:
        raise RuntimeError('expected training container is not live')
    if fields.get('ActiveState') != 'active' or fields.get('SubState') != 'running' or int(fields.get('MainPID', '0')) <= 0:
        raise RuntimeError('expected training unit is not live')
    if str(SOURCE / 'scripts' / launcher) + ' --execute' not in fields.get('ExecStart', ''):
        raise RuntimeError('unit command differs')
    command = inspect['Config']['Cmd']
    expected_command = ['python', '-u', 'scripts/spark_gpu_guard.py', '--min-free-gib', '20',
        '--allocator-fraction', '.45', '--margin-gib', '4', '--poll-seconds', '2', '--',
        'timeout', '-k', '20', '14400', 'python', '-u', '/workspace/diagnostic/train_fcp015_window_accumulation.py',
        '--source-root', '/workspace/project', '--diagnostic-script', '/workspace/p014_diagnostic.py',
        '--config', '/workspace/config.yaml', '--parent', '/workspace/parent',
        '--output', '/workspace/output/candidate', '--execute']
    if command != expected_command:
        raise RuntimeError('container command differs')
    host = inspect['HostConfig']
    requests = host.get('DeviceRequests', [])
    if (host.get('NetworkMode') != 'none' or host.get('ReadonlyRootfs') is not True
            or host.get('Memory') != 90 * 1024**3 or inspect['Config'].get('User') != '1000:1000'
            or len(requests) != 1 or requests[0].get('DeviceIDs') != ['0']
            or 'no-new-privileges' not in host.get('SecurityOpt', [])):
        raise RuntimeError('container isolation/resource contract differs')
    mounts = {item['Destination']: item for item in inspect['Mounts'] if item['Type'] == 'bind'}
    expected = {
        '/workspace/project': NUMERICAL_SOURCE,
        '/workspace/diagnostic': SOURCE / 'scripts',
        '/workspace/p014_diagnostic.py': ROOT / 'artifacts/fcp014_train_objective_source_20261005_immutable/scripts/diagnose_fcp014_train_objective.py',
        '/workspace/output': OUTPUT,
        '/workspace/parent': ROOT / 'artifacts/fcp009_joint_force_row_candidate_20261005/candidate_build/candidate',
        '/workspace/config.yaml': ROOT / 'artifacts/fcp011_resource_probe_runtime_20261005/resolved_config.yaml',
    }
    for key, family, data in (
        ('base', 'base', 'tandem_cylinders_matched_start_full40_dev30_v1'),
        ('train8', 'train8', 'tandem_cylinders_dynamic_train8_v1'),
        ('train16', 'train16', 'tandem_cylinders_directppo_train16_v1'),
    ):
        expected[f'/workspace/{key}'] = ROOT / f'artifacts/fcp011_official_datapipe_cpu_preflight_20261005/mount_view/{family}'
        expected[f'/workspace/{key}/train'] = ROOT / f'data/curated/{data}/train'
    if set(mounts) != set(expected):
        raise RuntimeError('unexpected bind mount including possible held-out exposure')
    for destination, source in expected.items():
        actual = mounts[destination]
        if Path(actual['Source']).resolve() != source.resolve() or actual['RW'] != (destination == '/workspace/output'):
            raise RuntimeError('bind source or read-only restriction differs')
    if digest(OUTPUT / 'execution_approval.json') != approval_sha:
        raise RuntimeError('approval differs')
    source_hashes = {name: digest(SOURCE / name) for name in (
        'scripts/train_fcp015_window_accumulation.py',
        'scripts/run_fcp015_window_accumulation_spark.sh',
    )}
    receipt = {
        'status': 'FC_P015_RUNNING_EXECUTION_OBSERVED',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'retrospective_launch_approval': False,
        'unit': fields, 'container_id': inspect['Id'], 'image_id': inspect['Image'],
        'container_started_at': inspect['State']['StartedAt'], 'command': command,
        'mounts': inspect['Mounts'], 'host_config': inspect['HostConfig'],
        'source_sha256': source_hashes, 'approval_sha256': approval_sha,
        'validation_or_frozen_mounted': False, 'training_complete': False,
    }
    actual_launcher = SOURCE / 'scripts' / launcher
    if digest(actual_launcher) != digest(OUTPUT / 'immutable_launcher.sh'):
        raise RuntimeError('executed launcher copy differs')
    approved = json.loads((OUTPUT / 'execution_approval.json').read_text())
    if approved['launcher_sha256'] != digest(actual_launcher):
        raise RuntimeError('launcher approval binding differs')
    for name, expected_sha in approved['source_sha256'].items():
        if digest(SOURCE / name) != expected_sha:
            raise RuntimeError('approved source differs: ' + name)
    receipt.update(attempt=1, executed_launcher_sha256=digest(actual_launcher))
    with (OUTPUT / 'running_execution_evidence.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'sha256': digest(OUTPUT / 'running_execution_evidence.json')}))


if __name__ == '__main__':
    main()
