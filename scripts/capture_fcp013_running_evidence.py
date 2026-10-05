#!/usr/bin/env python3
"""Capture live P013 execution evidence, not a retrospective launch approval."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/fluid_control')
OUTPUT = ROOT / 'artifacts/fcp013_independent_force_fno_training_20261005'
SOURCE = ROOT / 'artifacts/fcp013_training_source_1634c05_immutable'
IMAGE = 'sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e'
APPROVAL = '1bdcfcf71d1581bbae66fc6551dde61a500bd95a464d9f61d510cbac1721a120'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inspect = json.loads(subprocess.check_output(
        ['docker', 'inspect', 'fcp013-independent-force-training-20261005'], text=True))[0]
    unit = subprocess.check_output(['systemctl', '--user', 'show',
        'fluid-control-fcp013-training-20261005.service', '-p', 'MainPID', '-p', 'ActiveState',
        '-p', 'InvocationID', '-p', 'ExecStart'], text=True)
    fields = dict(line.split('=', 1) for line in unit.splitlines() if '=' in line)
    if inspect['Image'] != IMAGE or inspect['State']['Running'] is not True:
        raise RuntimeError('expected training container is not live')
    if fields.get('ActiveState') != 'active' or int(fields.get('MainPID', '0')) <= 0:
        raise RuntimeError('expected training unit is not live')
    if 'run_fcp013_training_spark.sh' not in fields.get('ExecStart', ''):
        raise RuntimeError('unit command differs')
    command = inspect['Config']['Cmd']
    if '--resource-probe' in command or 'scripts/train_fcp013_independent_force_fno.py' not in command:
        raise RuntimeError('container command differs')
    mounts = {item['Destination']: item for item in inspect['Mounts'] if item['Type'] == 'bind'}
    expected = {
        '/workspace/project': SOURCE,
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
    if digest(OUTPUT / 'execution_approval.json') != APPROVAL:
        raise RuntimeError('approval differs')
    source_hashes = {name: digest(SOURCE / name) for name in (
        'scripts/train_fcp013_independent_force_fno.py',
        'scripts/train_fcp011_decoder_scope.py', 'src/fluid_control/dual_fno.py',
        'scripts/run_fcp013_training_spark.sh', 'scripts/spark_gpu_guard.py',
    )}
    receipt = {
        'status': 'FC_P013_RUNNING_EXECUTION_OBSERVED',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'retrospective_launch_approval': False,
        'unit': fields, 'container_id': inspect['Id'], 'image_id': inspect['Image'],
        'container_started_at': inspect['State']['StartedAt'], 'command': command,
        'mounts': inspect['Mounts'], 'host_config': inspect['HostConfig'],
        'source_sha256': source_hashes, 'approval_sha256': APPROVAL,
        'validation_or_frozen_mounted': False, 'training_complete': False,
    }
    with (OUTPUT / 'running_execution_evidence.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'sha256': digest(OUTPUT / 'running_execution_evidence.json')}))


if __name__ == '__main__':
    main()
