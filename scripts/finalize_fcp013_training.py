#!/usr/bin/env python3
"""Project-only terminal audit; never evaluates quality or launches control."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from audit_fcp013_dual_candidate import (
    RECOVERY_ROOT_NAME, RECOVERY_OBSERVATION_SHA, atomic_json, sha256,
    validate_candidate,
)

UNIT = 'fluid-control-fcp013-training-r2-20261005.service'
INVOCATION = 'd0138175399a40f487493919a674e1a5'


def terminal_state(fields: dict) -> bool:
    if fields.get('LoadState') != 'loaded' or fields.get('InvocationID') != INVOCATION:
        raise ValueError('training unit missing or generation differs')
    if fields.get('ActiveState') in ('active', 'activating', 'deactivating'):
        return False
    if (fields.get('ActiveState'), fields.get('SubState'), fields.get('Result'),
            fields.get('ExecMainStatus'), fields.get('ExecMainCode')) != (
            'inactive', 'dead', 'success', '0', '1'):
        raise ValueError('training did not terminate successfully')
    return True


def read_unit() -> dict:
    properties = ('LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus',
                  'ExecMainCode', 'InvocationID')
    command = ['systemctl', '--user', 'show', UNIT]
    for key in properties:
        command += ['-p', key]
    output = subprocess.check_output(command, text=True, timeout=10)
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def persist_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'existing receipt differs: {path}')
    else:
        atomic_json(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=14600)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 14600:
        parser.error('timeout must be in 1..14600 seconds')
    candidate = args.repo / 'artifacts' / RECOVERY_ROOT_NAME
    if sha256(candidate / 'running_execution_evidence.json') != RECOVERY_OBSERVATION_SHA:
        raise ValueError('running observation differs')
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        fields = read_unit()
        if terminal_state(fields):
            break
        if not args.wait:
            print(json.dumps({'status': 'TRAINING_ACTIVE_NO_COMPLETION_RECEIPT'}))
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('observation deadline; training was not restarted')
        time.sleep(min(30, max(0, deadline-time.monotonic())))
    audit = validate_candidate(args.repo, candidate)
    if not terminal_state(read_unit()):
        raise ValueError('training state changed during audit')
    audit_path = candidate / 'candidate_audit.json'
    persist_or_verify(audit_path, audit)
    receipt = {
        'status': 'FC_P013_TRAINING_COMPLETE_NOT_ADMISSION',
        'execution_attempt': 2, 'unit': UNIT, 'terminal_unit': fields,
        'candidate_audit_sha256': sha256(audit_path),
        'finalizer_sha256': sha256(Path(__file__)),
        'optimizer_steps': audit['optimizer_steps'],
        'candidate_result_sha256': audit['candidate_result_sha256'],
        'dual_manifest_sha256': audit['dual_manifest_sha256'],
        'flow_model_sha256': audit['flow_model_sha256'],
        'flow_state_sha256': audit['flow_state_sha256'],
        'aerodynamic_model_sha256': audit['checkpoint_sha256'],
        'aerodynamic_state_sha256': audit['checkpoint_state_sha256'],
        'fixed_six_diagnostics_pending': True, 'formal_evaluation_pending': True,
        'scientific_admission': False, 'ppo_executed': False,
    }
    persist_or_verify(candidate / 'completion_receipt.json', receipt)
    print(json.dumps({'status': receipt['status'],
                      'receipt_sha256': sha256(candidate / 'completion_receipt.json')}))


if __name__ == '__main__':
    main()
