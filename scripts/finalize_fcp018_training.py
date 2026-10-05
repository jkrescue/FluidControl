#!/usr/bin/env python3
"""Verify retained P018 service termination and candidate integrity; no admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

UNIT = 'fluid-control-fcp018-reduced-rate-20261005.service'
INVOCATION = '1ca4654aab074278bb2efdfff8dbc1eb'
APPROVAL_SHA = 'eea5bd5c6a3fe585ae1104600421e50b335f61299c4014c5e3136722af3d4d39'
OBSERVATION_SHA = 'c040eae25fa31a98164e08e10dc4f007eb6a3ef38329ad0dcfaddd734c7eb53c'
ROOT_NAME = 'fcp018_reduced_rate_training_20261005'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def terminal_state(fields):
    if fields.get('LoadState') != 'loaded' or fields.get('InvocationID') != INVOCATION:
        raise ValueError('P018 service absent or invocation differs')
    if fields.get('ActiveState') == 'active' and fields.get('SubState') == 'running':
        if int(fields.get('MainPID', '0')) <= 0:
            raise ValueError('running service has no main process')
        return False
    expected = dict(ActiveState='active', SubState='exited', Result='success',
                    ExecMainStatus='0', ExecMainCode='1', MainPID='0')
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError('P018 retained service did not terminate successfully')
    return True


def read_unit():
    command = ['systemctl', '--user', 'show', UNIT]
    for key in ('LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus',
                'ExecMainCode', 'InvocationID', 'MainPID'):
        command += ['-p', key]
    output = subprocess.check_output(command, text=True, timeout=10)
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def persist_or_verify(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('existing receipt differs: ' + str(path))
        return
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.repo.resolve() / 'artifacts' / ROOT_NAME
    if sha(candidate / 'execution_approval.json') != APPROVAL_SHA:
        raise ValueError('execution approval differs')
    if sha(args.repo / 'docs/FC_P018_RUNNING_EXECUTION_20261005.json') != OBSERVATION_SHA:
        raise ValueError('running observation differs')
    fields = read_unit()
    if not terminal_state(fields):
        print('FC_P018_TRAINING_RUNNING_NO_COMPLETION_RECEIPT')
        return
    from audit_fcp018_candidate import validate_candidate
    audit = validate_candidate(args.repo, candidate, approval_sha=APPROVAL_SHA,
                               observation_sha=OBSERVATION_SHA)
    if not terminal_state(read_unit()):
        raise ValueError('service state changed during terminal audit')
    if audit['invocation_id'] != INVOCATION:
        raise ValueError('candidate audit invocation differs')
    audit_path = candidate / 'candidate_audit.json'
    persist_or_verify(audit_path, audit)
    receipt = dict(status='FC_P018_TRAINING_COMPLETE_NOT_ADMISSION',
                   unit=UNIT, terminal_unit=fields, execution_attempt=1,
                   finalizer_sha256=sha(Path(__file__)),
                   candidate_audit_sha256=sha(audit_path),
                   training_approval_sha256=APPROVAL_SHA,
                   execution_observation_sha256=OBSERVATION_SHA,
                   scientific_admission=False, ppo_executed=False,
                   formal_evaluation_pending=True,
                   dual_adapter_fresh_reload_verified=False)
    for key in ('training_experiment', 'training_windows', 'accumulation_windows',
                'optimizer_steps', 'candidate_result_sha256', 'dual_manifest_sha256',
                'flow_model_sha256', 'flow_state_sha256', 'checkpoint_sha256',
                'checkpoint_state_sha256', 'actual_learning_rate', 'training_protocol_sha256', 'training_protocol_file'):
        receipt[key] = audit[key]
    path = candidate / 'completion_receipt.json'
    persist_or_verify(path, receipt)
    print(json.dumps({'status': receipt['status'], 'receipt_sha256': sha(path)}))


if __name__ == '__main__':
    main()
