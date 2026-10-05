#!/usr/bin/env python3
"""External P018 resource supervision; never modify frozen numerical evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO = Path('/workspace/fluid_control')
CHAIN = REPO/'artifacts/p018_posteval_chain_d46622b7cb51_immutable'
CANDIDATE = REPO/'artifacts/fcp018_reduced_rate_training_20261005'
OUTPUT = CANDIDATE/'posteval_fc_p018'
RUNNER = CHAIN/'scripts/run_fcp008_posteval_spark.sh'
IMAGE = 'sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e'
FLOOR_KIB = 20*1024**2
LIMIT_SECONDS = 10800
POLL_SECONDS = 2


class GuardFailure(RuntimeError):
    pass


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_memory(text):
    fields={line.split()[0].rstrip(':'):int(line.split()[1]) for line in text.splitlines()
            if line.startswith(('MemAvailable:','MemFree:'))}
    if set(fields) != {'MemAvailable','MemFree'} or any(v < 0 for v in fields.values()):
        raise GuardFailure('host memory sample incomplete')
    return fields


def read_memory():
    return parse_memory(Path('/proc/meminfo').read_text())


def owns_container(info):
    state=info.get('State',{})
    if info.get('Image') != IMAGE or not (state.get('Running') is True or state.get('Status') == 'created'):
        return False
    mounts={m.get('Destination'):m for m in info.get('Mounts',[]) if m.get('Type') == 'bind'}
    expected={'/workspace/output':(OUTPUT,True),'/workspace/src':(CHAIN/'numerical_source/src',False),
              '/workspace/scripts':(CHAIN/'numerical_source/scripts',False)}
    return all(destination in mounts and mounts[destination].get('Source') == str(source)
               and mounts[destination].get('RW') is writable
               for destination,(source,writable) in expected.items())


def discover_owned():
    ids=subprocess.check_output(['docker','ps','--all','--quiet','--no-trunc'],text=True,timeout=5).split()
    if not ids: return []
    # Containers may exit between ps and inspect. Inspect each ID independently;
    # a confirmed missing object is benign, daemon/permission errors are not.
    result=subprocess.run(['docker','inspect',*ids],text=True,capture_output=True,timeout=5)
    if result.returncode and not all(('No such object:' in line or 'No such container:' in line)
                                   for line in result.stderr.splitlines() if line.strip()):
        raise GuardFailure('docker inspect failed: '+result.stderr[-500:])
    if result.returncode and not result.stderr.strip(): raise GuardFailure('docker inspect failed without diagnostics')
    return [info['Id'] for info in json.loads(result.stdout) if owns_container(info)]


def stop_owned(process):
    """First stop new child launches; stop ONLY freshly verified owned IDs."""
    if process.poll() is None:
        try: os.killpg(process.pid,signal.SIGSTOP)
        except ProcessLookupError: pass
    errors=[]
    try:
        stop_containers()
    except Exception as error:
        errors.append(str(error))
    finally:
        if process.poll() is None:
            for sig in (signal.SIGTERM,signal.SIGCONT):
                try: os.killpg(process.pid,sig)
                except ProcessLookupError: pass
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=5)
    # A Docker request already in flight can finish after the CLI is killed.
    # Reconcile for a bounded grace period after the launcher has terminated.
    try:
        for _ in range(3):
            stop_containers()
            time.sleep(2)
        if discover_owned(): raise GuardFailure('owned container remains after cleanup grace')
    except Exception as error: errors.append(str(error))
    if errors: raise GuardFailure('owned-container cleanup could not be verified: '+'; '.join(errors))


def stop_containers():
    identifiers=discover_owned()
    if not identifiers: return
    # Full immutable IDs only, including created containers with exact mounts.
    result=subprocess.run(['docker','rm','--force',*identifiers],capture_output=True,text=True,timeout=20)
    if result.returncode: raise GuardFailure('owned container removal failed: '+result.stderr[-500:])


def child_environment(args):
    env=os.environ.copy()
    env.update(FCP_POSTEVAL_PROFILE='p018',FCP008_REPO_ROOT=str(REPO),FCP008_POSTEVAL_CHAIN_ROOT=str(CHAIN),
        FCP_POSTEVAL_CANDIDATE=str(CANDIDATE),FCP008_HOST_PYTHON='/home/USER/env_isaaclab/bin/python',
        FCP008_EXECUTION_APPROVAL_SHA256=args.training_approval_sha256,
        FCP018_EXECUTION_OBSERVATION_SHA256=args.observation_sha256,
        FCP008_FORMAL_APPROVAL=str(args.approval.resolve()),FCP008_FORMAL_APPROVAL_SHA256=args.approval_sha256,
        FCP008_POSTEVAL_TOKEN='EXECUTE_APPROVED_FC_P018_POSTEVAL')
    return env


def preflight(args):
    checks=((RUNNER,args.runner_sha256),(CHAIN/'receipt.json',args.chain_receipt_sha256),
            (args.approval,args.approval_sha256),(CANDIDATE/'execution_approval.json',args.training_approval_sha256),
            (REPO/'docs/FC_P018_RUNNING_EXECUTION_20261005.json',args.observation_sha256))
    for path,expected in checks:
        if len(expected) != 64 or sha(path) != expected: raise GuardFailure('execution SHA differs: '+str(path))
    receipt=json.loads((CHAIN/'receipt.json').read_text())
    if receipt.get('status') != 'FC_P018_IMMUTABLE_POSTEVAL_CHAIN_STAGED':
        raise GuardFailure('wrong frozen chain profile')
    for relative,expected in receipt['sha256'].items():
        path=CHAIN/relative
        if not path.resolve().is_relative_to(CHAIN.resolve()) or sha(path) != expected:
            raise GuardFailure('frozen chain bytes differ')
    approval=json.loads(args.approval.read_text())
    if (approval.get('status') != 'FC_P018_FORMAL_EVALUATION_APPROVED'
            or approval.get('formal_evaluation_authorized') is not True
            or approval.get('ppo_auto_launch') is not False):
        raise GuardFailure('formal execution approval differs')
    if discover_owned(): raise GuardFailure('evaluation container already active')
    memory=read_memory()
    if min(memory.values()) < FLOOR_KIB: raise GuardFailure('initial host memory below 20 GiB')
    if args.monitor_dir.resolve().is_relative_to(OUTPUT.resolve()):
        raise GuardFailure('monitor files must not contaminate formal receipt output')
    if not args.monitor_dir.resolve().is_relative_to(CANDIDATE.resolve()):
        raise GuardFailure('monitor directory must be within the approved candidate directory')
    if args.monitor_dir.exists(): raise FileExistsError(args.monitor_dir)


def monitor(process, stream, cancelled, *, clock=time.monotonic, sleep=time.sleep,
            memory_reader=read_memory, discover=discover_owned):
    start=clock()
    while True:
        elapsed=clock()-start
        memory=memory_reader()
        owned=discover()
        row={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':elapsed,
             'mem_available_kib':memory['MemAvailable'],'mem_free_kib':memory['MemFree'],
             'owned_container_ids':owned}
        stream.write(json.dumps(row)+'\n');stream.flush()
        if cancelled(): raise GuardFailure('supervisor termination signal')
        if min(memory.values()) < FLOOR_KIB: raise GuardFailure('host memory below 20 GiB')
        if elapsed >= LIMIT_SECONDS: raise GuardFailure('formal three-hour deadline exceeded')
        code=process.poll()
        if code is not None:
            if owned: raise GuardFailure('child exited leaving an owned container active')
            if code: raise GuardFailure('formal runner exited '+str(code))
            return
        sleep(POLL_SECONDS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runner-sha256','chain-receipt-sha256','approval-sha256','training-approval-sha256','observation-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--monitor-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=('--execute','--resume'),default='--execute')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    preflight(args)
    if not args.execute:
        print('FC_P018_FORMAL_SUPERVISOR_PREFLIGHT_ONLY_NO_LAUNCH');return
    args.monitor_dir.mkdir(parents=True)
    cancelled=[False]
    def stop_signal(*_): cancelled[0]=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    process=None
    outcome={'status':'FC_P018_FORMAL_RESOURCE_GUARD_FAILED','scientific_admission':False}
    error=None
    try:
        with (args.monitor_dir/'runner.log').open('x') as log, (args.monitor_dir/'memory.jsonl').open('x') as memory:
            if cancelled[0]: raise GuardFailure('cancelled before launch')
            process=subprocess.Popen(['bash',str(RUNNER),args.mode],cwd=REPO,env=child_environment(args),
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            monitor(process,memory,lambda:cancelled[0])
        outcome['status']='FC_P018_FORMAL_RESOURCE_GUARD_COMPLETE_NOT_ADMISSION'
        outcome['runner_exit_code']=0
    except BaseException as failure:
        error=failure;outcome['error']=str(failure)
    finally:
        if process is not None:
            try: stop_owned(process)
            except Exception as failure:
                outcome['status']='FC_P018_FORMAL_RESOURCE_GUARD_FAILED'
                outcome['cleanup_error']=str(failure);error=error or failure
        outcome.update(supervisor_sha256=sha(Path(__file__)),runner_sha256=args.runner_sha256,
                       approval_sha256=args.approval_sha256,chain_receipt_sha256=args.chain_receipt_sha256,
                       training_approval_sha256=args.training_approval_sha256,
                       observation_sha256=args.observation_sha256,
                       command=['bash',str(RUNNER),args.mode],
                       execution_environment={k:v for k,v in child_environment(args).items() if k.startswith('FCP')},
                       time_limit_seconds=LIMIT_SECONDS,poll_seconds=POLL_SECONDS,min_memory_gib=20)
        (args.monitor_dir/'receipt.json').write_text(json.dumps(outcome,indent=2)+'\n')
    if error is not None: raise SystemExit(75)


if __name__ == '__main__': main()
