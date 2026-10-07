"""Identity-only I reuse of the reviewed D saved-array arithmetic; no model imports."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

BASE=Path('/tmp/audit_d_dev_arrays.py')
BASE_SHA='2d24af20d625ce2d04abbec7049e6ab32048f271d3ee04c99127ac54e0d3357f'

def gc_terminal(entries,unit,invocation,approval_sha,supervisor,pid_exists):
    assert not pid_exists
    rows=[x for x in entries if x.get('USER_UNIT')==unit and x.get('USER_INVOCATION_ID')==invocation]
    starts=[x for x in rows if x.get('MESSAGE_ID')=='39f53479d3a045ac8e11786248231fbf' and x.get('JOB_RESULT')=='done']
    ends=[x for x in rows if x.get('MESSAGE_ID')=='ae8f7b866b0347b9af31fe1c80b127c0']
    assert len(starts)==len(ends)==1
    assert approval_sha in starts[0]['MESSAGE'] and int(ends[0]['__REALTIME_TIMESTAMP'])>int(starts[0]['__REALTIME_TIMESTAMP'])
    assert not any(any(word in x.get('MESSAGE','').lower() for word in ('failed','main process exited','oom','out of memory')) for x in rows)
    assert supervisor['approval_sha256']==approval_sha and supervisor['error'] is None and supervisor['returncode']==0
    assert supervisor['limits']['memory_max']==12*2**30 and supervisor['limits']['memory_swap_max']==0
    assert Path(supervisor['limits']['path']).name==unit
    return {'mode':'transient_gc_manager_and_supervisor','invocation':invocation,'start_timestamp':starts[0]['__REALTIME_TIMESTAMP'],'completion_timestamp':ends[0]['__REALTIME_TIMESTAMP'],'manager_memory_peak':ends[0].get('MEMORY_PEAK'),'manager_swap_peak':ends[0].get('MEMORY_SWAP_PEAK'),'current_systemctl_limits_unavailable':True}

def adapted_source(text):
    changes={"choices=['D']":"choices=['I']", "'D_rearCl'":"'I_rearCl'", "'D_totalCd'":"'I_totalCd'", "'B/D matched array'":"'B/I matched array'"}
    for old,new in changes.items():
        assert text.count(old)==1,old
        text=text.replace(old,new)
    return text

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--unit',required=True)
    parser.add_argument('--invocation',required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--approval-sha256',required=True)
    args=parser.parse_args()
    assert hashlib.sha256(args.approval.read_bytes()).hexdigest()==args.approval_sha256
    spec=json.loads(args.approval.read_text())
    assert spec['candidate_label']=='I'
    assert spec['unit']==args.unit
    properties=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',args.unit,'-p','LoadState','-p','InvocationID','-p','MainPID','-p','Result','-p','ExecMainStatus'],text=True).splitlines())
    if properties['LoadState']=='not-found' and properties['MainPID']=='0':
        assert args.invocation=='51303b44822849a08c602e559a09739d'
        journal=subprocess.check_output(['journalctl','--user','-u',args.unit,'-o','json','--no-pager'],text=True)
        out=Path('/workspace/fluid_control')/spec['output']
        assert hashlib.sha256((out/'supervisor_result.json').read_bytes()).hexdigest()=='bd20ea7a7e33a3441c0c1a08a034a1029277eeef27a873192afea277f2390aae'
        assert hashlib.sha256((out/'supervisor_child.json').read_bytes()).hexdigest()=='43ef388a546f0ffc8a80aad376986399bd8d0353d8a1f8f3280ca5af3cb2e179'
        child=json.loads((out/'supervisor_child.json').read_text())
        assert child['pid']==822366 and len(child['token'])==64 and all(c in '0123456789abcdef' for c in child['token'])
        assert child['started_monotonic']==505176.432863128
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:cmd=proc.read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            assert str(spec['driver']['path']).encode() not in cmd,'owned evaluator still running'
        lifecycle=gc_terminal([json.loads(x) for x in journal.splitlines()],args.unit,args.invocation,args.approval_sha256,json.loads((out/'supervisor_result.json').read_text()),Path('/proc/822366').exists())
        lifecycle['bound_parent_token_receipt_sha256']='43ef388a546f0ffc8a80aad376986399bd8d0353d8a1f8f3280ca5af3cb2e179'
        lifecycle['owned_evaluator_processes_absent']=True
    else:
        assert properties['InvocationID']==args.invocation and properties['MainPID']=='0' and properties['Result']=='success' and properties['ExecMainStatus']=='0'
        lifecycle={'mode':'retained_unit','properties':properties}
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
    text=adapted_source(BASE.read_text())
    sys.argv=[str(BASE),'--arm','I','--approval',str(args.approval),'--approval-sha256',args.approval_sha256]
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(compile(text,str(BASE),'exec'),{'__name__':'__main__'})
    result=json.loads(stream.getvalue());result['independent_lifecycle']=lifecycle
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
