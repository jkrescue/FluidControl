"""Small read-only P064 stage card; actual invocation before any running claim."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

PROFILES={arm:{'unit':f'fluid-control-fcp064-aero-arm-{arm.lower()}-20261006.service',
              'output':f'artifacts/fcp064_controlled_aero_arm_{arm.lower()}_20261006'} for arm in ('A','B')}
PROFILES['A']['unit']='fluid-control-fcp064-aero-arm-a-r2-20261006.service'
LABELS={'A':'A 原数据对照','B':'B 加入真实闭环数据'}

def parse_journal(text):
    consumed=updates=0;last=None
    for line in text.splitlines():
        try:
            record=json.loads(line);message=json.loads(record.get('MESSAGE',''))
        except (ValueError,TypeError):continue
        if not isinstance(message,dict):continue
        event=message.get('event')
        if event not in ('training_window_complete','accumulation_update_complete'):continue
        if type(message.get('history_k')) is not int or message['history_k']!=1:
            raise ValueError('history contract')
        if event=='training_window_complete':
            value=message.get('consumed')
            if type(value) is not int or not consumed<=value<=256:raise ValueError('window count')
            consumed=value
        else:
            value=message.get('update')
            if type(value) is not int or not updates<=value<=32:raise ValueError('update count')
            updates=value
        timestamp=record.get('__REALTIME_TIMESTAMP')
        if isinstance(timestamp,str) and timestamp.isdecimal():
            last=datetime.datetime.fromtimestamp(int(timestamp)/1e6,datetime.timezone.utc).isoformat()
    return {'windows':consumed,'updates':updates,'last_update_utc':last}

def status(root, arm, registration=None, run=subprocess.check_output):
    info={'arm':arm,'label':LABELS[arm],'status':'尚未启动','windows':0,'updates':0,
          'last_update_utc':None,'invocation':None,'terminal_verified':False,
          'note':'顺序执行；计数不是模型或科学验收。'}
    if registration is None:return info
    info['previous_attempt_note']=registration.get('previous_attempt_note','')
    try:
        expected=PROFILES[arm]
        if registration['unit']!=expected['unit'] or registration['output']!=expected['output']:
            raise ValueError('profile identity')
        for key in ('approval','driver'):
            item=registration[key];path=Path(root)/item['path']
            if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():raise ValueError('bound path')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('bound SHA')
        invocation=registration['invocation']
        if not isinstance(invocation,str) or len(invocation)!=32 or any(c not in '0123456789abcdef' for c in invocation):raise ValueError('invocation')
        raw=run(['systemctl','--user','show',expected['unit'],'-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!=invocation:raise ValueError('actual invocation mismatch')
        info['invocation']=invocation
        journal=run(['journalctl','--user',f'_SYSTEMD_INVOCATION_ID={invocation}','-o','json','--no-pager','-n','2000'],text=True,timeout=3)
        info.update(parse_journal(journal))
        if state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0:
            info['status']='训练中'
        elif state.get('MainPID')=='0' and state.get('SubState') in ('exited','dead','failed'):
            info['status']='程序退出0，等待独立终态审查' if state.get('ExecMainStatus')=='0' else '失败/停止（未自动重试）'
        else:info['status']='状态待确认'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['status']='身份或状态未验证';info['note']=str(exc)
    return info
