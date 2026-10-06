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

def last_json_row(path):
    """Bounded tail of complete JSONL rows; incomplete current write is ignored."""
    if not Path(path).exists():return None
    with Path(path).open('rb') as f:
        f.seek(0,2);size=f.tell();f.seek(max(0,size-65536));raw=f.read()
    lines=raw.splitlines(keepends=True)
    for line in reversed(lines):
        if not line.endswith(b'\n'):continue
        try:
            value=json.loads(line)
            if isinstance(value,dict):return value
        except (ValueError,UnicodeDecodeError):continue
    return None

def candidate_ppo_status(root,run=subprocess.check_output):
    root=Path(root);info={'status':'身份或状态未验证','timesteps':0,'target':32768,
                         'reported_ppo_epochs':None,'optimizer_steps':None,'invocation':None}
    unit='fluid-control-p064-b-ppo-32768-20261006.service'
    invocation='f613395cbf1140549dc60e7b046e0f6b'
    try:
        approval=root/'docs/P064_B_PPO_APPROVAL_20261006.json'
        digest='ae327fee310bad562aceef35029595d20c9a3421d5d5be3dc3b68bb82649e9fe'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!=digest:raise ValueError('PPO approval SHA')
        spec=json.loads(approval.read_text())
        output=root/'artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload'
        if Path(spec['output']).resolve()!=output.resolve():raise ValueError('PPO output identity')
        raw=run(['systemctl','--user','show',unit,'-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!=invocation:raise ValueError('PPO invocation differs')
        info['invocation']=invocation
        if (output/'source_spec.json').exists() and json.loads((output/'source_spec.json').read_text())!=spec:raise ValueError('PPO executed spec differs')
        row=last_json_row(output/'transitions.jsonl')
        if row:
            n=row['num_timesteps']
            if type(n) is not int or not 0<=n<=32768:raise ValueError('PPO actual transition count')
            info['timesteps']=n
            info['last_event_file_utc']=datetime.datetime.fromtimestamp((output/'transitions.jsonl').stat().st_mtime,datetime.timezone.utc).isoformat()
        log=last_json_row(output/'progress.json')
        if log and 'train/n_updates' in log:info['reported_ppo_epochs']=log['train/n_updates']
        if state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0:
            info['status']='P064 B 候选代理上的 PPO 训练中'
        elif state.get('MainPID')=='0':
            info['status']='PPO程序退出0，等待独立终态审查' if state.get('ExecMainStatus')=='0' else 'PPO失败/停止（未自动重试）'
            if state.get('ExecMainStatus')=='0' and (output/'result.json').exists():
                info.update(producer_terminal_counts(json.loads((output/'result.json').read_text()),spec))
                report=root/'docs/P064_B_PPO_TERMINAL_REVIEW_20261006.md'
                if hashlib.sha256(report.read_bytes()).hexdigest()!='0cc1494286f85b930b43b1a11713ae6ac3719d8ff599cfa0780a8d9fe71b0d65':raise ValueError('PPO independent report SHA')
                info.update(status='PPO训练完成，独立工程审查通过（非物理验收）',terminal_verified=True,
                            counts_scope='实际终态与独立保存证据复核',review=str(report.relative_to(root)))
        else:info['status']='PPO状态待确认'
        info['note']='计数来自真实transition日志；训练步数不等于优化器更新，启动不代表更新成功。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def producer_terminal_counts(result,spec):
    """Producer telemetry only; independent-review status must remain separate."""
    if (result.get('status')!='P064_CANDIDATE_DIVERSE_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION'
        or result.get('candidate_arm')!=spec['candidate_arm']
        or result.get('candidate_manifest_sha256')!=spec['candidate_manifest_sha256']
        or result.get('timesteps')!=32768 or result.get('ppo_n_updates')!=256
        or result.get('fno_tensors_unchanged') is not True
        or [row.get('optimizer_step') for row in result.get('optimizer_steps',[])]!=list(range(1,513))):
        raise ValueError('PPO producer terminal contract differs')
    return {'timesteps':32768,'reported_ppo_epochs':256,'optimizer_steps':512,
            'counts_scope':'实际终态producer日志；仍待独立审查','terminal_verified':False}

def cfd_progress_counts(progress,start=148.):
    n=progress.get('completed_cycles');rows=progress.get('rows',[])
    if type(n) is not int or not 0<=n<=800 or len(rows)!=n:raise ValueError('CFD progress count')
    if rows and (rows[-1].get('step')!=n or abs(rows[-1]['end_time']-(start+.1*n))>1e-8):raise ValueError('CFD endpoint clock')
    return {'cycles':n,'target':800,'current_time':start if not rows else rows[-1]['end_time'],
            'applied_omega':None if not rows else rows[-1]['applied_omega']}

def candidate_cfd_status(root,run=subprocess.check_output):
    root=Path(root);info={'status':'身份或状态未验证','cycles':0,'target':800,'invocation':None,
                         'scientific_admission':False,'physical_pass':None}
    unit='fluid-control-p064-b-projected-ppo-b01-long-cfd-20261006.service'
    invocation='432c12de32b0444d9a6f6626e12616d1'
    try:
        approval=root/'docs/P064_B_PROJECTED_PPO_B01_LONG_CFD_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='3a19e326ebfd37df24060b8b5717b5af08b405ed63165e4034974aeb7b0abcb6':raise ValueError('CFD approval SHA')
        spec=json.loads(approval.read_text())
        output=root/'artifacts/p064_b_projected_ppo_b01_long_cfd_20261006'
        if (root/spec['output']).resolve()!=output.resolve() or spec['candidate_arm']!='B':raise ValueError('CFD output/arm')
        raw=run(['systemctl','--user','show',unit,'-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!=invocation:raise ValueError('CFD invocation')
        info['invocation']=invocation
        progress=output/'progress.json'
        if progress.exists():
            if progress.stat().st_size>8*2**20:raise ValueError('CFD progress size')
            info.update(cfd_progress_counts(json.loads(progress.read_text()),start=130.))
        if state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0:
            info['status']='同B策略＋镜像对称处理：b01第二相位CPU真实CFD运行中'
        elif state.get('MainPID')=='0':
            info['status']='CFD程序退出0，等待独立原始力复核' if state.get('ExecMainStatus')=='0' else 'CFD失败/停止（未自动重试）'
        else:info['status']='CFD状态待确认'
        report=root/'docs/P064_B_PROJECTED_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md'
        if hashlib.sha256(report.read_bytes()).hexdigest()!='7b453d9c529d9d5c52988608c89d61050204510d41549cbb2be620fcdcebbe04':raise ValueError('b00 review SHA')
        info['note']='b00已独审：减阻3.8953%、RMS降低18.44%、均值偏置1.1378%，原主窗口标准通过；属于训练内确认，非显著优于旧policy。当前b01与同起点zero配对，尚无结果；无GPU训练/FNO在线调用。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['status']='身份或状态未验证';info['note']=str(exc)
    return info

def formal_evaluation_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'正式评估身份/状态未验证','invocation':None,'training':False,'scientific_pass':None}
    try:
        approval=root/'docs/FC_P064_ARM_B_FORMAL_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='cd58fd47e991ec6dac200bd82d414347f72b778ea78415377427e430dfd47478':raise ValueError('formal approval SHA')
        unit='fluid-control-p064-b-formal-r2-20261006.service'
        raw=run(['systemctl','--user','show',unit,'-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='01806bdc150841f7b9efd04360a441f2':raise ValueError('formal invocation')
        info['invocation']=state['InvocationID']
        if state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0:
            info['status']='原完整formal评估进程运行中（非训练，尚不代表GPU forward或通过）'
        elif state.get('MainPID')=='0':
            info['status']='formal进程退出，等待各科学门槛独审' if state.get('ExecMainStatus')=='0' else 'formal进程失败/停止，保留证据'
        info['note']='R1路径错误保留；R2实际torch allocator为.15，外层.06仅启动核算，非强制上限。Lead批准同任务继续，72GiB/noSwap与Available22GiB保护不变。非训练；原H100失败与科学门槛保持，exit0不等于通过。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['note']=str(exc)
    return info

def development_summary(root):
    """Only small independently reviewed JSON; never load checkpoint/field arrays."""
    root=Path(root)
    bindings={
        'K1':('artifacts/p064_k1_development_h1_h5_20261006/result.json','9ea3e0e781e76265bbc65ea52d6fec92ebe5b5cb7a93d91c3c9addb454f7c4de'),
        'A':('artifacts/p064_arm_a_development_h1_h5_20261006/result.json','c8b0242658a101120603514e6d2e5076c827c518965c92470810fe9693840fe6'),
        'B':('artifacts/p064_arm_b_development_h1_h5_20261006/result.json','47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665')}
    try:
        report=root/'docs/P064_AB_DEVELOPMENT_COMPARISON_REVIEW_20261006.md'
        if hashlib.sha256(report.read_bytes()).hexdigest()!='2fa7e5d71e4b163bfff2261ec2b38ef43c1c5aa75acc7cea7d47c1f3296c225a':raise ValueError('comparison report SHA')
        rows=[]
        for label,(name,digest) in bindings.items():
            path=root/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('comparison result SHA')
            summary=json.loads(path.read_text())['summary']
            rows.append({'label':label, 'h1_cl':summary['1']['force_channel_mae'][3],
                'h1_cd':summary['1']['total_drag_mae'], 'h5_cl':summary['5']['force_channel_mae'][3],
                'h5_cd':summary['5']['total_drag_mae']})
        return {'verified':True,'rows':rows,'report':str(report.relative_to(root)),
                'scope':'已打开开发集；绝对力系数MAE，非百分比；非正式验收',
                'current_stage':'新 PPO 运行包检查中，尚未训练',
                'field_note':'流场冻结、三者预测相同：H1/H5速度相对L2 1.03%/4.30%，压力3.20%/13.70%，未改善。H100失败保留。'}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {'verified':False,'error':str(exc)}

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
            review=registration.get('terminal_review')
            if state.get('ExecMainStatus')=='0' and review:
                path=Path(root)/review['path']
                if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():raise ValueError('review path')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=review['sha256']:raise ValueError('review SHA')
                receipt=json.loads(path.read_text())
                if receipt['unit']['InvocationID']!=invocation or receipt['records']!=32 or receipt['consumed']!=256:raise ValueError('review identity/count')
                if receipt['status']!='P064_TERMINAL_ENGINEERING_REVIEW_NOT_ADMISSION':raise ValueError('review scope')
                info['status']='训练完成，独立工程检查通过（非精度验收）'
                info['terminal_verified']=True
        else:info['status']='状态待确认'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['status']='身份或状态未验证';info['note']=str(exc)
    return info
