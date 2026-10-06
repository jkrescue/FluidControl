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

def previous_candidate_cfd_status(root,run=subprocess.check_output):
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
        if state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0':
            result=output/'result.json'
            review=root/'docs/P064_B_PROJECTED_PPO_B01_LONG_CFD_TERMINAL_REVIEW_20261006.md'
            if (hashlib.sha256(result.read_bytes()).hexdigest()!='0be19e0dfdf8df4d60e2f5040673f2a3ec25cbadb133548671ce31f061e75c88'
                or hashlib.sha256(review.read_bytes()).hexdigest()!='de7a3f7fa89d50dad190272028f16ed0ea23214ac52cb23323f42c48df9945d3'):
                raise ValueError('b01 independent terminal binding')
            info.update(status='B策略b01配对800周期已独审完成',physical_pass=True,terminal_verified=True)
            info['note']='b01主(150,210]：减阻3.9275%、升力波动RMS降低18.4272%、均值偏置2.7295%，原标准通过；早首6.2偏置12.7825%仍失败。b00主窗口也通过（3.8953%减阻）；两相位均非新留出测试，未证明显著优于旧policy。无训练/CFD运行，不等于代理准入。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['status']='身份或状态未验证';info['note']=str(exc)
    return info

def candidate_cfd_status(root,run=subprocess.check_output):
    root=Path(root)
    previous=previous_candidate_cfd_status(root,run)
    info={'status':'b07尚未核验启动','cycles':0,'target':800,'invocation':None,
          'scientific_admission':False,'physical_pass':None,'terminal_verified':False,'running':False}
    try:
        approval=root/'docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='df4d7881226f8ebf53da3aa47ac29f32ba2f0c94d59e431dae3a84dced4b7f56':raise ValueError('b07 approval SHA')
        spec=json.loads(approval.read_text())
        output=root/'artifacts/p064_b_projected_ppo_b07_long_cfd_20261006'
        if (root/spec['output']).resolve()!=output.resolve() or spec['candidate_arm']!='B':raise ValueError('b07 output/arm')
        unit='fluid-control-p064-b-projected-ppo-b07-long-cfd-20261006.service'
        raw=run(['systemctl','--user','show',unit,'-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='c45add13aeff41fe9526e835e384a52d':raise ValueError('b07 invocation')
        info['invocation']=state['InvocationID']
        progress=output/'progress.json'
        if progress.exists():
            if progress.stat().st_size>8*2**20:raise ValueError('b07 progress size')
            info.update(cfd_progress_counts(json.loads(progress.read_text()),start=110.))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='同B策略＋镜像对称处理：b07 CPU真实CFD运行中' if info['running'] else 'b07进程已停止，等待独立终态复核'
        info['note']='当前b07与同起点zero配对，CPU策略推理＋CFD，无GPU训练、无在线FNO；计数不代表物理通过。'
        if previous.get('terminal_verified'):
            info['note']+='历史b00/b01已独审主窗通过，减阻分别3.8953%/3.9275%；早期偏置失败仍保留，旧结果不能授予b07通过。'
        if state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0':
            if (hashlib.sha256((output/'result.json').read_bytes()).hexdigest()!='dd579e7443c6693daef4173ed53ea2cb6836878fafff365bc12c1db8fe4ab7fc'
                or hashlib.sha256((root/'docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_TERMINAL_REVIEW_20261006.md').read_bytes()).hexdigest()!='6a76bbb74673dfdfdba57746471a836ce741f4855206efb5605677934133e953'):
                raise ValueError('b07 independent terminal binding')
            info.update(status='同B策略b07真实CFD已独审完成',physical_pass=True,terminal_verified=True)
            info['note']='b07主(130,190]减阻3.9027%、升力RMS降低18.5005%、均值偏置1.2921%；六个固定窗口均过原标准，早6.2偏置8.3039%也低于10%。b00/b01历史主窗约3.9%通过、早期失败仍保留。b07不是全新holdout；三相位非统计独立，完整代理精度仍FAIL。无当前训练/CFD运行。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['note']=str(exc)
    return info

FORMAL_TERMINAL_BINDINGS={
    'receipt':'30d3d0746580b8423a9f626a0ebe1b76c129acab7800158f74d7d8df3d8a1799',
    'review':'62a6234ed0e08ab70532a5252f34e6c8b203a22c6c6abeea5a67ba2635fc8cd6'}

def verified_formal_terminal(root):
    root=Path(root)
    receipt=root/'artifacts/fcp064_arm_b_formal_resume_r3_20261006/receipt.json'
    review=root/'docs/P064_B_FORMAL_TERMINAL_REVIEW_20261006.md'
    for key,path in [('receipt',receipt),('review',review)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=FORMAL_TERMINAL_BINDINGS[key]:
            raise ValueError('formal terminal '+key+' SHA')
    data=json.loads(receipt.read_text())
    if data['scientific_admission'] is not False:raise ValueError('unexpected admission')
    return {'terminal_verified':True,'scientific_pass':False,'running':False,
            'receipt_sha256':FORMAL_TERMINAL_BINDINGS['receipt'],'review_sha256':FORMAL_TERMINAL_BINDINGS['review'],
            'cached_comparison_result_sha256':'7158d4e7f977a4fa126c9293d85cad350ce1799b0685d691f02225834e550460',
            'cached_comparison_review_sha256':'c7433e7b7bc364ded006d9ae17d1f1afea9465ce461385b57b926e60dc8ba1b7',
            'status':'完整预测精度评估R3计算已完成／精度要求未全满足',
            'note':'独审：力窗口2/6通过，4条旋转分支升力波动预测仍失败；原门槛不变。b00/b01真实CFD主窗约3.9%减阻已通过，不能代替模型精度验收。same6缓存H1/AR比较已完成，但缺少有符号H1序列且有batch/precision差异；signed H1 batch1诊断准备中、未运行。当前无GPU训练。实际资源见实时监控；.06仅启动核算，原评估实际allocator .15。'}

def signed_h1_status(root,run=subprocess.check_output):
    info={'running':False,'training':False,'invocation':None,'completed_endpoints':None,
          'status':'signed H1 batch1诊断准备中，尚未核验启动'}
    try:
        approval=Path(root)/'docs/P064_TEACHER_FORCED_H1_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='482628b32be9fcd3879404356e3deeb186c38a96bf59635b76893e8d10d3e0c8':raise ValueError('H1 approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-teacher-forced-h1-20261006.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='76d21e62134b44c0a97d65b6ad991669':raise ValueError('H1 invocation')
        info['invocation']=state['InvocationID']
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='signed H1 batch1 GPU预测诊断进程运行中（非训练，无逐步进度）' if info['running'] else 'signed H1诊断进程已停止，等待独立终态复核'
        if state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0':
            paths={
                'artifacts/p064_teacher_forced_h1_signed_force_20261006/result.json':'1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef',
                'docs/P064_TEACHER_FORCED_H1_TERMINAL_REVIEW_20261006.md':'2141cc0f060e79acf57ad68c530038e1814fb228ba45be2fa4a8ac01da956b36'}
            for path,digest in paths.items():
                if hashlib.sha256((Path(root)/path).read_bytes()).hexdigest()!=digest:raise ValueError('H1 terminal binding')
            info.update(terminal_verified=True,completed_endpoints=600,
                        status='signed H1预测诊断已独审完成600端点（非训练）：真实输入仍有力误差，不能仅归因长AR；无新准入')
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['verification_note']=str(exc)
    return info

def scale_window_count(text):
    count=0
    for line in text.splitlines():
        try:row=json.loads(line)
        except (ValueError,TypeError):continue
        if isinstance(row,dict) and row.get('event')=='window_complete' and row.get('mode')=='scales':
            if row.get('split')!='train':raise ValueError('scales unexpected split')
            count+=1
    if count>1368:raise ValueError('scales excess count')
    return count

SCALES_TERMINAL_BINDINGS={'result':'4b8d28506bad47d76753848094c1b00195f96bb432cb9d96ef9031c8fdad2a8b','review':'ec796b416e956b2ea84c74517805af77c98462e1f48c96fefb94d6c963e60805'}

def verified_scales_terminal(root):
    root=Path(root)
    paths={'result':root/'artifacts/fcp064_b_h25_scales_20261006_r2/payload/result.json','review':root/'docs/P064_B_H25_SCALES_R2_TERMINAL_REVIEW_20261006.md'}
    for key,path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SCALES_TERMINAL_BINDINGS[key]:raise ValueError('scales terminal '+key+' SHA')
    result=json.loads(paths['result'].read_text())
    if result['training_windows']!=1368 or result['optimizer_steps']!=0 or result['model_saved'] is not False:raise ValueError('scales terminal counts')
    return {'terminal_verified':True,'fixed_scales':result['fixed_scales'],'status':'H25损失尺度计算已完成并独审（非训练）'}

def h25_scales_status(root,run=subprocess.check_output):
    root=Path(root);info={'running':False,'training':False,'optimizer_steps':0,'windows':0,'target':1368,'invocation':None,'status':'H25损失尺度计算尚未核验启动'}
    try:
        path=root/'docs/P064_B_H25_SCALES_R2_APPROVAL_20261006.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!='cb60cf7bcc146a51f085957d6c4c68a7792fe41803fd925fed1aa7c1b3c1d77e':raise ValueError('scales R2 approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-h25-scales-r2-20261006.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='88e5e31e611c45dab28dbe6c3ad11c8c':raise ValueError('scales R2 invocation')
        info['invocation']=state['InvocationID']
        log=root/'artifacts/fcp064_b_h25_scales_20261006_r2/run.log'
        if log.exists():
            if log.stat().st_size>8*2**20:raise ValueError('scales log size')
            info['windows']=scale_window_count(log.read_text())
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='H25准备：B父模型H10损失尺度计算运行中' if info['running'] else '损失尺度计算进程已停止，等待终态独审'
        info['note']=f"{info['windows']}/1368训练窗口；0优化器更新，不是模型训练。R1因loader未识别B kind退出，保留失败；R2仅修复来源绑定。日志计数不等于成功。"
        if not info['running'] and state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0':
            info.update(verified_scales_terminal(root))
            info['note']='1368/1368窗口原始均值已复算；0优化器更新、未保存模型。field尺度0.00145615，force尺度0.00378382；仅供后续单独批准的资源探针。R1 loader身份故障保留。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def h25_training_counts(text):
    windows=0;updates=[]
    for line in text.splitlines():
        try:row=json.loads(line)
        except (ValueError,TypeError):continue
        if row.get('mode')!='train':continue
        if row.get('event')=='window_complete':
            if row.get('split')!='train':raise ValueError('H25 unexpected split')
            windows+=1
        elif row.get('event')=='group_complete':updates.append(row['group'])
    if windows>256 or updates!=list(range(1,len(updates)+1)) or len(updates)>32 or windows<len(updates)*8:raise ValueError('H25 training count sequence')
    return windows,len(updates)

def h25_training_status(root,run=subprocess.check_output):
    root=Path(root);info={'running':False,'training':False,'windows':0,'updates':0,'target_windows':256,'target_updates':32,'invocation':None,'status':'H25训练尚未核验启动'}
    try:
        approval=root/'docs/P064_B_H25_TRAINING_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='922df0ce388cfb4ac2ed0f65156ce049629f0c7e991910b4ddd633ea926b61b4':raise ValueError('H25 train approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-h25-training-20261006.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='ebea7981d70745889f9a3249977bea16':raise ValueError('H25 train invocation')
        info['invocation']=state['InvocationID']
        log=root/'artifacts/fcp064_b_h25_training_20261006/run.log'
        if log.exists():
            if log.stat().st_size>8*2**20:raise ValueError('H25 train log size')
            info['windows'],info['updates']=h25_training_counts(log.read_text())
            info['last_log_update_unix']=log.stat().st_mtime
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['training']=info['running']
        if info['running']:info['status']='H25流场微调运行中（气动力网络冻结）'
        elif state.get('ExecMainStatus')!='0':info['status']='H25训练工程失败，未完成参数更新'
        else:info['status']='H25训练进程退出，等待终态独审'
        info['note']=f"实际窗口{info['windows']}/256，完成更新{info['updates']}/32；日志计数不是训练成功。尺度1368窗及单步资源探针已完成。首次训练在首窗后被旧H10累积器的horizon检查拒绝，失败记录保留；并非已证NaN。完整预测精度FAIL及三相位真实CFD物理成功结论不变。"
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def formal_evaluation_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'完整预测精度评估身份/状态未验证','invocation':None,'training':False,'scientific_pass':None}
    try:
        approval=root/'docs/FC_P064_ARM_B_FORMAL_RESUME_R3_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='19be2d6aad903ffc94b807803bd5fd0902c7ec5b7a0f0b4212423744db89cb56':raise ValueError('formal approval SHA')
        unit='fluid-control-p064-b-formal-r3-20261006.service'
        raw=run(['systemctl','--user','show',unit,'-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='3bded2dcb4a24f808879987962a9ef8b':raise ValueError('formal invocation')
        info['invocation']=state['InvocationID']
        if state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0:
            info['status']='完整预测精度评估R3续跑中（非训练，尚无完整科学结论）'
        elif state.get('MainPID')=='0':
            info['status']='完整预测精度评估R3退出，等待各门槛独审' if state.get('ExecMainStatus')=='0' else '完整预测精度评估R3失败/停止，证据保留'
        info['note']='精确复用R2 precision与validation10两阶段，仅运行其余6阶段；保留R1/R2失败与来源，未重算validation10。实际torch allocator .15，外层.06仅启动核算；原H100失败与科学门槛不变，exit0不等于通过。'
        if state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0':
            info.update(verified_formal_terminal(root))
            diagnostic=signed_h1_status(root,run)
            info['signed_h1']=diagnostic
            info['note']=info['note'].replace('signed H1 batch1诊断准备中、未运行。',diagnostic['status']+'。')
            scales=h25_scales_status(root,run)
            info['h25_scales']=scales
            info['note']+=' '+scales['status']+'。'+scales.get('note','')
            training=h25_training_status(root,run)
            info['h25_training']=training
            info['note']=training['status']+'。'+training.get('note','')+' '+info['note']
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
