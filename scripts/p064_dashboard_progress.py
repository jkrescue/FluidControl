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

def previous_candidate_ppo_status(root,run=subprocess.check_output):
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

def second_seed_ppo_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'身份或状态未验证','timesteps':0,'target':32768,'running':False,'training':False,'reported_ppo_epochs':None,'optimizer_steps':None,'invocation':None}
    try:
        approval=root/'docs/P064_B_SEED20261007_PPO_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='2a8b0bca02154ed5bf0e7b35035e01f0695c1763b183063251ee619db4fc6604':raise ValueError('seed approval SHA')
        spec=json.loads(approval.read_text())
        output=root/'artifacts/p064_b_seed20261007_diverse_h5_32768_ppo_20261006/payload'
        if Path(spec['output']).resolve()!=output.resolve() or spec['protocol']['seed']!=20261007:raise ValueError('seed output/protocol')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-seed20261007-ppo-32768-20261006.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='38013204a3c94b7fb14da04b78b5fb83':raise ValueError('seed invocation')
        info['invocation']=state['InvocationID']
        if (output/'source_spec.json').exists() and json.loads((output/'source_spec.json').read_text())!=spec:raise ValueError('seed executed spec')
        row=last_json_row(output/'transitions.jsonl')
        if row:
            n=row['num_timesteps']
            if type(n) is not int or not 0<=n<=32768:raise ValueError('seed transition count')
            info['timesteps']=n
            info['last_event_file_utc']=datetime.datetime.fromtimestamp((output/'transitions.jsonl').stat().st_mtime,datetime.timezone.utc).isoformat()
        active=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['running']=active;info['training']=active and info['timesteps']>0
        info['status']=('第二seed20261007：实际PPO训练中' if info['training'] else '第二seed程序已启动，尚未观测训练transition') if active else '第二seed训练进程已停止，等待独立终态审查'
        if state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0' and (output/'result.json').exists():
            info.update(producer_terminal_counts(json.loads((output/'result.json').read_text()),spec))
            review=root/'docs/P064_B_SEED20261007_PPO_TERMINAL_REVIEW_20261006.md'
            if hashlib.sha256((output/'result.json').read_bytes()).hexdigest()!='47dc970756bd608c8a1c3f4744c5b44a1ee87f3524737ea42c4f0fbdf9f82afd' or hashlib.sha256(review.read_bytes()).hexdigest()!='10f0689cd3bb0200542999f94dc05d35c2444904e589154aae8685e6d6aa1d4d':raise ValueError('seed terminal review binding')
            info.update(status='第二seed PPO训练完成，独立工程核验通过',terminal_verified=True,training=False,running=False,counts_scope='实际32768步/256epochs/512optimizer hooks独审')
        info['note']='只变seed，冻结B/H5/24真实起点/32768预算不变；计数来自真实transition，不等于optimizer更新。旧B真实闭环减阻约3.90%、升力波动降低18.4%，初始权重无减阻收益；完整预测精度FAIL另列。新策略CFD物理验证见实时CFD卡，无reward选checkpoint/seed扫描。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def canonical_seed20261007_ppo_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'身份或状态未验证','timesteps':0,'target':32768,'running':False,'training':False,'reported_ppo_epochs':None,'optimizer_steps':None,'invocation':None}
    try:
        approval=root/'docs/P064_B_SYMMETRY_CANONICAL_PPO_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='1a49d9a72a8dabcb1159b2d04ebcf19f039cbdb70f87ba1e1c4d714d898368c7':raise ValueError('canonical PPO approval SHA')
        spec=json.loads(approval.read_text())
        output=root/'artifacts/p064_b_symmetry_canonical_h5_32768_ppo_20261007/payload'
        if Path(spec['output']).resolve()!=output.resolve() or spec['protocol']['seed']!=20261007:raise ValueError('canonical output/protocol')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-symmetry-canonical-ppo-20261007.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='029124675a0d459e82455c59f831ba3b':raise ValueError('canonical PPO invocation')
        info['invocation']=state['InvocationID']
        if (output/'source_spec.json').exists() and json.loads((output/'source_spec.json').read_text())!=spec:raise ValueError('canonical executed spec')
        row=last_json_row(output/'transitions.jsonl')
        if row:
            n=row['num_timesteps']
            if type(n) is not int or not 0<=n<=32768:raise ValueError('canonical transition count')
            info['timesteps']=n
            info['last_event_file_utc']=datetime.datetime.fromtimestamp((output/'transitions.jsonl').stat().st_mtime,datetime.timezone.utc).isoformat()
        active=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['running']=active;info['training']=active and info['timesteps']>0
        info['status']=('对称坐标PPO：实际训练中' if info['training'] else '对称坐标PPO已启动，尚未观测transition') if active else '对称坐标PPO已停止，等待独立终态审查'
        if state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0' and (output/'result.json').exists():
            info.update(producer_terminal_counts(json.loads((output/'result.json').read_text()),spec,expected_status='P064_B_SYMMETRY_CANONICAL_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION'))
            review=root/'docs/P064_B_SYMMETRY_CANONICAL_PPO_TERMINAL_REVIEW_20261007.md'
            if hashlib.sha256((output/'result.json').read_bytes()).hexdigest()!='44ef56114e17db88077d5f3e7920a2a9031023be74e31d359bd8c66be7441d1e' or hashlib.sha256(review.read_bytes()).hexdigest()!='72a03435e0ed954743694ca785e8de1616c6860db457b2ad20ef6f40d330e562':raise ValueError('canonical terminal binding')
            info.update(status='对称坐标PPO训练完成，方向/动作映射独审通过',terminal_verified=True,training=False,running=False,counts_scope='32768步/256epochs/512参数更新；物理效果另验')
        info['note']='同B模型/seed20261007/32768步/H5/24起点，只改固定对称坐标处理；标准PPO优化不变。实际计数不是物理成功；此前第二seed增阻0.62%和旧seed减阻3.90%均保留，预测精度FAIL未消失。新策略物理验证见当前CFD卡。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def candidate_ppo_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'身份或状态未验证','timesteps':0,'target':32768,'running':False,'training':False,'reported_ppo_epochs':None,'optimizer_steps':None,'invocation':None}
    try:
        approval=root/'docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='d62cccbcf1d896ed812cca7cac27e8e5df9d1fd7623a84ae1b5478494cabdd39':raise ValueError('canonical seed20261006 approval SHA')
        spec=json.loads(approval.read_text());output=root/'artifacts/p064_b_symmetry_canonical_seed20261006_h5_32768_ppo_20261007/payload'
        if Path(spec['output']).resolve()!=output.resolve() or spec['protocol']['seed']!=20261006:raise ValueError('canonical replication output/seed')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-symmetry-canonical-seed20261006-ppo-20261007.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='e236b09e33564b0bb4e6aad5c46eff60':raise ValueError('canonical replication invocation')
        info['invocation']=state['InvocationID']
        if (output/'source_spec.json').exists() and json.loads((output/'source_spec.json').read_text())!=spec:raise ValueError('canonical replication executed spec')
        row=last_json_row(output/'transitions.jsonl')
        if row:
            n=row['num_timesteps']
            if type(n) is not int or not 0<=n<=32768:raise ValueError('canonical replication transition count')
            info['timesteps']=n
            info['last_event_file_utc']=datetime.datetime.fromtimestamp((output/'transitions.jsonl').stat().st_mtime,datetime.timezone.utc).isoformat()
        active=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['running']=active;info['training']=active and info['timesteps']>0
        info['status']=('固定seed20261006：对称坐标PPO实际训练中' if info['training'] else '训练程序启动中，尚未观测transition') if active else '训练进程已停止，等待独立终态核验'
        info['note']='FC-E084仅改既定seed，冻结B/对称坐标/H5/24起点/32768预算与奖励不变；不扫seed、不按reward选checkpoint。当前计数为真实环境步，不等于参数更新或物理成功。前次canonical b00主窗减阻3.96%，早期偏置失败保留；并行b01用前次已冻结策略。完整预测FAIL不变。'
        if not active and state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0' and (output/'result.json').exists():
            review=root/'docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_TERMINAL_REVIEW_20261007.md'
            if hashlib.sha256((output/'result.json').read_bytes()).hexdigest()!='cb1a2f0931fcf70c68802287bdb5e5f894553c0eab271a030a54ff7f2f598336' or hashlib.sha256(review.read_bytes()).hexdigest()!='d2bf82ce004e6d07213d50d930d0d58b4bafa009c44d87972ca0bdb8081aa707':raise ValueError('canonical replication terminal binding')
            info.update(producer_terminal_counts(json.loads((output/'result.json').read_text()),spec,expected_status='P064_B_SYMMETRY_CANONICAL_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION'))
            info.update(status='固定seed20261006 PPO训练完成，独立工程核验通过',training=False,running=False,terminal_verified=True)
            info['counts_scope']='实际32768步/256epochs/512参数更新已独立终态核验；不是物理指标。'
            info['note']='实际32768步/256epochs/512参数更新；方向动作映射、冻结FNO与资源独审通过。物理验证状态见真实CFD卡；训练完成不等于预测精度PASS。'
            physical_review=root/'docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_CFD_TERMINAL_REVIEW_20261007.md'
            if physical_review.is_file() and hashlib.sha256(physical_review.read_bytes()).hexdigest()=='24e9a5ad02b0b36b34910ebe32e72de4c8cc5692f6090e4efa0dcad8439ac7fc':
                info['note']+='该固定策略E086 b00已独审：主窗减阻3.7440%、后升力RMS降低22.6210%、偏置3.6741%，主窗通过；早期12.4/首6.2偏置11.2623%/21.1797%未过原10%。完整预测FAIL保留。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def producer_terminal_counts(result,spec,expected_status='P064_CANDIDATE_DIVERSE_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION'):
    """Producer telemetry only; independent-review status must remain separate."""
    if (result.get('status')!=expected_status
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

def trained_candidate_cfd_status(root,run=subprocess.check_output):
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

INITIAL_TERMINAL_BINDINGS={'result':'48b2b37ddccb6ab4bbf7f04ec6a51c5c071a9b7ed4de504afe0d725af2cebea1','review':'8e66c497acfcc3483d760a77469330028bd12f9cd233ec36d445d15557c34d34'}

def verified_initial_terminal(root):
    root=Path(root)
    result=root/'artifacts/p064_initial_projected_ppo_long_cfd_20261006/result.json'
    review=root/'docs/P064_INITIAL_POLICY_CFD_TERMINAL_REVIEW_20261006.md'
    for key,path in [('result',result),('review',review)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=INITIAL_TERMINAL_BINDINGS[key]:raise ValueError('initial terminal '+key+' SHA')
    data=json.loads(result.read_text())
    if data['cycles']!=800 or not data['owned_containers_cleaned']:raise ValueError('initial terminal incomplete')
    return {'status':'真实闭环有效；初始权重对照完成，学习后策略贡献已验证','cycles':800,'target':800,'current_time':228.,'running':False,'training':False,'physical_pass':False,'terminal_verified':True,'invocation':'dbc0e8f47f994e7280694e9ed6714c56','note':'同zero全部原始力匹配：初始权重主窗减阻−0.00756%，训练后B +3.89528%；同投影/限幅下学习后权重有贡献。不是RL单独归因或跨控制器最优证明。训练后早期偏置失败保留；动作平方成本不等于物理能耗。H25新模型未采用，预测精度仍待改善。本CFD对照已完成；第二seed复验状态见PPO实时卡。'}

def initial_candidate_cfd_status(root,run=subprocess.check_output):
    root=Path(root)
    trained=trained_candidate_cfd_status(root,run)
    info={'status':'初始权重CFD对照尚未核验启动','cycles':0,'target':800,'running':False,'training':False,'physical_pass':None,'terminal_verified':False,'invocation':None,'trained_reference':trained}
    try:
        approval=root/'docs/P064_INITIAL_PROJECTED_PPO_LONG_CFD_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='f4e35a92a227b29fcf216018f09b3d320b382ffc392d9aad7e73616dc32c3797':raise ValueError('initial policy CFD approval SHA')
        if (root/'docs/P064_INITIAL_POLICY_CFD_TERMINAL_REVIEW_20261006.md').exists():
            info.update(verified_initial_terminal(root))
            return info
        raw=run(['systemctl','--user','show','fluid-control-p064-initial-projected-ppo-long-cfd-20261006.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='dbc0e8f47f994e7280694e9ed6714c56':raise ValueError('initial policy CFD invocation')
        info['invocation']=state['InvocationID']
        progress=root/'artifacts/p064_initial_projected_ppo_long_cfd_20261006/progress.json'
        if progress.exists():
            if progress.stat().st_size>8*2**20:raise ValueError('initial CFD progress size')
            info.update(cfd_progress_counts(json.loads(progress.read_text()),start=148.))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='初始权重＋同投影/限幅：b00真实CFD对照进行中' if info['running'] else '初始权重CFD进程已停止，等待独立原始力复核'
        if not info['running'] and state.get('ExecMainStatus')!='0':info['status']='初始权重CFD失败/停止，保留证据'
        info['note']='CPU策略推理与真实CFD反馈，不是PPO训练、无在线FNO/MPC。只替换为同seed精确初始权重，归一化/投影/单次限幅/800周期不变；尚无对照收益结论，不凭当前动作预判。历史已训练B在b00/b01/b07主窗有效（b07减阻3.90%、波动降低18.50%、偏置1.29%）；那不是本次初始策略结果。H25新模型未采用，原预测精度FAIL保留。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

SEED_CFD_BINDINGS={'approval':'4bd940f088373c3c9e0c2d227d18364b23457e0ef933a4add9d2345d74038364','result':'6221a7d2f8868eba8622f2e6b76d110d206dd4d9304627d890e01d71b6a7d893','review':'637d1e4a14c6fbd114f0a399ece07ab767502a9f7a1d556336b7ccafdf59354b'}

def second_seed_cfd_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'第二seed真实CFD状态未核验','cycles':0,'target':800,'running':False,'training':False,'physical_pass':None,'terminal_verified':False,'invocation':None,'previous_initial':initial_candidate_cfd_status(root,run)}
    try:
        approval=root/'docs/P064_B_SEED20261007_CFD_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!=SEED_CFD_BINDINGS['approval']:raise ValueError('seed CFD approval SHA')
        result=root/'artifacts/p064_b_seed20261007_projected_ppo_long_cfd_20261006/result.json'
        review=root/'docs/P064_B_SEED20261007_CFD_TERMINAL_REVIEW_20261006.md'
        if result.exists() and review.exists():
            if hashlib.sha256(result.read_bytes()).hexdigest()!=SEED_CFD_BINDINGS['result'] or hashlib.sha256(review.read_bytes()).hexdigest()!=SEED_CFD_BINDINGS['review']:raise ValueError('seed CFD terminal binding')
            info.update(status='第二seed闭环完成但未减阻；旧seed成功尚未跨seed重现',cycles=800,current_time=228.,running=False,training=False,physical_pass=False,terminal_verified=True,invocation='111022bf633246e69165f7b8eb3edb01')
            info['note']='主窗新seed增阻0.6174%，RMS比1.00622、均值偏置1.6946%；六窗减阻均未达原2%，两升力条件通过。旧seed同zero减阻3.8953%仍真实，不能称跨seed稳健。匹配观测诊断显示新策略投影后的动作较小，但未证明修复方法。训练/CFD均已结束，预测FAIL与H25未采用保留。'
            return info
        raw=run(['systemctl','--user','show','fluid-control-p064-b-seed20261007-projected-ppo-long-cfd-20261006.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='111022bf633246e69165f7b8eb3edb01':raise ValueError('seed CFD invocation')
        info['invocation']=state['InvocationID']
        path=root/'artifacts/p064_b_seed20261007_projected_ppo_long_cfd_20261006/progress.json'
        if path.exists():info.update(cfd_progress_counts(json.loads(path.read_text()),start=148.))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='第二seed策略：真实CFD闭环复验中' if info['running'] else '第二seedCFD已停止，等待独立终态复核'
        info['note']='CPU策略反馈＋配对不旋转基线，800周期；不是GPU训练，无在线FNO/MPC。第二seed PPO已完成32768步并独审，新物理效果尚未知。旧B约3.90%减阻/18.4%升力波动改善，初始权重无减阻，均为已完成历史对照；原物理2%/1.05/10%与预测FAIL不变。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def canonical_b00_cfd_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'对称坐标CFD状态未核验','cycles':0,'target':800,'running':False,'training':False,'physical_pass':None,'terminal_verified':False,'invocation':None,'previous_seed':second_seed_cfd_status(root,run)}
    try:
        approval=root/'docs/P064_B_SYMMETRY_CANONICAL_CFD_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='afb03b9e86931d1031d8e0ab1ce76dc180816ac76446d46823cf4aa350ee9f1e':raise ValueError('canonical CFD approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-symmetry-canonical-ppo-long-cfd-20261007.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='545ba2aebfb6412aaf41ee3281ccc802':raise ValueError('canonical CFD invocation')
        info['invocation']=state['InvocationID']
        path=root/'artifacts/p064_b_symmetry_canonical_ppo_long_cfd_20261007/progress.json'
        if path.exists():info.update(cfd_progress_counts(json.loads(path.read_text()),start=148.))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='对称坐标策略：真实CFD闭环验证中' if info['running'] else '对称坐标CFD已停止，等待独立复核'
        info['note']='CPU策略＋真实CFD，800周期，不是训练、无在线FNO/MPC。训练已完成32768步，当前检验物理效果，尚无通过结论。旧seed减阻3.90%、第二seed增阻0.62%都保留；固定2%/1.05/10%标准不变，完整预测FAIL仍独立存在。'
        result_path=root/'artifacts/p064_b_symmetry_canonical_ppo_long_cfd_20261007/result.json'
        review=root/'docs/P064_B_SYMMETRY_CANONICAL_CFD_TERMINAL_REVIEW_20261007.md'
        if not info['running'] and state.get('ExecMainStatus')=='0' and result_path.exists():
            if hashlib.sha256(result_path.read_bytes()).hexdigest()!='165b78194f84676b5ea0091b1d160ccf25f913a9f49c74a9424a7d0f03cde7dc':raise ValueError('canonical CFD result SHA')
            if hashlib.sha256(review.read_bytes()).hexdigest()!='cf7975dbd02da5dca41ddfafe409b86b3676c49e541dc2026988b02ca9f5408e':raise ValueError('canonical CFD review SHA')
            result=json.loads(result_path.read_text());primary=result['windows']['primary_final_60']
            info.update(status='对称坐标策略：真实CFD完成，主窗减阻3.96%',terminal_verified=True,physical_pass=True,cycles=800,primary=primary)
            info['note']='独审800周期：主窗减阻3.9567%、升力波动降低18.3457%、均值偏置1.066%，原三项通过；早首6.2D/U偏置17.56%仍失败。同第二seed旧方式增阻0.62%的负结果保留；旧成功seed减阻3.90%，不宣称显著优越。当前无训练/CFD运行；完整预测精度FAIL未改变，非全目标完成。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def b02_conversion_status(root,run=subprocess.check_output):
    root=Path(root);info={'status':'b02转换状态待核验','running':False,'training':False,'written_frames':0,'expected_frames':801,'invocation':None,'official_reader_verified':False}
    try:
        approval=root/'docs/P064_B02_CONTROLLED_TRAIN_CONVERSION_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='64e910fc20c7f5beeb0805ad159be3e0ff3e8d599f4560b7a0f42149aa53766a':raise ValueError('conversion approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b02-controlled-train-conversion-20261007.service','-p','InvocationID','-p','MainPID','-p','ActiveState','-p','SubState','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='f7a7550e035e4ee482205379eefa016a':raise ValueError('conversion invocation')
        info['invocation']=state['InvocationID']
        path=root/'artifacts/p064_b02_controlled_train_conversion_20261007/progress.json'
        if path.exists():
            p=json.loads(path.read_text());n=p['written_frames']
            if type(n) is not int or not 0<=n<=801 or p['expected_frames']!=801:raise ValueError('conversion count')
            info.update(written_frames=n,last_global_index=p['last_global_index'],current_batch=p['current_batch'])
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='CPU数据转换：b02受控801帧，非训练' if info['running'] else ('转换程序退出0，等待独立完整性核验' if state.get('ExecMainStatus')=='0' else '转换工程失败，未自动重试')
        info['note']='已保存真实CFD→官方Curator/Reader训练HDF；只受控branch，整轨train，原归一化不重拟合。没有FNO/PPO训练或新CFD求解；801计数不等于转换验收。E089六窗物理结果已独审，C50拒绝和预测FAIL保留。'
        review=root/'docs/P064_B02_CONTROLLED_TRAIN_CONVERSION_TERMINAL_REVIEW_20261007.md'
        result=root/'artifacts/p064_b02_controlled_train_conversion_20261007/result.json'
        if not info['running'] and state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0' and review.exists() and result.exists():
            if hashlib.sha256(review.read_bytes()).hexdigest()!='561624b0fd98a6db0fbff5722f950444b19d0de1aca3eea71f83ba445eb28b90':raise ValueError('conversion review SHA')
            if hashlib.sha256(result.read_bytes()).hexdigest()!='3faf153b2a9857e880634be3347f6596834b3609a5235007f668b141d10b5ade':raise ValueError('conversion result SHA')
            r=json.loads(result.read_text())
            if r['frames']!=801 or r['official_reader_verified'] is not True:raise ValueError('conversion terminal count')
            info.update(status='b02数据转换已完成并独审：801帧',written_frames=801,official_reader_verified=True,terminal_verified=True)
            info['note']='1614源文件、全部时间/动作/原始力端点、17导出容器清理已核；原归一化未改。可供另批训练视图整理，当前不是训练或CFD。原物理收益、早期失败及完整预测FAIL保持；没有自动启动新候选。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['error']=str(exc)
    return info

def b02_acquisition_status(root,run=subprocess.check_output):
    root=Path(root);info={'status':'b02采集尚未核验启动','cycles':0,'target':800,'running':False,'training':False,'physical_pass':None,'invocation':None,'note':'固定E082已训练策略，CPU真实CFD反馈；不是C50模型/PPO训练，无在线FNO/MPC。补充长期受控train数据，已有b02短探索数据并非空白。转换及后续训练另批；旧物理收益和预测FAIL不变。'}
    try:
        approval=root/'docs/P064_B_SYMMETRY_CANONICAL_B02_TRAIN_CFD_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='4c0276eec8e4c2e871bf0fc93cd1ac780a5e3c7263096a87ef9a047a5998a966':raise ValueError('b02 approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-symmetry-canonical-ppo-b02-train-acquisition-20261007.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='330e850af9e040eaaf10897443807173':raise ValueError('b02 invocation')
        info['invocation']=state['InvocationID']
        path=root/'artifacts/p064_b_symmetry_canonical_ppo_b02_train_acquisition_20261007/progress.json'
        if path.exists():info.update(cfd_progress_counts(json.loads(path.read_text()),start=106.))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']=('训练数据采集：固定策略b02真实闭环' if info['cycles']>0 else 'b02采集进程初始化中') if info['running'] else ('b02采集进程已退出，等待独立终态复核' if state.get('ExecMainStatus')=='0' else 'b02采集工程失败，未自动重试')
        review=root/'docs/P064_B_SYMMETRY_CANONICAL_B02_TRAIN_CFD_TERMINAL_REVIEW_20261007.md'
        result=root/'artifacts/p064_b_symmetry_canonical_ppo_b02_train_acquisition_20261007/result.json'
        if not info['running'] and state.get('ExecMainStatus')=='0' and review.is_file() and result.is_file():
            if hashlib.sha256(review.read_bytes()).hexdigest()!='283c12c310a779e7c946c162ec1f9f07d4722e49b323157ca6206281c5361064' or hashlib.sha256(result.read_bytes()).hexdigest()!='6883ceda495a443bc0edf95047f61bdbfc585cea9217112a92cb21e78299b9eb':raise ValueError('b02 terminal binding')
            info.update(status='b02真实闭环采集完成：主窗减阻4.26%，六窗通过',terminal_verified=True,physical_pass=True)
            info['note']='主窗减阻4.2628%、后升力波动降低17.7028%、均值偏置4.4528%；六窗原标准通过，两分支各801组U/p已独审。可转换但尚未转换/训练，需另批。固定E082策略，不是C50晋级；旧早期失败和完整预测FAIL保留。基本官方RL→真实CFD在线反馈链已实现，在线FNO/MPC是可选后续，整体预测质量和验证范围仍不足。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['error']=str(exc)
    return info

def canonical_seed6_cfd_status(root,run=subprocess.check_output):
    root=Path(root);info={'status':'seed20261006 b00尚未核验启动','cycles':0,'target':800,'running':False,'training':False,'physical_pass':None,'invocation':None}
    try:
        approval=root/'docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_CFD_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='da5f4f7681174f24fc5315097de723cbe47e71679b7c202aee11a6277e6bbe52':raise ValueError('seed6 CFD approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-symmetry-canonical-seed20261006-ppo-long-cfd-20261007.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='be48ee07057c42879d222548045233ef':raise ValueError('seed6 CFD invocation')
        info['invocation']=state['InvocationID']
        path=root/'artifacts/p064_b_symmetry_canonical_seed20261006_ppo_long_cfd_20261007/progress.json'
        if path.exists():info.update(cfd_progress_counts(json.loads(path.read_text()),start=148.))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='新seed20261006策略：b00真实CFD验证中' if info['running'] else '新seed b00进程已停止，等待独立复核'
        result_path=root/'artifacts/p064_b_symmetry_canonical_seed20261006_ppo_long_cfd_20261007/result.json'
        if not info['running'] and state.get('ExecMainStatus')=='0' and result_path.exists():
            review=root/'docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_CFD_TERMINAL_REVIEW_20261007.md'
            if hashlib.sha256(result_path.read_bytes()).hexdigest()!='9a7db1cff37e614dbefa908ec132cd8376cdfddd1f3149e2ba3436cebad57df4' or hashlib.sha256(review.read_bytes()).hexdigest()!='24e9a5ad02b0b36b34910ebe32e72de4c8cc5692f6090e4efa0dcad8439ac7fc':raise ValueError('seed6 CFD terminal binding')
            info.update(status='新seed b00已独审：主窗减阻3.74%',cycles=800,terminal_verified=True,physical_pass=True)
            info['note']='主窗减阻3.7440%、升力波动降低22.6210%、偏置3.6741%；早12.4偏置11.26%和首6.2偏置21.18%仍失败（首段也超过20%敏感性）。两个指定canonical seed主窗均通过，不等于任意seed稳健或预测精度通过。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    return info

def candidate_cfd_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'canonical b01尚未核验启动','cycles':0,'target':800,'running':False,'training':False,'physical_pass':None,'terminal_verified':False,'invocation':None,'previous_canonical_b00':canonical_b00_cfd_status(root,run)}
    try:
        approval=root/'docs/P064_B_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='ee010bbe1932e0b48b2f2e90a1b9dd77d463f86dad6367e0c0dd49a46f0b45f1':raise ValueError('canonical b01 approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-symmetry-canonical-ppo-b01-long-cfd-20261007.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='3a678d0c1d604f0eb821255c2848fdd4':raise ValueError('canonical b01 invocation')
        info['invocation']=state['InvocationID']
        path=root/'artifacts/p064_b_symmetry_canonical_ppo_b01_long_cfd_20261007/progress.json'
        if path.exists():info.update(cfd_progress_counts(json.loads(path.read_text()),start=130.))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['status']='同一对称坐标策略：b01真实CFD验证中' if info['running'] else 'b01进程已停止，等待独立终态核验'
        info['note']='FC-E085：只变固定初相位，130→210/800次CPU策略＋真实CFD；无在线FNO/MPC，不是训练，当前无物理结论。已完成b00减阻3.9567%、升力波动降低18.3457%、bias1.066%；其早期17.56%失败保留。原2%/1.05/10%与预测FAIL不变。'
        result_path=root/'artifacts/p064_b_symmetry_canonical_ppo_b01_long_cfd_20261007/result.json'
        if not info['running'] and state.get('ExecMainStatus')=='0' and result_path.exists():
            review=root/'docs/P064_B_SYMMETRY_CANONICAL_B01_CFD_TERMINAL_REVIEW_20261007.md'
            if hashlib.sha256(result_path.read_bytes()).hexdigest()!='c56a5cbccdf218953a0e1ea040da1c1ee7e2f1b574e7b509939a622f65b7495f' or hashlib.sha256(review.read_bytes()).hexdigest()!='4d77f711e48a3e5472345f7c7e56253955c00ef281943183928e9a66bf4ff40a':raise ValueError('canonical b01 terminal binding')
            info.update(status='canonical b01已完成：减阻4.01%，六窗原标准通过',cycles=800,physical_pass=True,terminal_verified=True)
            info['note']='E085独审：主窗减阻4.0091%、升力波动降低18.2810%、均值偏置3.6366%；六窗通过，首6.2偏置8.0614%。原E083 b00早期17.56%失败与预测精度FAIL保留，不是全目标完成或新holdout泛化。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['note']=str(exc)
    info['secondary']=canonical_seed6_cfd_status(root,run)
    secondary=info['secondary']
    if secondary.get('invocation'):
        info['note']+=f" FC-E086：{secondary['status']}，{secondary['cycles']}/800周期。"+secondary.get('note','尚无物理结论。两项均为CPU反馈，不是GPU训练。')
        if secondary['running']:
            info['status']=f"新seed b00真实CFD运行中 {secondary['cycles']}/800；b01{'已独审完成' if info['terminal_verified'] else '并行验证中'}"
        elif secondary.get('terminal_verified'):
            info['status']='真实闭环复验已完成：两指定seed主窗通过，早期失败保留'
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
        approval=root/'docs/P064_B_H25_TRAINING_R2_APPROVAL_20261006.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='852736b03edbd987da0e4f5a0d0cc55f1b1145fa980c989dc5b9cab298346f02':raise ValueError('H25 train approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-b-h25-training-r2-20261006.service','-p','InvocationID','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if state.get('InvocationID')!='9449ac16be65406eabad0515e88b6513':raise ValueError('H25 train invocation')
        info['invocation']=state['InvocationID']
        log=root/'artifacts/fcp064_b_h25_training_20261006_r2/run.log'
        if log.exists():
            if log.stat().st_size>8*2**20:raise ValueError('H25 train log size')
            info['windows'],info['updates']=h25_training_counts(log.read_text())
            info['last_log_update_unix']=log.stat().st_mtime
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['training']=info['running']
        if info['running']:info['status']='H25流场微调运行中（气动力网络冻结）'
        elif state.get('ExecMainStatus')!='0':info['status']='H25训练工程失败，未完成参数更新'
        else:info['status']='H25训练进程退出，等待终态独审'
        info['note']=f"25步预测训练：实际窗口{info['windows']}/256，参数更新{info['updates']}/32；100步仅为原采样窗长度。尺度1368窗及单步资源探针已完成。R1首窗后被旧H10累积器horizon检查拒绝，0参数更新，失败保留。闭环：三相位各800次真实CFD反馈；b07总阻力−3.90%、后升力波动−18.50%、均值偏置1.29%。预测精度：四旋转分支RMS仍未满足项目开发阈值。PPO训练使用FNO，部署为CPU策略反馈，无在线FNO/MPC。"
        if not info['running'] and state.get('MainPID')=='0' and state.get('ExecMainStatus')=='0':
            for name,digest in [('artifacts/fcp064_b_h25_training_20261006_r2/payload/result.json','557e0792eee538d8152c4997032309423a1197067c7198089768d0ddb40f5cf7'),('docs/P064_B_H25_TRAINING_R2_TERMINAL_REVIEW_20261006.md','e72434c773433ffc3fa3f51c0cb8d8de368d0f43caae34b699378fd37d1fd0dd')]:
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('H25 training terminal SHA')
            info.update(terminal_verified=True,status='H25训练R2已完成：256窗口／32参数更新；精度待评估')
            q=root/'artifacts/p064_b_h25_quick_ar_20261006/result.json'
            review=root/'docs/P064_B_H25_QUICK_AR_TERMINAL_REVIEW_20261006.md'
            if hashlib.sha256(q.read_bytes()).hexdigest()=='1b7bd2a2e99f9d02398df4cbcefc2d7dc5a486a02866d9a64856d0db9e9dafe0' and hashlib.sha256(review.read_bytes()).hexdigest()=='14e9b96dd2d34aa0280707a998ce808d215bffbe3c6561ceecf2a1ac65a9a851':
                info['status']='真实闭环有效；H25新模型预测退化，未采用'
                info['note']='训练已完成256窗口／32参数更新，后续同六案例评估已独审：Cl MAE 0.0624→0.0830，Cd MAE 0.0207→0.0467，六案例场误差均退化。保留原B策略：b07减阻3.90%、升力波动降低18.50%、偏置1.29%，三相位各800次真实CFD反馈。原B预测精度仍待改善；PPO训练用FNO，CPU部署无在线FNO/MPC。当前初始权重CFD对照见实时卡，另行整理复现文档，不是训练；R1失败保留。'
                info['candidate_adopted']=False
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

def coverage_d_training_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'D25覆盖训练状态待核验','running':False,'training':False,'windows':0,'updates':0,'target_windows':256,'target_updates':32,'invocation':None,'last_update_utc':None,'scientific_pass':None,
          'note':'气动力FNO分支微调28参数tensor，两bias及flow分支冻结；不是PPO。固定192原训练窗＋32b00＋32b02，总受控比例仍25%。C50拒绝、B/canonical真实闭环收益与预测FAIL保留；本次尚无精度结论。'}
    try:
        approval=root/'docs/P064_B00_B02_COVERAGE_D_TRAINING_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='fa8a99d22426f7c6e73fc56ff8f0c6062c0b11c0dcae146ee51f40aa7e32a3e8':raise ValueError('D approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-controlled-coverage-d-20261007.service','-p','InvocationID','-p','MainPID','-p','ActiveState','-p','SubState','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        invocation='171686b7ec154a0194348d26fd736cec'
        if state.get('InvocationID')!=invocation:raise ValueError('D invocation')
        info['invocation']=invocation
        info.update(parse_journal(run(['journalctl','--user',f'_SYSTEMD_INVOCATION_ID={invocation}','-o','json','--no-pager','-n','2000'],text=True,timeout=3)))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['training']=info['running'] and info['windows']>0
        info['status']=('D25 气动力FNO训练进行中' if info['training'] else 'D25进程初始化中，尚无训练窗口') if info['running'] else ('D25程序退出0，等待独立终态审查' if state.get('ExecMainStatus')=='0' else 'D25工程失败，未自动重试')
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:info['error']=str(exc)
    return info

def c50_training_status(root,run=subprocess.check_output):
    root=Path(root)
    info={'status':'C50状态待核验','running':False,'training':False,'windows':0,'updates':0,'target_windows':256,'target_updates':32,'invocation':None,'last_update_utc':None,'scientific_pass':None,
          'note':'R1因PYTHONPATH遗漏在项目导入时退出，0模型/0训练；R2仅补运行环境。气动力FNO分支微调：除两bias外28参数tensor参与优化，非仅末层、非PPO；独立flow分支冻结。既有闭环收益与预测精度FAIL均保留。'}
    try:
        approval=root/'docs/P064_CONTROLLED_DATA_DOSE_C_TRAINING_R2_APPROVAL_20261007.json'
        if hashlib.sha256(approval.read_bytes()).hexdigest()!='6d0d0f8dbdbae17a89d3b7dcc1717145b8e5a44464e928b5cb1a4e6debf2800f':raise ValueError('C50 approval SHA')
        raw=run(['systemctl','--user','show','fluid-control-p064-controlled-dose-c50-r2-20261007.service','-p','InvocationID','-p','MainPID','-p','ActiveState','-p','SubState','-p','ExecMainStatus'],text=True,timeout=3)
        state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        invocation='d80f61c62da64497bf378b6c7fd9c917'
        if state.get('InvocationID')!=invocation:raise ValueError('C50 invocation')
        info['invocation']=invocation
        journal=run(['journalctl','--user',f'_SYSTEMD_INVOCATION_ID={invocation}','-o','json','--no-pager','-n','2000'],text=True,timeout=3)
        info.update(parse_journal(journal))
        info['running']=state.get('ActiveState')=='active' and state.get('SubState')=='running' and int(state.get('MainPID','0'))>0
        info['training']=info['running'] and info['windows']>0
        info['status']=('C50 FNO气动力训练进行中' if info['training'] else 'C50 R2进程初始化中，尚无训练窗口') if info['running'] else ('C50程序退出0，等待独立终态审查' if state.get('ExecMainStatus')=='0' else 'C50工程失败，未自动重试')
        review=root/'docs/P064_CONTROLLED_DATA_DOSE_C_DEVELOPMENT_REVIEW_20261007.md'
        result=root/'artifacts/p064_arm_c50_development_h1_h5_20261007/result.json'
        if not info['running'] and review.is_file() and result.is_file() and hashlib.sha256(review.read_bytes()).hexdigest()=='e26d6e49186616f3818c33f438055d707e3c66b881640696006e280a6bc4e091' and hashlib.sha256(result.read_bytes()).hexdigest()=='9fa7c88759bf83fdca87d79ff305c0f3d4654368c7ec3059d7f9ba3f96c48c1a':
            info.update(status='C50训练和开发评估已完成，预测未改善，不采用',terminal_verified=True,scientific_pass=False)
            info['note']='气动力FNO分支微调（除两bias外28参数tensor，并非仅末层）；真实256窗口/32更新已独审。同面板H1后Cl MAE B0.13900→C0.13915、总Cd0.03807→0.04295，H1–H5各phase均退化。独立flow分支冻结、预测完全相同。Lead拒绝C晋级PPO/CFD，不自动再扫比例。保留B及两指定seed闭环主窗收益、早期失败和完整预测FAIL。C50任务已结束，其他当前工作见主卡；R1导入工程失败仍保留。'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        info['error']=str(exc)
    return info

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
