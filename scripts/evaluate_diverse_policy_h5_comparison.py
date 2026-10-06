"""Fixed24 train resets, deterministic two-policy H5 inference; no optimization."""
import argparse
import hashlib
import importlib.util
import importlib
import json
import os
from pathlib import Path
import time

STATUS='DIVERSE_POLICY_H5_COMPARISON_EXECUTION_APPROVED'
COMPLETE='DIVERSE_POLICY_H5_COMPARISON_COMPLETE_NOT_ADMISSION'
TRAINER_SHA='4d681771736b63b628712d3b62fcdde831601e80221aef6f1fd78a4b6840ff01'
POLICIES={
 '4096':('8dc8cabf2104654345f270e3fb86edca7752cf4c883112c0a4cbd3a181acea9b','6988d4d161bc69c8bbd89d477e9320ad9ef264d35c9dee0bbf63954d4cdfce70','cd5775e4647280b77803de9a5ced6abdf6378cded4f676f935bd9836350c3640'),
 '32768':('5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a','3161ba46c65bac3bc23fa4ccb300c52c95fda63ee4190d9f30d2f0bd4b9040ec','ff3532a604b6816fb3ad4c7a11edfcd579bcb924445abca52a2fdab8ea4dcf20')}

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    spec=importlib.util.spec_from_file_location('bound_evaluation_reference',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def evaluate_panel(environments,policy,guard):
    rows=[]
    for phase,env in environments:
        for slot in range(6):
            obs,reset_info=env.reset()
            steps=[];total=0.
            for lead in range(1,6):
                guard()
                action,_=policy.predict(obs,deterministic=True)
                obs,reward,terminated,truncated,info=env.step(action)
                require(not terminated and bool(truncated)==(lead==5),'unexpected H5 endpoint')
                require(info['reset_panel_index']==slot,'reset order changed')
                total+=float(reward)
                steps.append({'lead':lead,'reward':float(reward),'info':info})
            rows.append({'phase':phase,'slot':slot,'case':info['reset_case'],'frame':info['reset_frame'],
                         'return':total,'steps':steps})
    require(len(rows)==24,'exact24 train starts')
    return rows

def aggregate(rows):
    require(len(rows)==24 and len({(r['phase'],r['slot']) for r in rows})==24,'fixed24 coverage')
    keys=sorted(k for k in rows[0]['steps'][0]['info'] if k.startswith('reward_'))
    return {'mean_case_return':sum(r['return'] for r in rows)/24,
            'mean_case_component_sums':{k:sum(sum(float(s['info'][k]) for s in r['steps']) for r in rows)/24 for k in keys}}

def execute(s,approval_sha,output):
    require(s['status']==STATUS and s['execution_authorized'] is True,'separate approval')
    require(os.environ.get('DIVERSE_POLICY_H5_EVAL_SUPERVISED')==approval_sha,'supervisor required')
    require(sha(__file__)==s['source_files'][str(Path(__file__).resolve())],'worker source')
    require(sha(s['reference_trainer'])==TRAINER_SHA,'reviewed frozen training API')
    ref=load(s['reference_trainer']);ref.validate_files(s);proof=ref.read_packet_verification(s)
    require(set(s['evaluation_policies'])==set(POLICIES),'exact two policies')
    for label,expected in POLICIES.items():
        for key,digest in zip(('policy','vecnormalize','result'),expected):
            b=s['evaluation_policies'][label][key]
            require(b['sha256']==digest and sha(b['path'])==digest,'actual policy binding')
        result=json.loads(Path(s['evaluation_policies'][label]['result']['path']).read_text())
        require(result['timesteps']==int(label) and result['fno_tensors_unchanged'] is True,'completed producer')
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from omegaconf import OmegaConf
    from physicsnemo.utils.checkpoint import load_checkpoint
    from train_tandem_fno import build_model
    from fluid_control.dual_fno import load_dual_fno
    from fluid_control.dual_control_contract import p026_runtime_binding
    from train_full40_hydrogym_ppo_canonical import validate_train20_baselines
    from exploratory_h5_hydrogym import make_exploratory_env
    from exploratory_diverse_h5_resets import load_packet,packet_identity,wrap_phase_cycle
    for name,path in s['import_bindings'].items():
        actual=Path(importlib.import_module(name).__file__).resolve()
        require(actual==Path(path).resolve() and sha(actual)=={**s['source_files'],**s['runtime_sources']}[str(actual)],'actual runtime import')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one GPU')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.06,6*2**30/total),0)
    torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    network,identity=load_dual_fno(Path(s['inputs']['manifest']['path']),OmegaConf.load(s['inputs']['config']['path']),
        torch.device('cuda:0'),build_model=build_model,load_checkpoint=load_checkpoint,expected_manifest_sha256=ref.K1_SHA)
    network.eval().requires_grad_(False)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=ref.model_digest(network);runtime=p026_runtime_binding(identity)
    packets=[load_packet(s['data_root'],s['cases_root'],r['case'],r['frame']) for r in ref.fixed_panel()]
    require(ref.same_json([packet_identity(p) for p in packets],proof['packets']),'actual packet proof')
    baselines=validate_train20_baselines(Path(s['inputs']['baseline']['path']))
    output.mkdir(parents=False,exist_ok=False);start=time.monotonic();panels={}
    def guard():
        ref.frozen(network)
        require(ref.precision(torch)=={'matmul':'highest','cuda_tf32':False,'cudnn_tf32':False},'precision changed')
        require(time.monotonic()-start<240,'worker deadline')
        require(torch.cuda.memory_allocated()<=6*2**30,'6GiB tensor allocation cap')
    for label in ('4096','32768'):
        # Both producers explicitly use identity normalization; never deserialize a normalizer.
        producer=json.loads(Path(s['evaluation_policies'][label]['result']['path']).read_text())
        require(producer['protocol']['norm_obs'] is False and producer['protocol']['norm_reward'] is False,'identity observation normalization')
        policy=PPO.load(s['evaluation_policies'][label]['policy']['path'],device='cuda:0')
        require(policy.observation_space.shape==(69,) and policy.action_space.shape==(1,)
                and np.array_equal(policy.action_space.low,np.array([-.75],np.float32))
                and np.array_equal(policy.action_space.high,np.array([.75],np.float32)),'actual policy spaces')
        policy.policy.set_training_mode(False);policy.policy.requires_grad_(False)
        policy_before=ref.model_digest(policy.policy);envs=[]
        try:
            for i,phase in enumerate(('00','02','04','06')):
                anchor=make_exploratory_env(data=s['data_root'],case=s['train_cases'][i],network=network,
                    checkpoint_epoch=identity.aerodynamic.epoch,baseline=baselines['b'+phase],device='cuda:0',
                    cases_root=s['cases_root'],fno_history_runtime=runtime)
                envs.append((phase,wrap_phase_cycle(anchor,packets[i*6:i*6+6])))
            with torch.no_grad():rows=evaluate_panel(envs,policy,guard)
            require(ref.model_digest(policy.policy)==policy_before,'policy tensors changed')
            panels[label]={'rows':rows,'aggregate':aggregate(rows),'policy_tensors_unchanged':True}
        finally:
            for _,env in envs:env.close()
            del policy
    require(ref.model_digest(network)==before,'FNO tensors changed')
    require([(r['phase'],r['slot'],r['case'],r['frame']) for r in panels['4096']['rows']]
            ==[(r['phase'],r['slot'],r['case'],r['frame']) for r in panels['32768']['rows']],'matched pair identities')
    result={'status':COMPLETE,'panels':panels,'paired_return_deltas':[b['return']-a['return'] for a,b in
        zip(panels['4096']['rows'],panels['32768']['rows'],strict=True)],'fno_tensors_unchanged':True,
        'packet_verification':s['packet_verification'],'approval_sha256':approval_sha,'optimizer_steps':0,
        'cfd_executed':False,'scientific_admission':False,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'wall_seconds':time.monotonic()-start}
    ref.write(output/'result.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);p.add_argument('--approval-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    require(a.execute and sha(a.approval)==a.approval_sha256,'explicit approved execution')
    s=json.loads(a.approval.read_text());require(a.output.resolve()==Path(s['output']).resolve(),'output binding')
    execute(s,a.approval_sha256,a.output)
