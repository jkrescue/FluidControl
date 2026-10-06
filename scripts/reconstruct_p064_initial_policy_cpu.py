"""CPU-only exact initial PPO weights; no learn, environment rollout, FNO or CFD."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle

ROOT=Path('/workspace/fluid_control')
INITIAL='6bc539885d8c63fc922eccaba0363593555cf1b85ece5e48783d79d2ea2fa1cf'
APPROVAL_SHA='ae327fee310bad562aceef35029595d20c9a3421d5d5be3dc3b68bb82649e9fe'
VEC_SHA='8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad'
RESULT_SHA='3c70e21327baae98f682fc0982ca3c3910cf6d1902f3d62175f980fd685817b3'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    approval=ROOT/'docs/P064_B_PPO_APPROVAL_20261006.json'
    assert sha(approval)==APPROVAL_SHA
    spec=json.loads(approval.read_text())
    for path,expected in spec['runtime_sources'].items():assert sha(path)==expected,path
    for package,version in spec['runtime_packages'].items():assert importlib.metadata.version(package)==version,package
    payload=ROOT/'artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload'
    result_path=payload/'result.json';vec_path=payload/'vecnormalize.pkl'
    assert sha(result_path)==RESULT_SHA and sha(vec_path)==VEC_SHA
    trained=json.loads(result_path.read_text());assert trained['policy_tensor_sha256_before']==INITIAL
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    torch.set_num_threads(1)
    with vec_path.open('rb') as stream:vec=pickle.load(stream)
    assert type(vec) is VecNormalize and vec.norm_obs is False and vec.norm_reward is False
    class NoRollout(gym.Env):
        observation_space=vec.observation_space
        action_space=vec.action_space
        def reset(self,**kwargs):raise RuntimeError('no environment reset authorized')
        def step(self,action):raise RuntimeError('no environment step authorized')
    env=DummyVecEnv([NoRollout for _ in range(4)])
    # Original official SB3 policy construction is on CPU before its .to(device).
    # A fresh empty optimizer exists, but no optimizer step is permitted.
    model=PPO('MlpPolicy',env,device='cpu',seed=20261006,n_steps=128,batch_size=256,
              n_epochs=4,learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2,
              ent_coef=0.,vf_coef=.5,max_grad_norm=.5,verbose=0)
    assert digest(model.policy)==INITIAL,'exact original initial weights were not reconstructed'
    assert model.num_timesteps==0 and model._n_updates==0 and not model.policy.optimizer.state
    assert all(p.device.type=='cpu' and p.grad is None for p in model.policy.parameters())
    assert model.observation_space == vec.observation_space and model.action_space == vec.action_space
    assert vec.observation_space.shape==(69,) and vec.action_space.shape==(1,)
    assert np.array_equal(vec.action_space.low,np.array([-.75],np.float32))
    assert np.array_equal(vec.action_space.high,np.array([.75],np.float32))
    vec.training=False
    before=pickle.dumps(vec.__dict__)
    probe=np.linspace(-20,20,69,dtype=np.float32)[None]
    assert np.array_equal(vec.normalize_obs(probe.copy()),probe)
    assert pickle.dumps(vec.__dict__)==before and sha(vec_path)==VEC_SHA
    a.output.mkdir()
    model.save(a.output/'ppo_initial.zip')
    reloaded=PPO.load(a.output/'ppo_initial.zip',device='cpu')
    assert reloaded.observation_space == vec.observation_space and reloaded.action_space == vec.action_space
    assert digest(reloaded.policy)==INITIAL and not reloaded.policy.optimizer.state
    assert digest(model.policy)==INITIAL and not model.policy.optimizer.state
    result=dict(status='P064_EXACT_INITIAL_POLICY_RECONSTRUCTED_CPU_NOT_CFD',
        exact_spaces_cpu_reload_verified=True,observation_dtype=str(vec.observation_space.dtype),action_dtype=str(vec.action_space.dtype),
        initial_tensor_sha256=INITIAL,seed=20261006,optimizer_created=True,optimizer_steps=0,
        timesteps=0,ppo_n_updates=0,environment_steps=0,fno_loaded=False,gpu_used=False,
        policy_file='ppo_initial.zip',policy_sha256=sha(a.output/'ppo_initial.zip'),
        trained_result_sha256=RESULT_SHA,trained_approval_sha256=APPROVAL_SHA,
        shared_identity_vecnormalize_sha256=VEC_SHA,normalization_nontrivial=False,
        vec_object_unchanged_by_normalize=True,source_sha256=sha(__file__),
        limitation='Same trained identity VecNormalize bytes reused; isolates policy weights, not all training artifacts. No initialization seed selection.',
        runtime_sources=spec['runtime_sources'],runtime_packages=spec['runtime_packages'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runtime_sources','runtime_packages')}),flush=True)
    env.close()

if __name__=='__main__':main()
