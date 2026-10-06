"""Actual HydroGym/canonical reset + H5 wrapper; synthetic state/stepper only."""
from collections import deque
from pathlib import Path
import unittest
import numpy as np
import torch
from hydrogym import FlowEnv, PDEBase, TransientSolver
from stable_baselines3.common.vec_env import DummyVecEnv
from fluid_control.full40_canonical_hydrogym import Full40CanonicalSurrogateFlow
from exploratory_h5_hydrogym import wrap_exploratory_h5
import p026_history_inference as history
import exploratory_diverse_h5_resets as adapter


def packets():
    result=[]
    for index,(case,frame) in enumerate(adapter.starts('00')):
        time=148. if frame==0 else 154.2
        omega=[0,-.75,-.375,0,.375,.75][index]
        forces=np.zeros((62,4));forces[:,0]=1;forces[:,2]=1;forces[:,3]=index*.01
        result.append(dict(case=case,frame=frame,path=Path(case+'.h5'),
            field=torch.full((3,128,256),index*.01),mask=torch.ones(1,128,256),
            omega=omega,force=forces[-1].astype(np.float32),time=time,
            history_times=np.round(time-.1*np.arange(61,-1,-1),8),history_forces=forces,
            x=np.linspace(8,25,256),y=np.linspace(4,11,128),provenance={'synthetic':True}))
    return result


class SyntheticCanonicalFlow(Full40CanonicalSurrogateFlow):
    def __init__(self):
        p=packets()[0]
        self.device=torch.device('cpu');self.MAX_CONTROL=.75;self.DEFAULT_DT=.1
        self.mask=p['mask'];self.x=p['x'];self.y=p['y'];self.case_path=p['path'];self.frame=0
        self.force_channels=('front_cd','front_cl','rear_cd','rear_cl')
        self.fno_history_runtime={'profile':'p026_k1','history_length':1}
        self.state_mean=torch.zeros(3,1,1);self.state_std=torch.ones(3,1,1)
        self.shedding_period=6.15;self.max_delta_omega=.1;self.max_abs_normalized_state_guard=12.
        self.canonical_baseline={'total_drag':2.,'rear_cl_fluctuation_rms':1.,'source':'synthetic'}
        self._canonical_initializing=False;self._reward_history=deque()
        self._probe_indices=self._build_probe_indices()
        self.initial_cfd_time=148.;self.initial_state={'field':p['field'],'omega':0.,'force':p['force'],
            'fno_history_runtime':dict(self.fno_history_runtime),
            'fno_history':history.trajectory_history(p['field'][None],torch.zeros(1),torch.tensor([0]),k=1)}
        self._initial_reward_history=tuple(zip(p['history_times'],p['history_forces']))
        PDEBase.__init__(self)


class SyntheticStepper(TransientSolver):
    def step(self,iteration,control):
        f=self.flow
        f.q=f.q+.001;f.t+=self.dt
        f.fno_history=history.trajectory_history(f.q[None],torch.tensor([f.omega/.75]),torch.tensor([0]),k=1)
        f.record_reward_sample()
        return f


def make_env():
    raw=FlowEnv({'flow':SyntheticCanonicalFlow,'solver':SyntheticStepper,
                 'solver_config':{'dt':.1},'max_steps':5})
    return adapter.wrap_phase_cycle(wrap_exploratory_h5(raw),packets())


class Tests(unittest.TestCase):
    def test_all_six_actual_reset_step_restore(self):
        env=make_env()
        try:
            for index in range(12):
                obs,info=env.reset();p=packets()[index%6];f=env.unwrapped.flow
                self.assertEqual(info['reset_panel_index'],index%6)
                self.assertEqual(f.t,p['time']);self.assertEqual(f.omega,p['omega'])
                self.assertTrue(torch.equal(f.q,p['field']))
                self.assertTrue(torch.equal(f.fno_history.states[:,0],f.q[None]))
                self.assertFalse(bool(f.fno_history.padding_mask.any()))
                np.testing.assert_array_equal(np.stack([x[1] for x in f._reward_history]),p['history_forces'])
                self.assertEqual(obs[-1],np.float32(p['omega']))
                for step in range(5):
                    _,_,terminated,truncated,stepinfo=env.step(np.array([p['omega']]))
                    self.assertFalse(terminated);self.assertEqual(truncated,step==4)
                    self.assertEqual(stepinfo['reset_panel_index'],index%6)
                    self.assertEqual(stepinfo['reset_frame'],p['frame'])
                snapshot=f.copy_runtime_snapshot()
                f.q.fill_(9);f.force[:]=9;f._reward_history.clear();f.omega=0;f.t=0
                f.restore_runtime_snapshot(snapshot)
                self.assertEqual(f.t,snapshot['time'])
                self.assertTrue(torch.equal(f.q,snapshot['state']['field']))
                f.q.fill_(8);f.fno_history.states.fill_(7)
            self.assertEqual(env.reset_counts,[2]*6)
        finally:env.close()

    def test_actual_sb3_autoreset_cycles_not_terminal_obs(self):
        env=DummyVecEnv([make_env])
        try:
            env.reset()
            for _ in range(5):obs,_,done,infos=env.step(np.zeros((1,1)))
            self.assertTrue(done[0]);self.assertTrue(infos[0]['TimeLimit.truncated'])
            self.assertEqual(obs[0,-1],-.75)
            self.assertEqual(infos[0]['terminal_observation'][-1],0)
        finally:env.close()


if __name__=='__main__':unittest.main()
