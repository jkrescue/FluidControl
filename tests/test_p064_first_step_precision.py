import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

def load(name):
    s=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1] / 'scripts' / (name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

w=load('diagnose_p064_first_step_precision')
sup=load('supervise_p064_first_step_precision')

@pytest.fixture(autouse=True)
def restore_flags():
    old=w.precision_flags(torch)
    yield
    torch.set_float32_matmul_precision(old['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=old['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=old['cudnn_allow_tf32']

def run(expected):
    states=np.full((101,3,2,2),99.,dtype=np.float32);states[0]=2
    calls=[]
    def reset(q,omega,k):
        assert k==1 and q.shape==(1,3,2,2) and torch.equal(q,torch.full_like(q,2))
        assert omega.tolist()==[0.]
        return q
    def dual(network,history,mask,following):
        calls.append(w.precision_flags(torch))
        assert following.tolist()==pytest.approx([.1/.75])
        raw=torch.zeros((1,7,2,2));raw[:,3:]=torch.arange(1.,5.)[None,:,None,None]
        return raw,None,None
    rows=w.teacher_forced_h1_predictions(states,np.ones((101,1,2,2)),
        np.array([0.]+[.1]*100),np.zeros((101,4)),np.arange(101)*.1,
        action_scale=.75,state_mean=torch.zeros(1,3,1,1),state_std=torch.ones(1,3,1,1),
        force_mean=torch.zeros(4),force_std=torch.ones(4),network=None,history_k=1,
        reset_history=reset,history_dual_step=dual,expected_high=expected)
    return rows,calls

def test_exact_two_calls_true_q0_and_effective_flags():
    rows,calls=run([1.,2.,3.,4.])
    assert len(rows)==len(calls)==2
    assert [r['step'] for r in rows]==[1,1]
    assert calls==[dict(float32_matmul_precision='high',cuda_matmul_allow_tf32=True,cudnn_allow_tf32=True),dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False)]

def test_high_replay_failure_closed():
    with pytest.raises(ValueError,match='exact replay'):run([1.,2.,3.,4.000001])

def test_signed_pairs_all_four_and_inputs_match():
    rows,_=run([1.,2.,3.,4.])
    reports=[dict(case=n,rows=rows) for n in w.CASES]
    result=w.precision_comparison(reports)
    assert len(result['action_minus_zero'])==8
    assert all(r['predicted_delta_total_cd']==0 for r in result['action_minus_zero'])
    assert len(result['per_case'])==6

def test_model_digest_detects_change():
    from types import SimpleNamespace
    m=SimpleNamespace(flow_model=torch.nn.Linear(2,2),aerodynamic_model=torch.nn.Linear(2,2))
    before=w.tensor_digest(m)
    with torch.no_grad():m.aerodynamic_model.weight.add_(1)
    after=w.tensor_digest(m)
    assert before['flow_model']==after['flow_model'] and before['aerodynamic_model']!=after['aerodynamic_model']

def test_supervisor_resource_contract_and_new_reference():
    assert sup.RESOURCES['memory_bytes']==12*2**30
    assert sup.RESOURCES['allocator_bytes']==6*2**30
    assert sup.RESOURCES['swap_bytes']==0
    assert sup.RESOURCES['startup_available_gib']==50 and sup.RESOURCES['runtime_available_gib']==22
    root=sup.ROOT
    names=['data_manifest','normalization','config','aerodynamic_model','candidate_manifest','training_config','autoregressive_reference','h1_reference']
    inputs={n:root/'dummy'/n for n in names}
    spec={'container_name':'p064-first-step-precision-20261007','inputs':{n:{'sha256':'pin'} for n in names}}
    cmd=sup.command(spec,root/'worker.py',inputs,root/'source',root/'output')
    assert '--h1-reference' in cmd and sup.IMAGE in cmd
    assert cmd[cmd.index('--memory-swap')+1]=='12g'

