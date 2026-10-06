import ast
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import json
from types import SimpleNamespace

HERE=Path(__file__).parent
PATH=HERE/'infer_projected_policy_h1_h5_replay.py'
if not PATH.exists(): PATH=HERE.parent/'scripts'/PATH.name
spec=importlib.util.spec_from_file_location('replay_infer',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def packed(q,mask,now,nxt):
    return torch.cat((q[0],mask,now.reshape(1,1,1).expand(1,128,256),
        nxt.reshape(1,1,1).expand(1,128,256)),0)

def test_actual_rollout_clock_and_residual_no_truth_access():
    seen=[]
    def flow(x):
        seen.append(x.clone());raw=torch.zeros((1,7,128,256));raw[:,:3]=.25;return raw
    def aero(x):
        raw=torch.zeros((1,7,128,256));raw[:,3:]=x[:,0:1];return raw
    q=torch.zeros((1,3,128,256));mask=torch.ones((1,1,128,256))
    transitions=[{'omega_now':i*.01,'omega_next':(i+1)*.01} for i in range(5)]
    sm=torch.tensor([1.,2.,3.]).reshape(1,3,1,1);ss=torch.ones_like(sm)*2
    states,forces=m.rollout(q,mask,transitions,flow,aero,packed,torch,
        torch.ones(4),torch.ones(4)*3,sm,ss,lambda:None)
    assert len(seen)==5
    for i,x in enumerate(seen):
        assert torch.all(x[:,:3]==i*.25)
        assert x[0,4,0,0]==torch.tensor(i*.01,dtype=torch.float32)/.75
        assert x[0,5,0,0]==torch.tensor((i+1)*.01,dtype=torch.float32)/.75
        assert np.array_equal(forces[i],np.ones(4)*(1+3*i*.25))
    assert states[-1][0,0,0]==3.5

def test_bad_model_shape_and_nonfinite_fail():
    args=(torch.zeros((1,3,128,256)),torch.ones((1,1,128,256)),
        [{'omega_now':0.,'omega_next':0.}],lambda x:torch.zeros((1,6,128,256)),
        lambda x:torch.zeros((1,7,128,256)),packed,torch,torch.zeros(4),torch.ones(4),
        torch.zeros((1,3,1,1)),torch.ones((1,3,1,1)),lambda:None)
    with pytest.raises(ValueError,match='output shape'):m.rollout(*args)

def test_pending_rejected_before_any_paths():
    with pytest.raises(ValueError,match='not approved'):m.validate_spec({'status':'PENDING'})

def test_precision_and_allocator_load_order_no_optimizer():
    text=PATH.read_text();tree=ast.parse(text)
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    src=ast.get_source_segment(text,worker)
    assert src.index('set_per_process_memory_fraction')<src.index('flow,aero,identity = load_bound_k1')
    assert src.index("set_float32_matmul_precision('high')")<src.index('flow,aero,identity = load_bound_k1')
    assert src.index('flow,aero,identity = load_bound_k1')<src.index('override_inference_precision')
    assert 'torch.inference_mode()' in src and '.backward(' not in text and 'torch.optim' not in text

def test_supervisor_guards_and_cleanup_are_real_parent_calls():
    text=PATH.read_text()
    for value in ['base.cgroup_limits()', 'base.memory_ok(base.memory(),True)',
        "row['elapsed']<600", 'base.stop(process)', 'start_new_session=True', 'time.sleep(.5)']:
        assert value in text

def test_exact_outer_and_child_link(monkeypatch,tmp_path):
    base=SimpleNamespace(cgroup_limits=lambda:{'memory_max':12*2**30,'memory_swap_max':0})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(m.importlib.metadata,'version',lambda k:m.VERSIONS[k])
    monkeypatch.setattr(m.os,'getppid',lambda:123)
    props={'MainPID':'123',**m.UNIT_PROPERTIES}
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'\n'.join(f'{k}={v}' for k,v in props.items()))
    (tmp_path/'supervisor_child.json').write_text(json.dumps({'pid':123,'token':'a'*64,
        'started_monotonic':m.time.monotonic()}))
    with pytest.raises(ValueError,match='authorized worker'):m.verify_execution({'unit':'trial.service'},base,tmp_path,True)
    monkeypatch.setenv('REPLAY_CHILD_TOKEN','a'*64)
    m.verify_execution({'unit':'trial.service'},base,tmp_path,True)
    props['MainPID']='999'
    with pytest.raises(ValueError,match='supervisor PID'):m.verify_execution({'unit':'trial.service'},base,tmp_path,True)

def test_smaller_cgroup_is_not_exact(monkeypatch,tmp_path):
    base=SimpleNamespace(cgroup_limits=lambda:{'memory_max':8*2**30,'memory_swap_max':0})
    with pytest.raises(ValueError,match='exact cgroup'):m.verify_execution({},base,tmp_path)
