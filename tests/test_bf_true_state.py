import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

spec=importlib.util.spec_from_file_location('diag',Path(__file__).resolve().parents[1]/'scripts/diagnose_bf_true_state.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    a=dict(initial_state=np.full((3,128,256),2,np.float32),
        truth_states=np.stack([np.full((3,128,256),j,np.float32) for j in range(3,8)]),
        predicted_states=np.zeros((5,3,128,256),np.float32),mask=np.ones((1,128,256),np.float32),
        omega=np.arange(6,dtype=np.float32)[:,None]/10,time=130+np.arange(6)*.1,
        truth_forces=np.arange(24,dtype=np.float32).reshape(6,4),predicted_forces=np.ones((5,4),np.float32),x=np.arange(256),y=np.arange(128))
    row=dict(branch='b01',start_index=0,predicted_forces=a['predicted_forces'].tolist(),
        transitions=[dict(lead=j,omega_now=float(a['omega'][j-1,0]),omega_next=float(a['omega'][j,0]),truth_force=a['truth_forces'][j].tolist()) for j in range(1,6)])
    return a,row

def test_complete_schema_and_clock():
    a,r=fixture();m.validate_arrays(a,r)
    # JSON double 0.3 is cast to FP32 by the original inference worker.
    r['transitions'][2]['omega_next']=.3
    r['transitions'][3]['omega_now']=.3
    m.validate_arrays(a,r)
    r['transitions'][2]['omega_now']=.7
    with pytest.raises(ValueError,match='action clock'):m.validate_arrays(a,r)

def test_current_not_target_no_future_history():
    a,_=fixture()
    for j in range(1,6):assert np.all(m.current_state(a,j)==j+1)
    before=m.current_state(a,3).copy();a['truth_states'][2:]=999
    np.testing.assert_array_equal(m.current_state(a,3),before)
    with pytest.raises(ValueError):m.current_state(a,0)

def test_actual_official_build_input_normalization_clock():
    path=Path('/workspace/fluid_control/artifacts/fcp064_training_source_20261006_immutable/scripts/p026_state_history.py')
    h=m.module(path,'fixture_official_history')
    a,_=fixture();a['mask'][:,:,0]=0
    norm=dict(state_mean=[1,1,1],state_std=[2,2,2])
    packed,mask=m.prepare_input(a,3,norm,h.build_input,torch,'cpu')
    assert packed.shape==(1,6,128,256)
    assert torch.all(packed[0,:3,:,1:]==1.5)
    assert torch.all(packed[0,:3,:,0]==0)
    torch.testing.assert_close(packed[0,4],torch.full((128,256),float(a['omega'][2,0])/.75))
    torch.testing.assert_close(packed[0,5],torch.full((128,256),float(a['omega'][3,0])/.75))

def test_signed_total_cd_before_absolute():
    r=m.metrics([[1,2,-1,3]],[[0,0,0,0]])
    assert r['mae']==[1,2,1,3,0]

def test_time_and_truth_fail_closed():
    a,r=fixture();a['time'][4]+=.1
    with pytest.raises(ValueError,match='time clock'):m.validate_arrays(a,r)
    a,r=fixture();r['transitions'][0]['truth_force'][0]+=1
    with pytest.raises(ValueError,match='force clock'):m.validate_arrays(a,r)

def test_fixed_counts_and_no_optimizer_or_flow_call_in_worker():
    import ast,inspect
    tree=ast.parse(inspect.getsource(m.worker))
    assert m.PROTOCOL['aero_calls']==2*len(m.KEYS)*5==160
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('backward','step','Adam','AdamW') for n in ast.walk(tree))
    assert 'aero_force(models[1],packed' in inspect.getsource(m.worker)

def test_pair_identity_rejects_mismatch():
    rows=[dict(model=label,phase=p,start=s,lead=j,current_state_sha256='same',normalized_input_sha256='same',time_current=j-1,time_target=j,omega_current=0,omega_next=0,truth=[0]*4) for label in ('B','F') for p,s in m.KEYS for j in range(1,6)]
    m.paired_input_check(rows)
    rows[-1]['normalized_input_sha256']='different'
    with pytest.raises(ValueError,match='paired input'):m.paired_input_check(rows)

def test_descriptive_differences_all_groups_and_signs():
    rows=[dict(model=label,phase=p,start=s,lead=j,teacher_forced=[v]*4,saved_ar=[3]*4,hold_origin_force=[0]*4,truth=[0]*4) for label,v in (('B',1),('F',2)) for p,s in m.KEYS for j in range(1,6)]
    d=m.descriptive_differences(m.summarize(rows))
    assert set(d)=={'pooled','b01','b03'}
    assert d['pooled']['5']['F_minus_B']['teacher_forced']['mae']==[1,1,1,1,2]
    assert d['b01']['1']['teacher_minus_savedAR']['B']['signed_bias']==[-2,-2,-2,-2,-4]

def test_aero_only_forward_mask_pool_and_denormalization():
    class Aero(torch.nn.Module):
        def forward(self,x):
            return torch.arange(7,dtype=x.dtype).reshape(1,7,1,1).expand(1,7,128,256)
    aero=Aero();calls=[];h=aero.register_forward_hook(lambda *_:calls.append(1))
    pred=m.aero_force(aero,torch.zeros(1,6,128,256),torch.ones(1,128,256),torch.ones(4),torch.full((4,),2.),torch)
    torch.testing.assert_close(pred,torch.tensor([7.,9.,11.,13.]))
    assert len(calls)==1;h.remove()

def test_preflight_rejects_resource_drift_before_files():
    bad=dict(protocol=m.PROTOCOL,resources=dict(m.RESOURCES,memory_bytes=1),runtime_versions=m.VERSIONS)
    with pytest.raises(ValueError,match='resources'):m.preflight(bad)
