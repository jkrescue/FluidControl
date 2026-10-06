import importlib.util
from pathlib import Path
import numpy as np
import torch
import sys

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('trainfit',HERE.parent/'scripts'/'diagnose_p064_k1bg_b00_train_fit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def sample(start=0):
    return dict(state=torch.zeros(3,128,256),target_state=torch.zeros(5,3,128,256),
                omega=torch.arange(6,dtype=torch.float32).reshape(6,1)/10,
                target_force=torch.arange(20,dtype=torch.float32).reshape(5,4),
                mask=torch.ones(1,128,256),time=torch.tensor(float(start)/10)), \
           dict(case='b00_projected_ppo_train',step=start,rollout_steps=5,split='train')

def test_protocol_is_fixed_train_only_true_state_not_ar():
    assert m.STARTS == tuple(range(0,701,100))
    assert m.PROTOCOL['true_state_single_step_predictions']==120
    assert m.PROTOCOL['aero_calls']==120 and m.PROTOCOL['flow_calls']==0
    assert m.PROTOCOL['selection']=='mechanical_starts_no_dev_selection'

def test_sample_contract_and_clock():
    s,metadata=sample(700);m.validate_sample(s,metadata)
    for lead in range(1,6):
        state=s['state'] if lead==1 else s['target_state'][lead-2]
        assert state.data_ptr()==(s['state'] if lead==1 else s['target_state'][lead-2]).data_ptr()
        assert s['target_force'][lead-1,0].item()==4*(lead-1)

def test_prepare_input_uses_true_current_state_and_actual_action_planes():
    scripts=Path('/workspace/fluid_control/artifacts/p064_ar5_reset_g_development_source_20261007_immutable/scripts')
    sys.path.insert(0,str(scripts))
    from p026_state_history import build_input
    s,_=sample(0)
    for lead in (1,2,5):
        s['state'].fill_(10.)
        for index in range(5):s['target_state'][index].fill_(20.+index)
        packed,mask,state=m.prepare_input(s,lead,build_input,'cpu')
        expected=10. if lead==1 else 20.+lead-2
        assert torch.all(state==expected)
        assert tuple(packed.shape)==(1,6,128,256)
        assert torch.all(packed[0,4]==s['omega'][lead-1])
        assert torch.all(packed[0,5]==s['omega'][lead])

def test_paired_check_requires_all_three_identical_inputs():
    rows=[]
    for label in m.LABELS:
        for start in m.STARTS:
            for lead in range(1,6):
                rows.append(dict(model=label,start=start,lead=lead,state_sha256='s',input_sha256='i',
                    target_sha256='t',omega_current=0.,omega_next=.1,time_current=float(start),
                    time_target=float(start)+.1,truth=[1.,2.,3.,4.],prediction=[1.,2.,3.,4.]))
    m.paired_check(rows)
    rows[-1]['input_sha256']='wrong'
    try:m.paired_check(rows)
    except ValueError:return
    raise AssertionError('mismatched model input accepted')

def test_metrics_include_rear_cl_and_total_cd():
    truth=np.array([[1.,2.,3.,4.],[2.,4.,6.,8.]])
    out=m.metrics(truth+1,truth)
    assert out['channels']==['front_Cd','front_Cl','rear_Cd','rear_Cl','total_Cd']
    assert out['mae']==[1.,1.,1.,1.,2.]

def test_model_kinds_are_exact():
    assert m.KINDS=={'K1':'FC_P026_K1_HISTORY_FORCE_FNO','B':'FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO','G':'FC_P064_AR5_RESET_K1_FRESH_FORCE_FNO'}

def test_e100_supervision_token_memory_ledger_and_cleanup(tmp_path):
    (tmp_path/'result.json').write_text('{}')
    class Base:
        observations=0;stopped=False
        def memory(self):self.observations+=1;return {'MemAvailable':100*2**30}
        def memory_ok(self,obs,*_):return obs['MemAvailable']>=22*2**30
        def stop(self,process):self.stopped=True
    class Process:
        returncode=0
        def __init__(self):self.calls=0
        def poll(self):self.calls+=1;return None if self.calls==1 else 0
    base=Base()
    def factory(command,**kwargs):
        assert kwargs['env']['REPLAY_CHILD_TOKEN']=='x'*64
        return Process()
    m.supervise(tmp_path,base,['worker'],{'REPLAY_CHILD_TOKEN':'x'*64},'a'*64,process_factory=factory,sleep_fn=lambda _:None)
    assert base.observations==2 and base.stopped
    assert len((tmp_path/'memory.jsonl').read_text().splitlines())==2

def test_actual_official_train_reader_fixed_eight_origins_and_clocks():
    import importlib.util
    if importlib.util.find_spec('physicsnemo') is None:
        import pytest
        pytest.skip('run separately with the reviewed curator Python')
    src=Path('/workspace/fluid_control/artifacts/p064_ar5_reset_g_development_source_20261007_immutable/src')
    sys.path.insert(0,str(src))
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    root=Path('/workspace/fluid_control/artifacts/b00_controlled_train_dataset_view_20261006')
    dataset=TandemRolloutDataset(root,'train',5,stride=1,num_workers=1,force_indices=(0,1,2,3))
    try:
        assert len(dataset.paths)==1 and dataset.paths[0].name=='b00_projected_ppo_train.h5'
        selected={step:index for index,(file_index,step) in enumerate(dataset.index) if file_index==0 and step in m.STARTS}
        assert set(selected)==set(m.STARTS)
        reader=dataset._reader(0)
        for start,index in selected.items():
            s,metadata=dataset[index];m.validate_sample(s,metadata)
            times=[float(reader[start+j][0]['time']) for j in range(6)]
            assert abs(times[0]-(148+start*.1))<1e-5
            assert all(abs(times[j]-(times[0]+j*.1))<1e-5 for j in range(6))
    finally:dataset.close()
