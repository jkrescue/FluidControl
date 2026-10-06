import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts/diagnose_projected_policy_h1_h5_replay.py"
spec = importlib.util.spec_from_file_location("projected_replay", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def observation(forces, omega):
    value = np.zeros(69, dtype=np.float64)
    value[64:68] = forces
    value[68] = omega
    return value.tolist()


def progress():
    rows = []
    previous = 0.0
    for index in range(800):
        applied = float(np.clip(np.sin(index / 20) * .6, previous - .1, previous + .1))
        rows.append({"step": index + 1, "start_time": 148 + .1 * index,
                     "end_time": 148 + .1 * (index + 1),
                     "requested_omega": applied, "applied_omega": applied,
                     "input_observation": observation([10+index]*4, previous),
                     "output_observation": observation([20+index]*4, applied),
                     "zero_observation": observation([30+index]*4, 0.0)})
        previous = applied
    return {"completed_cycles": 800, "rows": rows}


def packet(time, pressure_offset=0.0):
    mask = np.ones((1,128,256), dtype=np.uint8)
    state = np.zeros((3,128,256), dtype=np.float32)
    state[0] = np.float32(time); state[1] = np.float32(-time)
    state[2,0,0] = np.float32(1 + pressure_offset)
    state[2,0,1] = np.float32(-1 + pressure_offset)
    state[2] -= state[2].mean(dtype=np.float64)
    return {"state":state,"mask":mask,"time":np.asarray([time],dtype=np.float64),
            "x":np.linspace(8,25,256,dtype=np.float32),
            "y":np.linspace(4,11,128,dtype=np.float32)}


def test_selection_is_fixed_96_frames_and_branch_aware_forces():
    document = progress()
    selection = m.make_selection(document, Path("/source"), require_files=False)
    assert selection["starts"] == [0,100,200,300,400,500,600,700]
    assert selection["frames"] == 96 and selection["endpoints"] == 80
    records = {(row["branch"],row["start_index"]):row for row in selection["records"]}
    assert records[("mpc",100)]["initial_force"] == [110.0]*4
    assert records[("zero",100)]["initial_force"] == [129.0]*4
    assert records[("zero",100)]["transitions"][0]["truth_force"] == [130.0]*4
    assert records[("zero",100)]["transitions"][0]["omega_now"] == 0
    assert records[("mpc",100)]["transitions"][0]["omega_now"] == pytest.approx(
        document["rows"][99]["applied_omega"])
    assert records[("mpc",100)]["transitions"][0]["omega_next"] == pytest.approx(
        document["rows"][100]["applied_omega"])


def test_progress_rejects_requested_action_as_executed_or_broken_continuity():
    document=progress();document["rows"][20]["input_observation"][68]+=.01
    with pytest.raises(ValueError,match="input action continuity"):
        m.validate_progress(document)
    document=progress();document["rows"][20]["applied_omega"]=.75
    with pytest.raises(ValueError,match="executed action bounds"):
        m.validate_progress(document)


def test_selection_requires_actual_u_and_p_files(tmp_path):
    with pytest.raises(ValueError,match="selected saved field missing"):
        m.make_selection(progress(),tmp_path)


def test_packet_binds_grid_mask_time_and_centered_pressure():
    p=packet(148.0);state,mask=m.validate_packet(p,expected_time=148.0)
    assert state.shape==(3,128,256) and mask.shape==(1,128,256)
    bad=packet(148.0);bad["x"][2]+=.1
    with pytest.raises(ValueError,match="x"):
        m.validate_packet(bad,expected_time=148.0)
    bad=packet(148.0);bad["state"][2]+=.1
    with pytest.raises(ValueError,match="ROI centered"):
        m.validate_packet(bad,expected_time=148.0)


def test_free_ar_uses_only_initial_truth_and_all_five_recorded_action_pairs():
    states=[np.full((3,128,256),i,dtype=np.float32) for i in range(6)]
    transitions=[{"omega_now":i/100,"omega_next":i/100+.01} for i in range(5)]
    seen=[]
    def predict(current,mask,now,nxt):
        seen.append((current.copy(),now,nxt))
        return current+np.float32(1),np.asarray([now,nxt,now,nxt],np.float32)
    result=m.replay_one(states,np.ones((1,128,256),np.uint8),transitions,predict)
    assert [float(row[0,0,0]) for row in result["predicted_states"]]==[1,2,3,4,5]
    assert [(float(now),float(nxt)) for _,now,nxt in seen]==[
        (float(np.float32(i/100)),float(np.float32(i/100+.01))) for i in range(5)]
    poisoned=[states[0]]+[np.full_like(states[0],999) for _ in range(5)]
    assert np.array_equal(m.replay_one(poisoned,np.ones((1,128,256),np.uint8),transitions,
        lambda q,mask,now,nxt:(q+np.float32(1),np.zeros(4,np.float32)))["predicted_states"][-1],
        result["predicted_states"][-1])


def test_mini_hdf_roundtrip_contract_and_all_masks_equal(tmp_path):
    record=m.make_selection(progress(),Path('/source'),require_files=False)["records"][0]
    packets=[packet(frame["time"]) for frame in record["frames"]]
    def reader(path):
        with h5py.File(path,'r') as handle:
            fields={key:torch.from_numpy(handle[key][:]) for key in
                    ('state','mask','omega','force','time')}
            static={key:handle[key][:] for key in ('x','y')}
        return fields,static
    receipt=m.pack_and_verify_mini_hdf(record,packets,tmp_path/'one.h5',reader)
    assert receipt["official_reader_verified"] and receipt["frames"]==6
    with h5py.File(tmp_path/'one.h5','r') as handle:
        assert handle['state'].shape==(6,3,128,256)
        assert handle['omega'].dtype==np.float32 and handle['force'].shape==(6,4)
        assert np.array_equal(handle['force'][0],np.asarray(record['initial_force'],np.float32))
    bad=[dict(row) for row in packets];bad[3]=dict(bad[3]);bad[3]['mask']=bad[3]['mask'].copy();bad[3]['mask'][0,0,0]=0
    bad[3]['state']=bad[3]['state'].copy();bad[3]['state'][:,0,0]=0;bad[3]['state'][2,0,1]=0
    with pytest.raises(ValueError,match="mask changed"):
        m.pack_and_verify_mini_hdf(record,bad,tmp_path/'bad.h5',reader)


def test_pooled_metrics_use_sse_and_separate_force_persistence():
    mask=np.ones((1,128,256),np.uint8)
    truth=[np.ones((3,128,256),np.float32)*i for i in range(1,6)]
    record={"mask":mask,"truth_states_initial":np.zeros((3,128,256),np.float32),
      "truth_states":truth,"predicted_states":[row*.5 for row in truth],
      "truth_force_initial":np.zeros(4),
      "truth_forces":[np.ones(4)*i for i in range(1,6)],
      "predicted_forces":[np.ones(4)*(i+.25) for i in range(1,6)]}
    summary=m.sufficient_statistics([record,record])
    assert summary['5']['velocity_relative_l2']==pytest.approx(.5)
    assert summary['5']['force_channel_mae']==pytest.approx([.25]*4)
    assert summary['5']['persistence_force_channel_mae']==pytest.approx([5.]*4)
    assert summary['5']['segments']==2
    assert len(summary['5']['field_squared_error_sums_u_v_p'])==3

def test_total_drag_mae_allows_front_rear_error_cancellation():
    mask=np.ones((1,128,256),np.uint8);state=np.ones((3,128,256),np.float32)
    truth=np.asarray([1.,2.,3.,4.]);pred=np.asarray([2.,2.,2.,4.])
    record={"mask":mask,"truth_states_initial":state,"truth_states":[state]*5,
      "predicted_states":[state]*5,"truth_force_initial":truth,
      "truth_forces":[truth]*5,"predicted_forces":[pred]*5}
    row=m.sufficient_statistics([record])['1']
    assert row['force_channel_mae']==[1.,0.,1.,0.]
    assert row['total_drag_mae']==0
    assert row['persistence_total_drag_mae']==0
