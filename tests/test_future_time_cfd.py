import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scripts/run_p064_b_future_time_cfd.py'
spec=importlib.util.spec_from_file_location('future_cfd',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_pair_copies_only_real_new_restart(tmp_path):
    src=tmp_path/'source';out=tmp_path/'out';out.mkdir()
    for name in ('constant','system','248','228','postProcessing'):
        (src/name).mkdir(parents=True);(src/name/'fixture').write_text(name)
    cases=m.copy_pair_at_248(src,out)
    for p in cases.values():
        assert {x.name for x in p.iterdir()}=={'constant','system','248'}
        assert (p/'248/fixture').read_text()=='248'


def test_baseline_200_zero_segments_and_real_endpoint_observation(tmp_path):
    parent=tmp_path/'parent';out=tmp_path/'out';out.mkdir()
    for name in ('228','constant','system'): (parent/name).mkdir(parents=True)
    actions=[];checks=[];queries=[];clock=[228.];writes=[]
    class Solver:
        def __init__(self,cases,*args): self.source=cases['baseline']
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def pair(self,step,end,check,guard):
            checks.append((step,end));clock[0]=end
            return {'baseline':{'steps':20,'solver_ended_cleanly':True}}
    def observe(source,t,w):
        queries.append((t,w));p=source/'new_probes';p.write_text('real fixture at248')
        return np.arange(69,dtype=np.float32)*0,dict(time=t,probe_sources=[str(p)],force_sources=[str(p)],front_force_sources=[str(p)])
    base=SimpleNamespace(PairSolvers=Solver,atomic_json=lambda p,r,**k:writes.append((p,r)),tree=lambda p:{'fixture':'digest'})
    transport=SimpleNamespace(substitute=lambda *a:None,latest_time=lambda p:clock[0],configure_interval=lambda p,b,e,x,y:actions.append((b,e,x,y)),check_segment=lambda *a:None,total_drag_observation_at=observe)
    _,obs,_,record=m.prepare_future_restart(parent,out,base,transport,'image',lambda:None)
    assert len(actions)==len(checks)==200 and actions[0]==(228.,228.1,0.,0.) and actions[-1]==(247.9,248.,0.,0.)
    assert queries==[(248.,0.)] and record['zero_cycles']==200 and len(obs)==69
    assert len(writes)==201 and record['initial_observation_source_sha256']


def test_missing_endpoint_is_not_zero_filled(tmp_path):
    # The production helper calls the exact-time reader directly and does not catch its failure.
    tree=ast.parse(SOURCE.read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prepare_future_restart')
    assert not any(isinstance(x,ast.Try) for x in ast.walk(f))
    call=next(x for x in ast.walk(f) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='total_drag_observation_at')
    assert [ast.literal_eval(x) for x in call.args[1:]]==[248.,0.]


def test_original_control_math_unchanged():
    original=Path('/workspace/fluid_control/scripts/run_p064_symmetry_canonical_b_ppo_long_cfd.py').read_text()
    def funcs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(original),funcs(SOURCE.read_text())
    for name in ('observation','validate_training','training_bindings','validate_symmetry_binding','validate_vec','predict','fixed_window','values_summary'):
        assert a[name]==b[name],name
    old=original.split('for step in range(1, 801):',1)[1].split('windows, raw_sha',1)[0]
    new=SOURCE.read_text().split('for step in range(1, 801):',1)[1].split('windows, raw_sha',1)[0]
    assert old.replace('148+','248+')==new


def test_six_windows_shift_only():
    text=SOURCE.read_text()
    for phrase in ["'early_12p4',248.,260.4", "'early_first_6p2',248.,254.2", "'early_trailing_6p2',254.2,260.4", "'primary_final_60',268.,328.", "'historical_inclusive_final_60',268.,328.", "'full_80',248.,328."]:
        assert phrase in text
    assert "'original_mean_bias_reference': .10" in text


def test_pending_requires_explicit_authorization():
    pending=ROOT/'PENDING.json'
    if not pending.exists():pending=ROOT/'docs/P064_B_FUTURE_TIME_CFD_PENDING_20261007.json'
    s=json.loads(pending.read_text())
    with pytest.raises(ValueError,match='explicit approval'):m.validate_spec(s)
    assert s['baseline_start_time']==228 and s['start_time']==248 and s['steps']==800
    assert s['future_time_protocol']['select_start_after_observation'] is False
