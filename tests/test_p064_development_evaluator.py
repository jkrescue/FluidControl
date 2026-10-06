import ast
import os
from pathlib import Path
import pytest
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE if (HERE/'prepare_p064_development_evaluator.py').exists() else HERE.parent/'scripts'))
from prepare_p064_development_evaluator import derive

PARENT=Path(os.environ.get('P064_PARENT','/workspace/fluid_control/artifacts/projected_policy_h1_h5_inference_source_20261006_immutable/infer_projected_policy_h1_h5_replay.py'))

def functions(source):
    return {n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def test_exact_numerics_lifecycle_unchanged():
    old=PARENT.read_text();new=derive(old)
    before,after=functions(old),functions(new)
    for name in ('rollout','tensor_digest','verify_execution','supervise','main','bound','sha'):
        assert before[name]==after[name]
    assert "enumerate(('b01','b03'))" in new
    assert new.index('load_evaluation_pair(s,inputs')<new.index('precision = base.override_inference_precision(torch)')
    assert "'candidate_label':s['candidate_label']" in new

def test_parent_tampering():
    with pytest.raises(ValueError):derive(PARENT.read_text()+'\n')

def test_loader_is_explicit_no_kind_alias():
    source=derive(PARENT.read_text());tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_evaluation_pair')
    ns={'json':__import__('json'),'require':lambda yes,why:yes or (_ for _ in ()).throw(ValueError(why))}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<test>','exec'),ns)
    result=ns['load_evaluation_pair']({'candidate_label':'K1'},None,None,None,None,None,lambda *a,**k:('flow','aero','identity'))
    assert result==('flow','aero','identity')

@pytest.mark.parametrize('label',['A','B'])
def test_arm_load_binding_and_reject_wrong_kind(tmp_path,label):
    import json
    from types import SimpleNamespace
    source=derive(PARENT.read_text());tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_evaluation_pair')
    ns={'json':json,'require':lambda yes,why:yes or (_ for _ in ()).throw(ValueError(why))}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<test>','exec'),ns)
    kind='FC_P064_ARM_'+label+'_CONTROLLED_AERO_FORCE_FNO'
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'kind':kind}))
    calls=[]
    def loader(*a,**k):
        calls.append(k)
        return SimpleNamespace(flow_model='flow',aerodynamic_model='aero'),SimpleNamespace(payload={'kind':kind})
    spec={'candidate_label':label,'inputs':{'manifest':{'sha256':'a'*64}}}
    result=ns['load_evaluation_pair'](spec,manifest,None,None,loader,None,None)
    assert result[:2]==('flow','aero') and calls[0]['expected_manifest_sha256']=='a'*64
    manifest.write_text(json.dumps({'kind':'FC_P026_K1_HISTORY_FORCE_FNO'}))
    with pytest.raises(ValueError):ns['load_evaluation_pair'](spec,manifest,None,None,loader,None,None)

def test_fixed_phase_panel_receipt(tmp_path):
    import json,hashlib
    from prepare_p064_development_evaluator import RECORDS
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ns={'json':json,'Path':Path,'sha':sha,'require':lambda yes,why:yes or (_ for _ in ()).throw(ValueError(why))}
    exec(RECORDS,ns)
    selection={'phases':{}}
    rows=[]
    for phase in ('b01','b03'):
        selection['phases'][phase]={'split':'development_already_opened','frames':48,'records':[]}
        for start in range(0,701,100):
            selection['phases'][phase]['records'].append({'branch':'mpc','phase':phase,'start_index':start})
            path=tmp_path/f'{phase}_{start}.h5';path.write_bytes(b'fixture not real HDF')
            rows.append({'phase':phase,'start_index':start,'frames':6,'official_reader_verified':True,'path':str(path),'sha256':sha(path),'mask_sha256':'m','x_sha256':'x','y_sha256':'y'})
    sp=tmp_path/'selection.json';sp.write_text(json.dumps(selection))
    receipt={'status':'DEVELOPMENT_PHASE_CONVERSION_COMPLETE_NOT_ADMISSION','frames':96,'trajectories':16,'endpoints':80,'split':'development_already_opened','source_unchanged':True,'owned_containers_cleaned':True,'model_loaded':False,'selection_sha256':sha(sp),'hdf':rows}
    rp=tmp_path/'result.json';rp.write_text(json.dumps(receipt))
    records,hdfs=ns['conversion_records']({'conversion_result':rp,'selection':sp})
    assert len(records)==16 and records[0]['branch']=='b01' and records[-1]['branch']=='b03'
    receipt['hdf'].reverse();rp.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):ns['conversion_records']({'conversion_result':rp,'selection':sp})
