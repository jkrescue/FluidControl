import copy
import importlib.util
from pathlib import Path
import pytest

HERE = Path(__file__).parent
PATH = HERE / 'prepare_p064_candidate_evaluation.py'
if not PATH.exists():
    PATH = HERE.parent / 'scripts/prepare_p064_candidate_evaluation.py'
spec = importlib.util.spec_from_file_location('candidate_pending', PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def valid():
    return dict(status='FC_P064_ARM_A_TRAINING_COMPLETE_NOT_ADMISSION', arm='A', history_k=1,
        optimizer_steps=32, training_windows=256,
        records=[dict(update=i, consumed_windows=8*i) for i in range(1,33)],
        flow_tensor_sha256=m.FLOW_SHA, official_fresh_reload_verified=True,
        scientific_admission=False, trainer_sha256=m.TRAINER_SHA)

def test_actual_record_contract():
    m.validate_result(valid(), 'A')

@pytest.mark.parametrize('key,value', [('optimizer_steps',31),('training_windows',255),
    ('flow_tensor_sha256','0'*64),('official_fresh_reload_verified',False),
    ('engineering_fixture_not_candidate',True),('scientific_admission',True),
    ('arm','B'),('trainer_sha256','0'*64)])
def test_reject_incomplete_or_wrong_proof(key,value):
    r=valid();r[key]=value
    with pytest.raises(ValueError):m.validate_result(r,'A')

def test_intermediate_record_is_checked():
    r=valid();r['records'][9]['consumed_windows']=79
    with pytest.raises(ValueError):m.validate_result(r,'A')

def test_terminal_identity():
    p=dict(InvocationID='a'*32,MainPID='0',ExecMainStatus='0',Result='success',SubState='exited')
    m.validate_terminal(p,'a'*32)
    for key,value in [('MainPID','3'),('ExecMainStatus','1'),('InvocationID','b'*32)]:
        bad=copy.deepcopy(p);bad[key]=value
        with pytest.raises(ValueError):m.validate_terminal(bad,'a'*32)

def test_pending_only_no_model_construction():
    source=PATH.read_text()
    assert 'validate_dual_fno_manifest(' in source
    assert 'load_dual_fno(' not in source and 'allow_engineering_fixture=True' not in source
    assert "execution_authorized=False" in source
    assert "SOURCE / 'scripts'), str(SOURCE / 'src')" in source

def test_worker_exact_import_only_delta():
    import hashlib
    path=PATH.with_name('evaluate_p064_candidate_development_h1_h5.py')
    source=path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==m.CANDIDATE_WORKER_SHA
    old=source.replace("load_bound_k1 = module(bound(s['sources']['selector']), 'fluid_control.exploratory_short_mpc').load_bound_k1",
                       'from fluid_control.exploratory_short_mpc import load_bound_k1')
    assert hashlib.sha256(old.encode()).hexdigest()==m.WORKER_SHA
