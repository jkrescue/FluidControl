import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p064_audit_test', ROOT/'scripts/audit_fcp064_candidate.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def fixture():
    files = {'candidate/result.json': 'a'*64}
    receipt = dict(status='P064_TERMINAL_ENGINEERING_REVIEW_NOT_ADMISSION', result_sha256='a'*64,
        records=32, consumed=256, producer_official_reload=True, independent_model_reload=False,
        checkpoint=dict(adam_states=28, all_steps=32, weights_only_cpu=True),
        unit=dict(InvocationID=audit.PRIOR['B'][1], MainPID='0', Result='success', ExecMainStatus='0'))
    result = dict(status='FC_P064_ARM_B_TRAINING_COMPLETE_NOT_ADMISSION', arm='B', history_k=1,
        optimizer_steps=32, training_windows=256, scientific_admission=False, official_fresh_reload_verified=True)
    return receipt, result, files

def test_actual_contract():
    receipt, result, files = fixture()
    audit.validate_prior(receipt, result, 'B', files)

@pytest.mark.parametrize('key,value', [('arm','A'),('optimizer_steps',171),('training_windows',1368),('scientific_admission',True),('history_k',True)])
def test_wrong_contract(key, value):
    receipt, result, files = fixture()
    result[key] = value
    with pytest.raises(ValueError): audit.validate_prior(receipt, result, 'B', files)

def test_seven_files(tmp_path):
    for name in audit.FILES:
        path=tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'toy')
    assert set(audit.candidate_files(tmp_path)) == {'candidate/'+name for name in audit.FILES}
    (tmp_path/'result.json').unlink()
    (tmp_path/'result.json').symlink_to(tmp_path/'training_protocol.json')
    with pytest.raises(ValueError):audit.candidate_files(tmp_path)

def test_reload_is_k1_cpu_no_forward():
    source=(ROOT/'scripts/verify_fcp064_dual_reload.py').read_text()
    assert 'actual_optimizer_steps=32, actual_training_windows=256' in source
    assert 'choices=(1,)' in source
    assert 'torch.set_float32_matmul_precision("high")' in source
    assert 'dual.load_dual_fno(args.root/"dual_model_manifest.json"' in source
    assert 'observed == expected["tensor_sha256"]' in source
    assert 'not torch.cuda.is_initialized()' in source
