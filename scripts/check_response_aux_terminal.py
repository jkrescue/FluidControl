"""E-only identity extension of reviewed B checker; CPU terminal audit, no forward."""
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

BASE = Path(__file__).with_name('check_p064_terminal_original.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == '8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587'
spec = importlib.util.spec_from_file_location('base', BASE)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
CAPTURE = {}
OUTPUT = Path('/workspace/fluid_control/artifacts/p064_response_aux_candidate_e_20261007')

def check_aux(result, protocol):
    assert result['auxiliary_profile'] == 'FC_P064_RESPONSE_AUX_K1_FRESH'
    records = result['auxiliary_records']
    assert len(records) == 32
    for r in records:
        assert set(r) == {'loss','coefficient','backward_multiplier','aerodynamic_calls','sample_count','flow_calls'}
        assert math.isfinite(r['loss']) and r['loss'] >= 0
        assert [r[k] for k in ('coefficient','backward_multiplier','aerodynamic_calls','sample_count','flow_calls')] == [1,8,1,12,0]
    p = protocol['auxiliary_response']
    assert p['phases'] == [0,2,4,6] and p['roles'] == ['m075','zero','p075']
    assert p['coefficient'] == 1 and p['updates'] == 32 and p['extra_aerodynamic_samples'] == 384
    assert p['pre_accumulator_divide_compensation'] == 8 and p['extra_flow_forward_calls'] == 0
    assert not p['validation_accessed'] and not p['frozen_test_accessed']
    expected = [f'matched_start_acquisition_train_b{phase:02d}_{role}' for phase in (0,2,4,6) for role in ('m075','zero','p075')]
    assert [r['case'] for r in result['auxiliary_input_records']] == expected
    assert all(r['frames'] == [0,1] for r in result['auxiliary_input_records'])
    return {'auxiliary_records':32,'evaluated_samples':384,'distinct_train_cases':12,'shared_q0_states':4,
            'aux_loss_finite':True,'independent_aux_loss_recomputed':False,
            'reason':'producer saves scalar losses, not auxiliary prediction arrays or separate loss journal'}

for name, before, after in (
    ('check_records', "f'FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION'", "'FC_P064_RESPONSE_AUX_TRAINING_COMPLETE_NOT_ADMISSION'"),
    ('checkpoint_cpu', "f'FC_P064_ARM_{arm}_CONTROLLED_AERO_CHECKPOINT'", "'FC_P064_RESPONSE_AUX_AERODYNAMIC_CHECKPOINT'"),
    ('main', 'f"FC_P064_ARM_{approval[\'arm\']}_DUAL_FNO_MANIFEST_VERIFIED"', "'FC_P064_RESPONSE_AUX_DUAL_FNO_MANIFEST_VERIFIED'"),
):
    source = inspect.getsource(getattr(base,name))
    assert source.count(before) == 1
    exec(source.replace(before,after), base.__dict__)

original_read = base.read
def read(path):
    value = original_read(path)
    name = Path(path).name
    if name in ('result.json','training_protocol.json','dual_model_manifest.json') and Path(path).resolve().parent == OUTPUT:
        CAPTURE[name] = value
    if name == 'P064_RESPONSE_AUX_E_TRAINING_APPROVAL_20261007.json':
        CAPTURE['approval'] = value
    return value
base.read = read

def main():
    base.main()  # terminal gate precedes all candidate/checkpoint access
    approval = CAPTURE['approval']; r = CAPTURE['result.json']; p = CAPTURE['training_protocol.json']
    evidence = check_aux(r,p)
    assert p['auxiliary_response'] == approval['protocol']['auxiliary_response']
    assert len(approval['overlay_sources']) == 3
    for path,digest in approval['overlay_sources'].items(): assert base.sha(Path(path)) == digest
    argv=approval['argv']; cli={argv[i][2:].replace('-','_'):argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
    receipt = original_read(cli['response_receipt'])
    assert base.sha(Path(cli['response_receipt'])) == approval['response_receipt_sha256']
    mapping={x['case']:x for x in receipt['rows']}
    for row in r['auxiliary_input_records']: assert row['hdf_sha256']==mapping[row['case']]['hdf_sha256']
    assert min(x['MemAvailable'] for x in r['resources']) >= 22
    assert CAPTURE['dual_model_manifest.json']['response_aux_sha256'] == cli['response_aux_sha256']
    evidence.update(overlay_sources_rehashed=3,original_B_schedule_retained=True,
                    result_sha256=base.sha(Path(approval['planned_output'])/'result.json'))
    print(json.dumps(evidence,indent=2,allow_nan=False))

if __name__ == '__main__': main()
