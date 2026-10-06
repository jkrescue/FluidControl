import copy
import importlib.util
from pathlib import Path
import pytest

script = Path(__file__).resolve().parents[1] / 'scripts/check_h1_only_terminal.py'
spec = importlib.util.spec_from_file_location('checker', script)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def fixture():
    protocol = dict(training_experiment='FC-P064-H1-ONLY', arm='B',
        candidate_profile='FC_P064_H1_ONLY_K1_FRESH',
        objective='H1_only_half_equal_four_half_rearCl_normalized_MSE',
        diagnostic_objective='equal_H1_AR_half_equal_four_half_rearCl_normalized_MSE',
        mixed_forward_preserved=True, h1_gradient_multiplier_vs_b_component=2.0,
        ar_gradient_multiplier_vs_b_component=0.0, training_windows=256,
        optimizer_steps=32, accumulation_windows=8, b00_windows=64, b00_weight=.25,
        replacement_within_each_update=[0,4], validation_accessed=False,
        frozen_test_accessed=False, selection_performed=False)
    row = dict(training_objective=2., h1_balanced=2., ar_balanced=6., total=4.,
               diagnostic_total_semantics='unchanged_half_H1_half_AR')
    result = {'records':[{'records':[copy.deepcopy(row) for _ in range(8)]} for _ in range(32)]}
    manifest = dict(kind='FC_P064_H1_ONLY_K1_FRESH_FORCE_FNO', training_experiment='FC-P064-H1-ONLY')
    return result, protocol, manifest


def test_h1_not_diagnostic_total():
    evidence = checker.check_h1_objective(*fixture())
    assert evidence['mean_saved_training_objective'] == 2.
    assert evidence['checked_windows'] == 256
    assert not evidence['independent_training_loss_recomputed']


@pytest.mark.parametrize('key,value', [('training_objective',4.), ('total',2.),
    ('ar_balanced',float('nan')), ('diagnostic_total_semantics','backward')])
def test_reject_wrong_objective(key,value):
    result, protocol, manifest = fixture()
    result['records'][0]['records'][0][key] = value
    with pytest.raises(ValueError): checker.check_h1_objective(result,protocol,manifest)


def test_reject_ar_gradient_and_aux():
    result, protocol, manifest = fixture()
    protocol['ar_gradient_multiplier_vs_b_component'] = 1.
    with pytest.raises(ValueError): checker.check_h1_objective(result,protocol,manifest)
    result, protocol, manifest = fixture()
    result['auxiliary_records'] = []
    with pytest.raises(ValueError): checker.check_h1_objective(result,protocol,manifest)


def test_full_path_capture():
    capture = {}
    values = {'/a/approval.json':{'planned_output':'/candidate'},
              '/candidate/dual_model_manifest.json':{'candidate':True},
              '/parent/dual_model_manifest.json':{'candidate':False}}
    read = checker.capture_read(lambda p:values[p], '/a/approval.json',capture)
    for path in values: read(path)
    assert capture['dual_model_manifest.json']['candidate'] is True


def test_live_gate_still_rejects():
    with pytest.raises(ValueError):
        checker.base.terminal({'InvocationID':'x','MainPID':'99'},'x')


def test_original_schedule_unchanged():
    rows = checker.base.schedule(list(range(1368)), 'B')
    assert len(rows)==256
    assert sum(r['source']=='controlled_b00' for r in rows)==64
    assert rows[252]['b00_start']==700
