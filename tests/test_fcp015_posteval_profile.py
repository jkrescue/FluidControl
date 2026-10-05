import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p015_validator_test',ROOT/'scripts/validate_fcp008_posteval.py')
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def test_p015_profile_requires_exact_experiment_without_relaxing_old_profiles():
    v.configure_profile('p015')
    assert v.DUAL_PROFILE and v.ACCUMULATION_PROFILE and v.TRAINED_PROFILE
    assert v.CANDIDATE_KIND == v.DIAGNOSTIC_KIND == 'fcp015_window_accumulation_dual_fno'
    assert v.COMPLETE_STATUS == 'FC_P015_POSTEVAL_COMPLETE'
    fields=dict(training_experiment='FC-P015',training_windows=1368,accumulation_windows=8,optimizer_steps=171)
    v.validate_training_experiment(fields)
    for k in fields:
        bad={**fields,k:'wrong'}
        with pytest.raises(ValueError): v.validate_training_experiment(bad)
    for profile in ('p013','p011_head_only','p011_decoder_tail'):
        v.configure_profile(profile)
        assert not v.ACCUMULATION_PROFILE
        v.validate_training_experiment({'optimizer_steps':1368})
        with pytest.raises(ValueError): v.validate_training_experiment(fields)
    v.configure_profile('p008')
    assert not v.DUAL_PROFILE and not v.TRAINED_PROFILE


def test_runner_exact_protocol_dependency_closure_and_auditor_args():
    s=(ROOT/'scripts/run_fcp008_posteval_spark.sh').read_text()
    for name in ('audit_fcp015_candidate.py','audit_fcp013_dual_candidate.py','audit_fcp011_candidate.py',
                 'evaluate_fcp013_fixed_train_windows.py','train_fcp013_independent_force_fno.py',
                 'diagnose_fcp014_train_objective.py','verify_fcp015_dual_reload.py'):
        assert 'scripts/'+name in s
    assert '--execution-observation-sha256 "$FCP015_EXECUTION_OBSERVATION_SHA256"' in s
    assert 'posteval_fc_p015' in s
    assert 'EXECUTE_APPROVED_FC_P015_POSTEVAL' in s
    assert '--check-receipt' in s and 'dual_reload_receipt_sha256' in s
    assert '--segment-stride 25 --evaluation-batch-size 4' in s
    assert '--segment-stride 1 --evaluation-batch-size 8' in s
    assert s.count('"${dual_args[@]}"') == 3
    assert 'numerical_commit="7216214b545fbbd50b2fb5ed866f231039b06b18"' in s
    for name in ('run_fcp008_posteval_spark.sh','run_fcp015_posteval_spark.sh'):
        subprocess.run(['bash','-n',str(ROOT/'scripts'/name)],check=True)


def test_dev30_kind_must_match_actual_manifest(monkeypatch,tmp_path):
    import fluid_control.dual_fno as dual
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('p015_dev30_test',ROOT/'scripts/audit_dev30_validation_diagnostic.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for kind, actual in ((m.P015_KIND,'FC_P013_INDEPENDENT_FORCE_FNO'),
                         (m.P013_KIND,'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO')):
        monkeypatch.setattr(dual,'validate_dual_fno_manifest',lambda *a,**kw:SimpleNamespace(payload={'kind':actual}))
        with pytest.raises(ValueError,match='actual dual experiment'):
            m.validate_report_contract({}, {}, candidate_kind=kind,checkpoint_dir=tmp_path)


def test_training_outer_unit_requires_exact_clean_terminal():
    observation={'unit':{'InvocationID':'a'*32}}
    unit=dict(InvocationID='a'*32,Result='success',ExecMainCode='1',ExecMainStatus='0',MainPID='0',
              ActiveState='active',SubState='exited')
    v.validate_p015_unit_exit(observation,unit)
    with pytest.raises(ValueError):
        v.validate_p015_unit_exit(observation,{**unit,'ActiveState':'inactive','SubState':'dead'})
    for key,value in [('InvocationID','b'*32),('Result','exit-code'),('ExecMainCode','2'),
                      ('ExecMainStatus','1'),('MainPID','123'),('SubState','running')]:
        with pytest.raises(ValueError):v.validate_p015_unit_exit(observation,{**unit,key:value})
    with pytest.raises(ValueError):v.validate_p015_unit_exit(observation,{})
