"""P015 cannot fall through the legacy single-policy CFD admission path."""
from pathlib import Path

import pytest
from test_full40_canonical_openfoam_feedback import MODULE


@pytest.mark.parametrize('marker', [
    'FC_P015_WINDOW_ACCUMULATION_FORCE_FNO',
    'FC_P015_POSTEVAL_COMPLETE',
    'fcp015_window_accumulation_dual_fno',
    'FC-P015',
    'FC_P013_INDEPENDENT_FORCE_FNO',
    'fcp013_independent_force_dual_fno',
])
@pytest.mark.parametrize('side', ['audit', 'readiness', 'gate'])
def test_dual_marker_without_binding_is_rejected(marker, side):
    evidence = dict(audit={}, readiness={}, gate={})
    evidence[side] = {'nested': [{'candidate_kind': marker}]}
    with pytest.raises(ValueError, match='dual_control_binding'):
        MODULE.validate_dual_policy_contract(
            **evidence, policy=Path('not_read.zip'), normalization_sha256='a' * 64,
            vecnormalize=None, expected_vecnormalize_sha256=None)


def test_single_policy_behavior_unchanged():
    assert MODULE.validate_dual_policy_contract(
        audit={}, readiness={}, gate={}, policy=Path('not_read.zip'),
        normalization_sha256='a' * 64, vecnormalize=None,
        expected_vecnormalize_sha256=None) == {}
