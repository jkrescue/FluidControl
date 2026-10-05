"""Software-only recovery evidence checks; not CFD accuracy tests."""
import json
import pytest
from audit_fcp013_dual_candidate import validate_recovery_watch


def sample(**changes):
    value = dict(timestamp=100, mem_available_kib=100*1024**2,
                 mem_free_kib=30*1024**2, log_age_seconds=10)
    value.update(changes)
    return json.dumps(value)


def test_resource_watch_samples_are_physically_separate():
    value = validate_recovery_watch(sample() + '\n' + sample(timestamp=110))
    assert value == dict(samples=2, min_mem_available_gib=100, min_mem_free_gib=30)


@pytest.mark.parametrize('changes', [dict(mem_free_kib=19*1024**2),
    dict(mem_available_kib=19*1024**2), dict(log_age_seconds=301),
    dict(log_age_seconds=-1), dict(timestamp=float('nan')),
    dict(event='recovery_resource_or_progress_stop')])
def test_resource_or_progress_failure_is_not_completion(changes):
    with pytest.raises(ValueError):
        validate_recovery_watch(sample(**changes))


def test_empty_or_duplicate_watch_rejected():
    for text in ('', sample()+'\n'+sample()):
        with pytest.raises(ValueError):
            validate_recovery_watch(text)
