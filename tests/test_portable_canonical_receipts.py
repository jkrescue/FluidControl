"""Receipt serialization fixtures only; no numerical gate is recomputed here."""
import pytest
from produce_canonical_surrogate_compatibility_gates import portable_receipt_paths, sha256


def receipt(root):
    result = {'status': 'UNCHANGED', 'metric_pass': False, 'measured_error': 0.123}
    for name in ('producer_script', 'protocol_path', 'evidence_path'):
        path = root/name
        path.write_text(name)
        result[name] = str(path)
        result[{'producer_script':'producer_script_sha256', 'protocol_path':'protocol_sha256',
                'evidence_path':'evidence_sha256'}[name]] = sha256(path)
    return result


def test_only_three_paths_change_and_metrics_hashes_stay_identical(tmp_path):
    original = receipt(tmp_path)
    portable = portable_receipt_paths(original, tmp_path)
    keys = {'producer_script', 'protocol_path', 'evidence_path'}
    assert {k for k in original if original[k] != portable[k]} == keys
    assert all(portable[k] == k for k in keys)
    assert portable['metric_pass'] is False


def test_outside_root_or_changed_source_rejected(tmp_path):
    sub = tmp_path/'repo'; sub.mkdir()
    original = receipt(sub)
    other = tmp_path/'outside'; other.write_text('outside')
    broken = dict(original, evidence_path=str(other), evidence_sha256=sha256(other))
    with pytest.raises(ValueError):
        portable_receipt_paths(broken, sub)
    (sub/'evidence_path').write_text('changed')
    with pytest.raises(ValueError, match='bytes differ'):
        portable_receipt_paths(original, sub)
