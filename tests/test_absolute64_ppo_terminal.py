import importlib.util
from pathlib import Path
import pytest

worker = Path(__file__).with_name('audit_absolute64_ppo_terminal.py')
if not worker.exists():
    worker = Path(__file__).resolve().parents[1] / 'scripts/audit_absolute64_ppo_terminal.py'
spec = importlib.util.spec_from_file_location('wrapper', worker)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

def fixture():
    base = wrapper.load_base('scripts/audit_canonical_ppo_terminal.py')
    status, kind, digest = wrapper.PROFILE
    result = dict(status=status, candidate_arm='ABSOLUTE64', candidate_manifest_kind=kind,
                  candidate_manifest_sha256=digest, fno_runtime=dict(profile='p026_k1',
                  manifest_kind=kind, dual_manifest_sha256=digest))
    approval = dict(candidate_manifest_kind=kind, candidate_manifest_sha256=digest,
                    inputs=dict(manifest=dict(sha256=digest)))
    return base, result, approval

def test_exact_identity_and_old_profiles_preserved():
    base, result, approval = fixture()
    base.identity(result, approval, 'ABSOLUTE64')
    assert set(base.PROFILES) == {'B', 'G', 'ABSOLUTE64'}

@pytest.mark.parametrize('field,value', [('candidate_arm','B'), ('candidate_arm','G'),
                                        ('candidate_manifest_sha256','0'*64)])
def test_wrong_identity_rejected(field, value):
    base, result, approval = fixture()
    result[field] = value
    with pytest.raises(AssertionError):
        base.identity(result, approval, 'ABSOLUTE64')

def test_wrong_base_rejected(tmp_path):
    path = tmp_path/'base.py'
    path.write_text('raise RuntimeError("must not import")')
    with pytest.raises(AssertionError):
        wrapper.load_base(path)
