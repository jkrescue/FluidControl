import copy
import importlib.util
from pathlib import Path
import pytest

p = Path(__file__).with_name('reproduce_canonical_closed_loop.py')
if not p.exists():
    p = Path(__file__).resolve().parents[1]/'scripts/reproduce_canonical_closed_loop.py'
s = importlib.util.spec_from_file_location('bg_entry', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)

def pair(digest=m.BASE_SHA):
    base = {'output': 'artifacts/old', 'lead_statement': 'old', 'inputs': {'policy': 'fixed'}}
    new = copy.deepcopy(base)
    new.update(output='artifacts/new', lead_statement='new authority', reproduction={
        'base_approval_sha256': digest, 'unit': 'fluid-control-canonical-reproduce-fixture.service',
        'execution_authorized': False})
    return base, new

def test_default_b_contract():
    assert m.PROFILES['b'] == {'base': m.BASE, 'base_sha': m.BASE_SHA, 'driver': m.DRIVER}
    base, new = pair()
    m.compare_approval(base, new, new['reproduction']['unit'])
    m.verify_profile_terminal(Path('/nonexistent'), 'b', m.PROFILES['b'])
    with pytest.raises(ValueError):
        m.execution_gate(new, True)

def test_unknown_profile():
    with pytest.raises(ValueError):
        m.profile_binding('arbitrary-model')

def test_wrong_policy_and_cross_profile_rejected():
    base, new = pair(m.PROFILES['g']['base_sha'])
    unit = new['reproduction']['unit']
    m.compare_approval(base, new, unit, m.PROFILES['g']['base_sha'])
    with pytest.raises(ValueError):
        m.compare_approval(base, new, unit)
    new['inputs']['policy'] = 'other-policy'
    with pytest.raises(ValueError):
        m.compare_approval(base, new, unit, m.PROFILES['g']['base_sha'])

def test_g_unreviewed_always_rejected():
    pending = dict(m.PROFILES['g'], verified_reproduction_accepted=False)
    with pytest.raises(ValueError, match='not bound'):
        m.verify_profile_terminal(Path('/nonexistent'), 'g', pending)
    profile = dict(m.PROFILES['g'], terminal_review=None)
    with pytest.raises(ValueError, match='missing'):
        m.verify_profile_terminal(Path('/nonexistent'), 'g', profile)

def test_default_cli_stays_b():
    assert "choices=tuple(PROFILES), default='b'" in p.read_text()
    assert m.ALLOWED == {'output', 'lead_statement', 'interpretation', 'reproduction'}

def test_g_evidence_must_match(monkeypatch):
    monkeypatch.setattr(m, 'sha', lambda _: 'wrong')
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.verify_profile_terminal(Path('/nonexistent'), 'g', m.PROFILES['g'])

def test_g_still_needs_new_authority():
    base,new=pair(m.PROFILES['g']['base_sha'])
    new['lead_statement']=base['lead_statement']
    with pytest.raises(ValueError, match='not new authority'):
        m.compare_approval(base,new,new['reproduction']['unit'],m.PROFILES['g']['base_sha'])
