import numpy as np
import pytest
import audit_p064_b04_raw_terminal as a

def test_actual_component_parser_import():
    module=a.load('b04_test_actual_component_parser', a.ROOT/'src/fluid_control/p064_coarse_force_recoverability.py')
    assert callable(module.parse_force_components)
    assert callable(module.component_vectors)

def test_full_clock_rejects_missing_duplicate():
    t=120+.1*np.arange(801);a.clock(t,120,801,.1)
    with pytest.raises(AssertionError):a.clock(t[:-1],120,801,.1)
    t[1]=t[0]
    with pytest.raises(AssertionError):a.clock(t,120,801,.1)

def test_components_are_not_front_rear_pseudocomponents():
    assert a.check_components([3,3,3,3],[1,1,1,1],[2,2,2,2])==0
    with pytest.raises(AssertionError):a.check_components([3,3,3,3],[1,1,1,1],[3,3,3,3])
