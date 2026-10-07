import math
from pathlib import Path

import numpy as np
import pytest

from fluid_control import p064_coarse_force_recoverability as m


TEXT = """
forceFront
{
 scalar
 {
  Cd 3; CdPressure 1; CdViscous 2;
  Cl 7; ClPressure 3; ClViscous 4;
 }
}
forceRear
{
 scalar
 {
  Cd 11; CdPressure 5; CdViscous 6;
  Cl 15; ClPressure 7; ClViscous 8;
 }
}
"""


def foam_text(front=(3,1,2,7,3,4), rear=(11,5,6,15,7,8)):
    def body(name, v):
        return f"""{name}\n{{\n scalar\n {{\n Cd {v[0]};\n CdPressure {v[1]};\n CdViscous {v[2]};\n Cl {v[3]};\n ClPressure {v[4]};\n ClViscous {v[5]};\n }}\n}}\n"""
    return body("forceFront",front)+body("forceRear",rear)


def test_component_parser_alignment_and_fail_closed():
    parsed=m.parse_force_components(foam_text())
    assert m.component_vectors(parsed)[0].tolist()==[3,7,11,15]
    out=m.validate_alignment(np.float32(148.1),148.1,np.asarray([3,7,11,15],np.float32),parsed)
    assert out['component_max_abs']==0
    with pytest.raises(ValueError,match='time'):
        m.validate_alignment(np.float32(148.2),148.1,np.asarray([3,7,11,15]),parsed)
    with pytest.raises(ValueError,match='pressure'):
        m.validate_alignment(148.1,148.1,np.asarray([3,7,11,15]),m.parse_force_components(foam_text(front=(4,1,2,7,3,4))))
    with pytest.raises(ValueError,match='incomplete'):
        m.parse_force_components(foam_text().replace('CdViscous 2;',''))
    with pytest.raises(ValueError,match='nonfinite'):
        m.parse_force_components(foam_text().replace('CdViscous 2;','CdViscous nan;'))
    with pytest.raises(ValueError,match='exactly one'):
        m.parse_force_components(foam_text()+foam_text())
    with pytest.raises(ValueError,match='exactly one'):
        m.parse_force_components(foam_text().replace(' scalar\n',' scalar\n {}\n scalar\n',1))


def grid(n=301):
    x=np.linspace(-1.5,1.5,n);y=np.linspace(-1.5,1.5,n)
    yy,xx=np.meshgrid(y,x,indexing='ij')
    mask=((xx*xx+yy*yy)>.25).astype(np.uint8)
    return x,y,xx,yy,mask


def test_constant_gauge_invariant_and_cos_sin_signs():
    x,y,xx,yy,mask=grid()
    g=m.ForceGeometry(0,0,rho=1,u_inf=1,area_ref=.1)
    constant=m.coarse_pressure_coefficients(np.full_like(xx,3.7),mask,x,y,g)
    assert np.max(np.abs(constant)) < 1e-12
    h=max(np.diff(x).max(),np.diff(y).max());rp=.5+2*h
    cd=m.coarse_pressure_coefficients(xx/rp,mask,x,y,g)
    cl=m.coarse_pressure_coefficients(yy/rp,mask,x,y,g)
    expected=-math.pi*g.span*g.radius/(.5*g.area_ref)
    assert np.allclose(cd,[expected,0],rtol=2e-4,atol=1e-10)
    assert np.allclose(cl,[0,expected],rtol=2e-4,atol=1e-10)


def test_kinematic_pressure_rho_restored_without_changing_coefficient():
    x,y,xx,yy,mask=grid()
    h=max(np.diff(x).max(),np.diff(y).max());rp=.5+2*h
    a=m.coarse_pressure_coefficients(xx/rp,mask,x,y,m.ForceGeometry(0,0,rho=1))
    b=m.coarse_pressure_coefficients(xx/rp,mask,x,y,m.ForceGeometry(0,0,rho=4,kinematic_pressure=True))
    assert np.allclose(a,b,rtol=0,atol=1e-12)


def test_solid_stencil_rejected():
    x,y,xx,yy,mask=grid(51)
    mask[:]=0
    with pytest.raises(ValueError,match='solid'):
        m.coarse_pressure_coefficients(xx,mask,x,y,m.ForceGeometry(0,0))


def test_reader_reads_only_fixed_indices_and_closes():
    calls=[]
    class Reader:
        def __init__(self,path,fields):
            assert fields==['state','mask','omega','force','time'];calls.append(('open',Path(path)))
        def __getitem__(self,index):
            calls.append(('get',index));return {'time':index}, {'i':index}
        def close(self):calls.append(('close',))
    rows=m.read_selected_frames(Reader,Path('/fixed.h5'),(0,100,800))
    assert [r[0] for r in rows]==[0,100,800]
    assert calls==[('open',Path('/fixed.h5')),('get',0),('get',100),('get',800),('close',)]


def test_fixed_selection_is_not_mutable():
    assert m.CASES == ('matched_start_acquisition_train_b00_m075','matched_start_acquisition_train_b00_zero','matched_start_acquisition_train_b00_p075')
    assert m.FRAME_INDICES == (0,100,200,300,400,500,600,700,800)
    assert m.NTHETA == 128 and m.TIME_ATOL == 2e-5
