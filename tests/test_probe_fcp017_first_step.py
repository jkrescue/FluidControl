import copy

import pytest
import torch

import probe_fcp017_first_step as p


class Toy(torch.nn.Module):
    def __init__(self,dtype):
        super().__init__()
        values=[1+2j,-3+.5j] if dtype.is_complex else [1.,-3.]
        self.weight=torch.nn.Parameter(torch.tensor(values,dtype=dtype))
        self.frozen=torch.nn.Parameter(torch.ones(1,dtype=dtype),requires_grad=False)


def initial(model): return {n:v.detach().clone() for n,v in model.state_dict().items()}


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32,torch.complex128,torch.complex64])
def test_adam_first_formula_real_component_clip_and_decay(dtype):
    model=Toy(dtype);before=initial(model)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    def run(i):
        loss=(model.weight.real.square()+model.weight.imag.square()).sum()*(i+1) if dtype.is_complex else model.weight.square().sum()*(i+1)
        loss.backward();return float(loss.detach())
    delta,report=p.first_step(model,opt,before,run,list(range(6)))
    g=p.real64(before['weight'])*7
    norm=torch.linalg.vector_norm(g)
    gc=g*min(1,1/(float(norm)+1e-6))
    ideal=-1e-5*gc/(gc.abs()+1e-8)-1e-9*p.real64(before['weight'])
    assert report['raw_mean_gradient_norm_float64']==pytest.approx(float(norm),rel=1e-7)
    assert report['clip_returned_norm']>1
    assert report['directional_dot']==pytest.approx(float((g*delta['weight']).sum()),rel=1e-7)
    torch.testing.assert_close(delta['weight'],ideal,atol=3e-7 if dtype in (torch.float32,torch.complex64) else 1e-14,rtol=0)
    assert torch.equal(model.frozen,before['frozen']) and model.frozen.grad is None
    assert int(opt.state[model.weight]['step'])==1
    assert 'observational' in report['formula_comparison']
    if dtype.is_complex:
        # Componentwise denominator differs from a common complex magnitude.
        assert abs(float(delta['weight'][0,0])-float(delta['weight'][0,1])) < 3e-7


@pytest.mark.parametrize('factor',[1.,1/64,-1/64])
def test_independent_restore_and_rounding(factor):
    model=Toy(torch.float32);before=initial(model)
    delta={'weight':torch.tensor([1e-8,-1e-5],dtype=torch.float64)}
    with torch.no_grad():model.weight.fill_(999)
    record=p.apply_displacement(model,before,delta,factor)
    expected=(p.real64(before['weight'])+factor*delta['weight']).float()
    assert torch.equal(model.weight,expected)
    assert record['disappeared_real_components']>=1
    p.restore(model,before)
    assert all(torch.equal(v,before[n]) for n,v in model.state_dict().items())


def test_complex_actual_dot_matches_conjugate():
    g=torch.tensor([1+2j,3-4j],dtype=torch.complex128)
    delta=torch.tensor([.2-.1j,.3+.4j],dtype=torch.complex128)
    assert float((p.real64(g)*p.real64(delta)).sum())==pytest.approx(float((g.conj()*delta).sum().real))


def test_reject_nonfresh_and_wrong_config():
    model=Toy(torch.float32);opt=torch.optim.AdamW(model.parameters(),lr=.1)
    with pytest.raises(ValueError):p.first_step(model,opt,initial(model),lambda _:None,[0]*6)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    opt.state[model.weight]['step']=torch.tensor(1)
    with pytest.raises(ValueError):p.first_step(model,opt,initial(model),lambda _:None,[0]*6)


def test_missing_gradient_fails_before_step():
    model=Toy(torch.float32);opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    with pytest.raises(FloatingPointError):p.first_step(model,opt,initial(model),lambda _:None,[0]*6)
    assert not opt.state


def test_no_scientific_pass_from_completion_or_repeat_spread():
    vals=dict(base1=1.,base2=1.,full=2.,plus=.9,minus=1.1,restored=1.)
    evaluations={name:dict(macro={d:v for d in ('h1_balanced','ar_balanced','total')}) for name,v in vals.items()}
    report=p.summarize(evaluations,dict(directional_dot=-1))
    assert not report['scientific_admission'] and not report['automatic_scientific_pass']
    assert report['domains']['total']['observed_repeat_spread']==0
    assert report['domains']['total']['central_directional_difference']==pytest.approx(-6.4)
    assert 'not a rigorous' in report['numerical_caveat']


def test_exact_mean_gradient_not_double_divided():
    model=Toy(torch.float64);opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    def run(i):
        loss=(model.weight*(i+1)).sum();loss.backward();return 0
    _,report=p.first_step(model,opt,initial(model),run,list(range(6)))
    assert report['raw_mean_gradient_norm_float64']==pytest.approx((2*3.5**2)**.5)
