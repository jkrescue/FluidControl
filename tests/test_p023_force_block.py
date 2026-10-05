import copy
import importlib.util
from pathlib import Path
import pytest
import torch
import p023_force_block as p

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lift=torch.nn.Conv2d(12,24,1,dtype=torch.float64)
        with torch.no_grad():self.lift.weight[:,6:10].zero_();self.lift.weight.mul_(.3)
    def forward(self,x):return self.lift(x)[:,:4].tanh()

class ComplexToy(Toy):
    def __init__(self):
        super().__init__();self.gain=torch.nn.Parameter(torch.tensor(.9+.2j,dtype=torch.complex128))
    def forward(self,x):return (super().forward(x).to(torch.complex128)*(1+1j)*self.gain).real

def adapter(complex=False):
    torch.manual_seed(18);return p.FrozenForceBlock(ComplexToy() if complex else Toy(),lift_name='lift.weight')

def test_zero_and_materialized_gradient():
    a=adapter();x=torch.randn(2,12,2,2,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(a(x),a.base(x),rtol=0,atol=0)
    with torch.no_grad():a.block.fill_(.04)
    material=copy.deepcopy(a.base)
    with torch.no_grad():material.lift.weight.copy_(a.effective_weight())
    material.lift.weight.requires_grad_(True);y=x.detach().clone().requires_grad_(True)
    out=a(x);reference=material(y)
    out.square().sum().backward();reference.square().sum().backward()
    torch.testing.assert_close(out,reference,rtol=0,atol=0)
    torch.testing.assert_close(a.block.grad,material.lift.weight.grad[:,6:10],rtol=0,atol=0)
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0);a.verify_frozen()

@pytest.mark.parametrize('lr',[1.5625e-7,1e-5])
def test_only96_optimizer_with_decay_and_restore(lr):
    a=adapter();old=a.diagnostic_identity();o=torch.optim.AdamW([a.block],lr=lr,weight_decay=.7)
    p.assert_optimizer_scope(a,o)
    a(torch.ones(2,12,1,1,dtype=torch.float64)).square().sum().backward();before=a.block.detach().clone();o.step()
    p.assert_optimizer_scope(a,o,step=1);assert a.block.numel()==96
    assert a.diagnostic_identity()['base_tensor_sha256']==old['base_tensor_sha256']
    assert p.block_update_metrics(before,a.block)['update_l2']>0
    a.restore_zero();fresh=torch.optim.AdamW([a.block],lr=lr);p.assert_optimizer_scope(a,fresh)
    assert a.diagnostic_identity()==old

def test_wrong_optimizer_and_nonfinite_rejected():
    a=adapter();wrong=torch.optim.AdamW(list(a.parameters()))
    with pytest.raises(ValueError):p.assert_optimizer_scope(a,wrong)
    with torch.no_grad():a.block[0,0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):a(torch.ones(1,12,1,1,dtype=torch.float64))

def test_old_inplace_and_data_mutation_rejected():
    a=adapter()
    with torch.no_grad():a.base.lift.bias.add_(1)
    with pytest.raises(RuntimeError):a.verify_frozen()
    b=adapter();b.base.lift.bias.data.add_(1)
    with pytest.raises(RuntimeError):b.verify_frozen()

def test_effective_only_force_columns_change():
    a=adapter();before=a.effective_weight().detach().clone()
    with torch.no_grad():a.block.fill_(.2)
    after=a.effective_weight();torch.testing.assert_close(after[:,:6],before[:,:6],rtol=0,atol=0);torch.testing.assert_close(after[:,10:],before[:,10:],rtol=0,atol=0)
    assert torch.all(after[:,6:10]==.2)

@pytest.fixture
def causal():
    path=Path('/workspace/fluid_control/scripts/p021_causal_force.py')
    if not path.exists():pytest.skip('pinned Main module absent')
    assert p.sha(path)=='bcefcad2fa622e5b69133725aa8d39db5a1b0a41a1b5e6c7dcb1f6b7d99a4962'
    s=importlib.util.spec_from_file_location('p023_test_p021',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def data():
    torch.manual_seed(3)
    return [torch.randn(100,6,1,1,dtype=torch.float64)*.1,torch.randn(100,6,1,1,dtype=torch.float64)*.1,torch.ones(1,1,1,1,dtype=torch.float64),torch.randn(101,4,dtype=torch.float64)*.1,torch.randn(100,4,dtype=torch.float64)*.1]

def predict(model,x,mask):return None,model(torch.cat((x,torch.zeros(x.shape[0],2,*x.shape[-2:],dtype=x.dtype)),1)).mean((-2,-1))

def balanced(x,y):
    z=(x-y).square().mean((0,1));return {'balanced':.5*z.mean()+.5*z[3]}

@pytest.mark.parametrize('complex',[False,True])
def test_full_h100_checkpoint_causal_and_ablation(causal,complex):
    a=adapter(complex)
    with torch.no_grad():a.block.fill_(.03)
    b=adapter(complex)
    with torch.no_grad():b.block.copy_(a.block)
    d=data();initial=a.diagnostic_identity()
    x=causal.recurrent_objective(a,*d,predict,balanced,checkpointed=True);y=causal.recurrent_objective(b,*d,predict,balanced,checkpointed=False)
    x['total'].backward();y['total'].backward();torch.testing.assert_close(a.block.grad,b.block.grad,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(x['ar'],y['ar'],rtol=0,atol=0)
    d2=[v.clone() for v in d];d2[3][1:]+=20;d2[4]+=10
    with torch.no_grad():
        z=causal.recurrent_objective(a,*d2,predict,balanced,checkpointed=False)
        torch.testing.assert_close(x['ar'],z['ar'],rtol=0,atol=0)
        d2[3][0]+=1;z2=causal.recurrent_objective(a,*d2,predict,balanced,checkpointed=False)
        assert not torch.equal(x['ar'][0],z2['ar'][0])
        rng=torch.get_rng_state();grad=a.block.grad.clone();ab=causal.recurrent_objective(a,*d,predict,balanced,zero_conditioning=True,checkpointed=False)
        assert not torch.equal(x['ar'],ab['ar']);assert torch.equal(rng,torch.get_rng_state());assert torch.equal(grad,a.block.grad)
    assert a.diagnostic_identity()==initial

def test_forward_inplace_effective_weight_rejected():
    class Bad(Toy):
        def forward(self,x):
            self.lift.weight.add_(1)
            return super().forward(x)
    a=p.FrozenForceBlock(Bad(),lift_name='lift.weight')
    with pytest.raises(RuntimeError):a(torch.ones(1,12,1,1,dtype=torch.float64))
    a.verify_frozen()  # temporary composed weight, not original base, was modified
