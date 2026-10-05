import pytest
import torch
import p021_causal_force as m

def balanced(p,y):
    mse=(p-y).square().mean((0,1));return {'balanced':.5*mse.mean()+.5*mse[3],'channel_mse':mse}

class Toy(torch.nn.Module):
    def __init__(self,complex=False):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.45+.08j if complex else .45,dtype=torch.complex128 if complex else torch.float64))
    def forward(self,x):
        current=x[:,6:10,0,0]
        z=self.weight.real*current+.03*x[:,:4,0,0]
        if self.weight.is_complex():z=z+self.weight.imag*current.flip(-1)
        return z.real

def predict(model,x,mask):return None,model(x)

def fixtures():
    g=torch.Generator().manual_seed(9)
    return (torch.randn(100,6,1,1,generator=g,dtype=torch.float64),torch.randn(100,6,1,1,generator=g,dtype=torch.float64),torch.ones(1,1,1,1,dtype=torch.float64),torch.randn(101,4,generator=g,dtype=torch.float64),torch.randn(100,4,generator=g,dtype=torch.float64))

@pytest.mark.parametrize('complex',[False,True])
def test_checkpoint_full_chain(complex):
    data=fixtures();a=Toy(complex);b=Toy(complex)
    x=m.recurrent_objective(a,*data,predict,balanced,checkpointed=False)
    y=m.recurrent_objective(b,*data,predict,balanced,checkpointed=True)
    x['total'].backward();y['total'].backward()
    torch.testing.assert_close(x['ar'],y['ar'],rtol=0,atol=0)
    torch.testing.assert_close(a.weight.grad,b.weight.grad,rtol=1e-13,atol=1e-13)
    if complex:assert a.weight.grad.imag.abs()>0

def test_future_truth_not_ar_input():
    data=list(fixtures());a=m.recurrent_objective(Toy(),*data,predict,balanced)
    data[3]=data[3].clone();data[3][1:]+=100;data[4]=data[4]+22
    b=m.recurrent_objective(Toy(),*data,predict,balanced)
    torch.testing.assert_close(a['ar'],b['ar'],rtol=0,atol=0)
    assert not torch.equal(a['h1'],b['h1'])
    data[3][0]+=1;c=m.recurrent_objective(Toy(),*data,predict,balanced)
    assert not torch.equal(b['ar'][:12],c['ar'][:12])

def test_cross_boundary_derivative_not_detached():
    data=list(fixtures());data[0].zero_();data[1].zero_();data[3].fill_(1)
    model=Toy();out=m.recurrent_objective(model,*data,predict,balanced)
    out['ar'][10].sum().backward()
    torch.testing.assert_close(model.weight.grad,torch.tensor(4*11*.45**10,dtype=torch.float64),rtol=1e-12,atol=1e-14)

def test_zero_control_no_current_information():
    data=list(fixtures());a=m.recurrent_objective(Toy(),*data,predict,balanced,zero_conditioning=True)
    data[3]*=20;b=m.recurrent_objective(Toy(),*data,predict,balanced,zero_conditioning=True)
    torch.testing.assert_close(a['h1'],b['h1']);torch.testing.assert_close(a['ar'],b['ar'])

def test_j0_weighting():
    out=m.recurrent_objective(Toy(),*fixtures(),predict,balanced)
    data=fixtures();expected=0
    for d in ['h1','ar']:
        for k in range(100):expected+=balanced(out[d][None,k:k+1],data[-1][None,k:k+1])['balanced']/200
    torch.testing.assert_close(out['total'],expected)

def test_expansion_coordinates_bias_preserved():
    old={m.LIFT:torch.arange(24*8.).reshape(24,8,1,1),m.FROZEN[0]:torch.randn(24),m.FROZEN[1]:torch.randn(48)}
    new={**old,m.LIFT:torch.empty(24,12,1,1)};out=m.expanded_state(old,new)
    assert not out[m.LIFT][:,6:10].count_nonzero()
    torch.testing.assert_close(out[m.LIFT][:,10:],old[m.LIFT][:,6:])
    for key in m.FROZEN:torch.testing.assert_close(out[key],old[key],rtol=0,atol=0)
    x=torch.randn(2,8,3,3);expanded=torch.cat((x[:,:6],torch.randn(2,4,3,3),x[:,6:]),1)
    torch.testing.assert_close(torch.nn.functional.conv2d(x,old[m.LIFT]),torch.nn.functional.conv2d(expanded,out[m.LIFT]),rtol=1e-5,atol=1e-4)

def test_normalization_channels_and_bad_std():
    x=torch.arange(8.).reshape(2,4);mean=torch.arange(4.);std=torch.arange(1.,5.)
    torch.testing.assert_close(m.normalize(x,mean,std)[0],torch.zeros(4))
    with pytest.raises(ValueError):m.normalize(x,mean,std*0)

def test_persistence_distinct_rms_and_waveform():
    x=torch.arange(404.).reshape(101,4);y=x[1:]+3
    r=m.persistence(x,y)
    assert r['h1']['absolute_rms_error']==[0]*4
    assert r['h1']['mae']==[7]*4
    assert r['ar']['predicted_rms']==[0]*4

def test_provenance_rejects_missing_pin(tmp_path):
    p=tmp_path/'fake';p.write_text('{}')
    with pytest.raises(ValueError):m.load_causal_inputs(p,p,tmp_path)

def test_frozen_data_rejected():
    data=list(fixtures());data[1].requires_grad_(True)
    with pytest.raises(ValueError):m.recurrent_objective(Toy(),*data,predict,balanced)

def test_pinned_actual_objective():
    from pathlib import Path
    path=Path('/workspace/fluid_control/artifacts/fcp013_training_source_1634c05_immutable/scripts/train_fcp013_independent_force_fno.py')
    if not path.exists():pytest.skip('immutable Main fixture absent')
    objective=m.load_objective(path)
    a=Toy(True);b=Toy(True);data=fixtures()
    x=m.recurrent_objective(a,*data,predict,objective,checkpointed=True)
    y=m.recurrent_objective(b,*data,predict,balanced,checkpointed=False)
    x['total'].backward();y['total'].backward()
    torch.testing.assert_close(x['total'],y['total'],rtol=0,atol=0)
    torch.testing.assert_close(a.weight.grad,b.weight.grad,rtol=1e-13,atol=1e-13)

@pytest.mark.parametrize('layer',[torch.nn.BatchNorm1d(4),torch.nn.Dropout(.2),torch.nn.Dropout2d(.2),torch.nn.InstanceNorm1d(4,track_running_stats=True),torch.nn.RNN(4,4)])
@pytest.mark.parametrize('training',[False,True])
def test_reject_batch_leak_and_stochastic_layers(layer,training):
    layer.train(training)
    with pytest.raises(ValueError):m.validate_paired_model(layer)

def test_reject_unreviewed_buffers():
    toy=Toy();toy.register_buffer('running_counter',torch.zeros(()))
    with pytest.raises(ValueError):m.validate_paired_model(toy)

def test_only_empty_official_device_marker_allowed():
    toy=Toy();toy.register_buffer('device_buffer',torch.empty(0));m.validate_paired_model(toy)
    toy.device_buffer=torch.zeros(1)
    with pytest.raises(ValueError):m.validate_paired_model(toy)
