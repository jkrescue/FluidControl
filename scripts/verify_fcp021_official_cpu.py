"""Tiny official-FNO CPU fixture, no CFD/model checkpoint or optimizer."""
import inspect,json,copy
import torch
from physicsnemo.models.fno import FNO
import p021_causal_force as m
torch.set_num_threads(2);torch.manual_seed(41)
kwargs=dict(out_channels=7,dimension=2,latent_channels=4,num_fno_layers=5,num_fno_modes=2,padding=0,decoder_layers=2,decoder_layer_size=8,coord_features=True)
old=FNO(in_channels=6,**kwargs);new=FNO(in_channels=10,**kwargs)
source=inspect.getfile(FNO)
m.warmstart(old,new,source)
print(json.dumps({'official_buffer_inventory':[(n,list(b.shape),str(b.dtype)) for n,b in new.named_buffers()]}),flush=True)
m.validate_paired_model(new)
x=torch.randn(2,6,4,4);forces=torch.randn(2,4,4,4)
with torch.no_grad():a=old(x);b=new(torch.cat((x,forces),1))
frozen={k:v.detach().clone() for k,v in new.named_parameters() if not v.requires_grad}
new(torch.cat((x,forces),1)).square().mean().backward()
assert set(frozen)==set(m.FROZEN)
assert all(p.grad is None and torch.equal(p,frozen[k]) for k,p in new.named_parameters() if k in frozen)
assert sum(p.requires_grad for p in new.parameters())==28
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters() if p.requires_grad)
warmstart_difference=float((a-b).abs().max())
# Deliberately nonzero added columns only in this synthetic gradient fixture.
with torch.no_grad():dict(new.named_parameters())[m.LIFT][:,6:10].fill_(.01)
new.zero_grad(set_to_none=True);full=copy.deepcopy(new)
h1=torch.randn(100,6,4,4)*.1;ar=torch.randn_like(h1)*.1
mask=torch.ones(1,1,4,4);current=torch.randn(101,4)*.1;target=torch.randn(100,4)*.1
def predict(net,x,mask):
    raw=net(x);return raw[:,:3],(raw[:,3:]*mask).mean((-2,-1))
def balanced(p,y):
    channel=(p-y).square().mean((0,1));return dict(balanced=.5*channel.mean()+.5*channel[3])
data=(h1,ar,mask,current,target,predict,balanced)
cp=m.recurrent_objective(new,*data,checkpointed=True)
fg=m.recurrent_objective(full,*data,checkpointed=False)
cp['total'].backward();fg['total'].backward()
torch.testing.assert_close(cp['ar'],fg['ar'],rtol=1e-5,atol=1e-6)
torch.testing.assert_close(cp['h1'],fg['h1'],rtol=1e-5,atol=1e-6)
differences={};imag_nonzero=0;real_pair_imag_nonzero=0;parameter_schema={}
for (name,p),(other,q) in zip(new.named_parameters(),full.named_parameters()):
    assert name==other
    if not p.requires_grad:
        assert p.grad is None and q.grad is None and torch.equal(p,frozen[name]);continue
    assert p.grad is not None and q.grad is not None
    torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)
    differences[name]=float((p.grad-q.grad).abs().max())
    parameter_schema[name]=dict(shape=list(p.shape),dtype=str(p.dtype))
    if p.grad.is_complex() and p.grad.imag.abs().max()>0:imag_nonzero+=1
    if '.spconv_layers.' in name and name.endswith(('weights1','weights2')) and not p.is_complex() and p.shape[-1]==2 and p.grad[...,1].abs().max()>0:real_pair_imag_nonzero+=1
assert imag_nonzero+real_pair_imag_nonzero>0
changed=current.clone();changed[0]+=1
with torch.no_grad():perturbed=m.recurrent_objective(new,h1,ar,mask,changed,target,predict,balanced,checkpointed=False)
initial_effect=float((cp['ar'][0].detach()-perturbed['ar'][0]).abs().max())
assert initial_effect>0
print(json.dumps(dict(status='TINY_OFFICIAL_CPU_ENGINEERING_ONLY_NOT_ADMISSION',official_source_sha256=m.sha(source),module_sha256=m.sha(m.__file__),maximum_output_absolute_difference=warmstart_difference,outputs_bitwise_equal=torch.equal(a,b),trainable_tensors=28,frozen_biases_unchanged=True,cuda_available=torch.cuda.is_available(),no_optimizer=True,no_model_save=True,
    fullgraph_checkpoint_h100=dict(output_max_abs_difference=float((cp['ar']-fg['ar']).abs().max()),gradient_max_abs_difference=max(differences.values()),per_parameter_gradient_max_abs_difference=differences,parameter_schema=parameter_schema,nonzero_native_complex_imaginary_gradient_tensors=imag_nonzero,nonzero_spectral_real_pair_imaginary_slot_gradient_tensors=real_pair_imag_nonzero,initial_force_effect=initial_effect,output_rtol=1e-5,output_atol=1e-6,gradient_rtol=1e-4,gradient_atol=1e-6,added_force_columns_for_test_only=.01))))
