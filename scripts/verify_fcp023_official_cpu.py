"""Prepared official tiny CPU verification; execute only after Root review."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import torch
from physicsnemo.models.fno import FNO
import p023_force_block as block

P021_SHA='bcefcad2fa622e5b69133725aa8d39db5a1b0a41a1b5e6c7dcb1f6b7d99a4962'
path=Path('/work/p021_causal_force.py')
if block.sha(path)!=P021_SHA:raise ValueError('P021 helper differs')
spec=importlib.util.spec_from_file_location('p023_official_p021',path);causal=importlib.util.module_from_spec(spec);spec.loader.exec_module(causal)
torch.set_num_threads(2);torch.manual_seed(41)
# Production lift24 requires latent48; modes/grid/decoder are tiny CPU fixtures.
kwargs=dict(out_channels=7,dimension=2,latent_channels=48,num_fno_layers=5,num_fno_modes=2,padding=0,decoder_layers=2,decoder_layer_size=8,coord_features=True)
old=FNO(in_channels=6,**kwargs);base=FNO(in_channels=10,**kwargs)
causal.warmstart(old,base,inspect.getfile(FNO));source=block.verify_official_sources(base)
base2=copy.deepcopy(base);material=copy.deepcopy(base)
adapter=block.FrozenForceBlock(base);other=block.FrozenForceBlock(base2)
initial=adapter.diagnostic_identity()
x=torch.randn(2,10,4,4,requires_grad=True)
with torch.no_grad():
    zero_output=adapter(x);parent_output=old(x[:,:6])
    torch.testing.assert_close(zero_output,parent_output,rtol=1e-5,atol=1e-6)
    zero_difference=float((zero_output-parent_output).abs().max())
with torch.no_grad():adapter.block.fill_(.01);other.block.copy_(adapter.block);dict(material.named_parameters())[block.LIFT].copy_(adapter.effective_weight())
for p in material.parameters():p.requires_grad_(False)
dict(material.named_parameters())[block.LIFT].requires_grad_(True)
y=x.detach().clone().requires_grad_(True)
a=adapter(x);b=material(y);a.square().mean().backward();b.square().mean().backward()
torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
torch.testing.assert_close(adapter.block.grad,dict(material.named_parameters())[block.LIFT].grad[:,6:10],rtol=1e-4,atol=1e-6)
torch.testing.assert_close(x.grad,y.grad,rtol=1e-4,atol=1e-6)
materialized=dict(output_max_abs=float((a-b).detach().abs().max()),block_gradient_max_abs=float((adapter.block.grad-dict(material.named_parameters())[block.LIFT].grad[:,6:10]).abs().max()),input_gradient_max_abs=float((x.grad-y.grad).abs().max()))
adapter.block.grad=None;del a,b,x,y,material
h1=torch.randn(100,6,4,4)*.1;ar=torch.randn_like(h1)*.1;mask=torch.ones(1,1,4,4);current=torch.randn(101,4)*.1;target=torch.randn(100,4)*.1
def predict(net,x,mask):
    raw=net(x);return raw[:,:3],(raw[:,3:]*mask).mean((-2,-1))
def balanced(p,y):
    mse=(p-y).square().mean((0,1));return dict(balanced=.5*mse.mean()+.5*mse[3])
cp=causal.recurrent_objective(adapter,h1,ar,mask,current,target,predict,balanced,checkpointed=True)
fg=causal.recurrent_objective(other,h1,ar,mask,current,target,predict,balanced,checkpointed=False)
cp['total'].backward();fg['total'].backward()
torch.testing.assert_close(cp['ar'],fg['ar'],rtol=1e-5,atol=1e-6)
torch.testing.assert_close(adapter.block.grad,other.block.grad,rtol=1e-4,atol=1e-6)
h100=dict(output_max_abs=float((cp['ar']-fg['ar']).detach().abs().max()),block_gradient_max_abs=float((adapter.block.grad-other.block.grad).abs().max()),block_gradient_norm=float(adapter.block.grad.norm()))
assert torch.isfinite(adapter.block.grad).all() and h100['block_gradient_norm']>0
with torch.no_grad():
    perturbed=current.clone();perturbed[0]+=1
    changed=causal.recurrent_objective(adapter,h1,ar,mask,perturbed,target,predict,balanced,checkpointed=False)
    h100['initial_force_perturbation_first_ar_max_abs']=float((changed['ar'][0]-cp['ar'][0]).abs().max())
    assert h100['initial_force_perturbation_first_ar_max_abs']>0
before=adapter.block.detach().clone();opt=torch.optim.AdamW([adapter.block],lr=1e-5,weight_decay=1e-4)
block.assert_optimizer_scope(adapter,opt);opt.step();block.assert_optimizer_scope(adapter,opt,step=1)
update=block.block_update_metrics(before,adapter.block)
with torch.no_grad():causal.recurrent_objective(adapter,h1,ar,mask,current,target,predict,balanced,zero_conditioning=True,checkpointed=False)
adapter.verify_frozen();adapter.restore_zero();assert adapter.diagnostic_identity()==initial
print(json.dumps(dict(status='FC_P023_TINY_CPU_ENGINEERING_NOT_ADMISSION',sources=source,module_sha256=block.sha(block.__file__),script_sha256=block.sha(__file__),zero_parent_output_max_abs=zero_difference,materialized=materialized,h100=h100,
    optimizer_parameter_scalars=96,optimizer_states=1,update=update,all_old_tensors_unchanged=True,final_zero_restore=True,cuda_available=torch.cuda.is_available(),output_rtol=1e-5,output_atol=1e-6,gradient_rtol=1e-4,gradient_atol=1e-6)))
