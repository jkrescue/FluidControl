"""Project-only96-coefficient adapter around unchanged official FNO forward."""
import hashlib
import inspect
from pathlib import Path
import torch
from torch.func import functional_call

LIFT='spec_encoder.lift_network.0.conv.weight'
OFFICIAL_SHA='e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9'
FUNCTIONAL_SHA='dcee4f31fd66bb7641fc1a99dd6eb6ccf1fa1e7e7f3df9a343f24a36d9db2407'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_official_sources(base):
    paths=dict(official=inspect.getfile(type(base)),functional=inspect.getfile(functional_call))
    if sha(paths['official'])!=OFFICIAL_SHA or sha(paths['functional'])!=FUNCTIONAL_SHA:raise ValueError('pinned official/PyTorch implementation differs')
    return {k:dict(path=v,sha256=sha(v)) for k,v in paths.items()}

def tensor_digest(named_tensors):
    digest=hashlib.sha256()
    for name,value in sorted(named_tensors):
        x=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(x.dtype).encode());digest.update(str(tuple(x.shape)).encode())
        digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

class FrozenForceBlock(torch.nn.Module):
    """Base stays frozen; only block receives optimizer gradients/moments.

    Caller must pin official sources for real use. Custom toy bases are CPU tests.
    No cache of assembled weights: checkpoint replay rebuilds differentiable cat.
    Construct after moving base to its final device/dtype. Reconstruct a new
    adapter for independent models; deep-copying cached version counters is not
    a supported state-transfer operation. Restore only the independent block.
    """
    def __init__(self,base,*,lift_name=LIFT):
        super().__init__();self.base=base;self.lift_name=lift_name
        parameters=dict(base.named_parameters())
        if lift_name not in parameters or parameters[lift_name].shape!=(24,12,1,1):raise ValueError('expanded24x12 lifting weight required')
        weight=parameters[lift_name]
        if weight.is_complex() or not weight.is_floating_point() or not torch.isfinite(weight).all() or torch.count_nonzero(weight[:,6:10]).item()!=0:raise ValueError('finite real zero-force-column initial weight required')
        for p in base.parameters():p.requires_grad_(False);p.grad=None
        self.block=torch.nn.Parameter(torch.zeros_like(weight[:,6:10]))
        self._base_signature=tensor_digest(self._base_tensors())
        self._schema={n:(tuple(v.shape),v.dtype) for n,v in self._base_tensors()}
        self._versions={n:v._version for n,v in self._base_tensors()}

    def _base_tensors(self):return list(self.base.named_parameters())+list(self.base.named_buffers())

    def verify_frozen(self,*,hash_values=True):
        tensors=self._base_tensors()
        if {n:(tuple(v.shape),v.dtype) for n,v in tensors}!=self._schema:raise RuntimeError('base schema changed')
        if any(p.requires_grad or p.grad is not None for p in self.base.parameters()):raise RuntimeError('base gradient scope changed')
        if {n:v._version for n,v in tensors}!=self._versions:raise RuntimeError('in-place base mutation forbidden')
        if hash_values and tensor_digest(tensors)!=self._base_signature:raise RuntimeError('base tensor bytes changed')

    def effective_weight(self):
        weight=dict(self.base.named_parameters())[self.lift_name]
        return torch.cat((weight[:,:6],self.block,weight[:,10:12]),dim=1)

    def forward(self,inputs):
        self.verify_frozen(hash_values=False)
        if not torch.isfinite(self.block).all():raise FloatingPointError('nonfinite96-block')
        mapping=dict(self._base_tensors());effective=self.effective_weight();version=effective._version;mapping[self.lift_name]=effective
        result=functional_call(self.base,mapping,(inputs,),strict=True)
        if effective._version!=version:raise RuntimeError('in-place effective weight mutation forbidden')
        self.verify_frozen(hash_values=False)
        return result

    def restore_zero(self):
        self.verify_frozen()
        with torch.no_grad():self.block.zero_()
        self.block.grad=None

    def diagnostic_identity(self):
        self.verify_frozen()
        return dict(base_tensor_sha256=self._base_signature,block_sha256=tensor_digest([('block',self.block)]),effective_lifting_sha256=tensor_digest([(self.lift_name,self.effective_weight())]))

def assert_optimizer_scope(adapter,optimizer,*,step=None):
    groups=optimizer.param_groups
    if len(groups)!=1 or len(groups[0]['params'])!=1 or groups[0]['params'][0] is not adapter.block:raise ValueError('optimizer must own only96-block leaf')
    if adapter.block.shape!=(24,4,1,1) or not adapter.block.requires_grad:raise ValueError('96-block scope differs')
    adapter.verify_frozen()
    if step is None:
        if optimizer.state:raise ValueError('fresh optimizer state required')
    else:
        if set(optimizer.state)!={adapter.block}:raise ValueError('only one optimizer state permitted')
        state=optimizer.state[adapter.block]
        if float(state['step'])!=step:raise ValueError('optimizer step differs')
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].shape!=adapter.block.shape or not torch.isfinite(state[key]).all():raise FloatingPointError('finite96 moment required')
    if not torch.isfinite(adapter.block).all() or (adapter.block.grad is not None and not torch.isfinite(adapter.block.grad).all()):raise FloatingPointError('nonfinite block/gradient')

def block_update_metrics(before,after):
    if before.shape!=(24,4,1,1) or after.shape!=before.shape:raise ValueError('96-block shape differs')
    a=after.detach().cpu().double();delta=a-before.detach().cpu().double()
    if not torch.isfinite(a).all() or not torch.isfinite(delta).all():raise FloatingPointError('nonfinite update')
    return dict(values=a.reshape(-1).tolist(),update_l2=float(delta.norm()),update_max_abs=float(delta.abs().max()),cumulative_l2=float(a.norm()),cumulative_max_abs=float(a.abs().max()))
