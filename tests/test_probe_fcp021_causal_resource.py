from types import SimpleNamespace
import pytest
import torch
import probe_fcp021_causal_resource as p

@pytest.mark.parametrize('free,available,startup,okay',[(30,50,True,True),(29.9,60,True,False),(31,49,True,False),(20,20,False,True),(19.99,60,False,False),(30,19.9,False,False)])
def test_resource_floors(free,available,startup,okay):
    if okay:p.check_limits({'MemFree':free,'MemAvailable':available},1,startup=startup)
    else:
        with pytest.raises(RuntimeError):p.check_limits({'MemFree':free,'MemAvailable':available},1,startup=startup)

def test_deadline():
    with pytest.raises(RuntimeError):p.check_limits({'MemFree':40,'MemAvailable':80},900.01)

def model():
    x=torch.nn.Module();x.register_parameter('lift',torch.nn.Parameter(torch.ones(2,12,1,1)))
    for i in range(27):x.register_parameter('extra'+str(i),torch.nn.Parameter(torch.ones(1)))
    x.register_parameter('frozen',torch.nn.Parameter(torch.ones(1),requires_grad=False))
    for v in x.parameters():
        if v.requires_grad:v.grad=torch.zeros_like(v)
    return x

def test_storage_projection_and_added_columns():
    x=model();r=p.gradient_report(x,'lift')
    assert r['new_force_column_gradient_norm']==0
    assert r['adam_two_moment_bytes_projection']==2*(24+27)*4
    x.lift.grad[:,6:10]=1
    assert p.gradient_report(x,'lift')['new_force_column_gradient_norm']==pytest.approx(8**.5)

@pytest.mark.parametrize('bad',['missing','nan','frozen','count'])
def test_fail_closed_gradients(bad):
    x=model()
    if bad=='missing':x.lift.grad=None
    if bad=='nan':x.lift.grad.fill_(float('nan'))
    if bad=='frozen':x.frozen.grad=torch.ones_like(x.frozen)
    if bad=='count':x.register_parameter('extra28',torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(RuntimeError):p.gradient_report(x,'lift')

def test_expanded_identity_distinct_from_old_six():
    a,b,c=[model() for _ in range(3)]
    for x in [a,b,c]:x.zero_grad(set_to_none=True)
    a.sig='flow';b.sig='old';c.sig='expanded'
    objective=SimpleNamespace(tensor_state_sha256=lambda x:x.sig)
    helper=SimpleNamespace(TENSORS={'flow':'flow','aerodynamic':'old'})
    p.verify_identities(a,b,c,objective,helper,'expanded')
    c.sig='old'
    with pytest.raises(RuntimeError):p.verify_identities(a,b,c,objective,helper,'expanded')

def test_module_pin_rejects_before_import(tmp_path):
    path=tmp_path/'bad.py';path.write_text('raise AssertionError("must not import")')
    with pytest.raises(ValueError):p.pinned_module(path,'0'*64,'bad')

def test_harness_has_no_optimizer_or_model_save():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(p.__file__).read_text())
    attrs=[x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)]
    assert 'AdamW' not in attrs and 'step' not in attrs and 'save_checkpoint' not in attrs
