import ast
import importlib.util
from pathlib import Path

PATH=(Path(__file__).resolve().parents[1]
      / 'artifacts/p064_representative256_source_20261007_immutable/train_representative256.py')

def worker_ast():
    return next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')

def test_actual_objective_all256_gradient():
    import torch
    node=next(n for n in worker_ast().body if isinstance(n,ast.FunctionDef) and n.name=='objective')
    x=torch.arange(1024,dtype=torch.float64).reshape(256,4)/1000
    target=x.square();weights=x.new_tensor([.125,.125,.125,.625])
    a=x.new_tensor([.4,.8,1.2,1.6],requires_grad=True);b=a.detach().clone().requires_grad_(True)
    env=dict(target=target,weights=weights,prediction=lambda i:x[i:i+10]*a)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_objective','exec'),env)
    got=env['objective']();expected=((x*b-target).square()*weights).sum()/256;expected.backward()
    torch.testing.assert_close(got,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)

def test_actual_precision_load_then_override():
    import torch
    nodes=[];seen=[]
    for n in worker_ast().body:
        code=ast.unparse(n)
        if 'pair, identity = load_dual_fno' in code:nodes+=ast.parse('check()').body
        elif code.startswith(('torch.set_float32_matmul_precision(', 'torch.backends.cuda.matmul.allow_tf32 =','torch.backends.cudnn.allow_tf32 =')):nodes.append(n)
    def check():
        assert torch.get_float32_matmul_precision()=='high'
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        seen.append(True)
    old=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_precision','exec'),dict(torch=torch,check=check))
        assert seen==[True] and torch.get_float32_matmul_precision()=='highest'
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(old[0]);torch.backends.cuda.matmul.allow_tf32=old[1];torch.backends.cudnn.allow_tf32=old[2]

def test_coverage_json_keys_canonicalized():
    import json
    actual=dict(family_counts={0:109,1:45,2:38,3:64})
    saved=json.loads(json.dumps(actual))
    assert actual!=saved
    assert json.loads(json.dumps(actual))==saved
    assert "json.loads(json.dumps(coverage))" in PATH.read_text()
