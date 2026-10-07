import ast
import importlib.util
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'scripts/run_p064_fixed_small_fit_r2.py'

def load():
    spec=importlib.util.spec_from_file_location('fit_worker_test',PATH)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_protocol_and_resources():
    m=load()
    assert m.PROTOCOL['starts']==list(range(0,800,100))
    assert m.PROTOCOL['points']==40 and m.PROTOCOL['microbatch']==10
    assert m.RESOURCES['memory_bytes']==24*2**30
    assert m.RESOURCES['allocator_bytes']==16*2**30
    assert m.PROTOCOL['no_candidate_save'] and m.PROTOCOL['no_dev']

def test_chunked_absolute_objective_matches_full_panel():
    import torch
    x=torch.arange(160,dtype=torch.float64).reshape(40,4)/100
    w=torch.tensor([.125,.125,.125,.625],dtype=torch.float64)
    a=torch.tensor([.5,.8,1.2,1.7],dtype=torch.float64,requires_grad=True)
    b=a.detach().clone().requires_grad_(True)
    direct=((x*a-x.square()).square()*w).sum()/40;direct.backward()
    accumulated=0.
    for i in range(0,40,10):
        loss=((x[i:i+10]*b-x[i:i+10].square()).square()*w).sum()/40
        loss.backward();accumulated+=float(loss.detach())
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)
    assert abs(accumulated-float(direct.detach()))<1e-12

def test_force_pooling_seven_channels_preserves_mask():
    import torch
    raw=torch.arange(2*7*3*4,dtype=torch.float64).reshape(2,7,3,4)
    mask=torch.ones(2,1,3,4,dtype=torch.float64);mask[:,:,:,0]=0
    got=(raw[:,3:7]*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1)
    torch.testing.assert_close(got,raw[:,3:7,: ,1:].mean((-2,-1)))

def test_no_model_checkpoint_save_api():
    tree=ast.parse(PATH.read_text())
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'torch.save' not in calls and 'save_checkpoint' not in calls
    assert 'pair.flow_model' not in calls

def test_actual_worker_precision_order_cpu():
    import torch
    worker=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    picked=[];seen=[]
    for node in worker.body:
        code=ast.unparse(node)
        if 'pair, identity = load_dual_fno' in code:
            picked.append(ast.parse('check_loader_precision()').body[0])
        elif code.startswith('torch.set_float32_matmul_precision(') or code.startswith('torch.backends.cuda.matmul.allow_tf32 =') or code.startswith('torch.backends.cudnn.allow_tf32 ='):
            picked.append(node)
    def check_loader_precision():
        assert torch.get_float32_matmul_precision()=='high'
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        seen.append('load')
    previous=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=picked,type_ignores=[])),'actual_worker_precision','exec'),dict(torch=torch,check_loader_precision=check_loader_precision))
        assert seen==['load']
        assert torch.get_float32_matmul_precision()=='highest'
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(previous[0]);torch.backends.cuda.matmul.allow_tf32=previous[1];torch.backends.cudnn.allow_tf32=previous[2]
