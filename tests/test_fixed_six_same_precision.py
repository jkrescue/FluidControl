"""Synthetic CPU plumbing tests; not scientific model/CFD validation."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch

spec=importlib.util.spec_from_file_location('six',Path(__file__).parents[1]/'scripts'/'p064_fixed_six_same_precision.py')
six=importlib.util.module_from_spec(spec);spec.loader.exec_module(six)

def fixtures():
    calls=[]
    items=[]
    for index,family,case,start in six.WINDOWS:
        sample=dict(state=torch.zeros(3,2,2),target_state=torch.zeros(100,3,2,2),
                    mask=torch.ones(1,2,2),omega=torch.zeros(101,1),target_force=torch.zeros(100,4))
        items.append(dict(global_index=index,family=family,
                          identity=dict(case=case,start=start,split='train'),sample=sample))
    def tsha(t):return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()
    def msha(m):return ''.join(tsha(x) for x in m.state_dict().values())
    def preceding(ds,ident,sample,k,history):
        assert k==1
        return torch.empty(1,0,3,2,2),torch.empty(1,0,1),{'frame_indices':[ident['start']]}
    def predict(*args):
        assert torch.get_float32_matmul_precision()=='highest'
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert not torch.is_grad_enabled()
        calls.append('predict')
    def frozen(flow,state,mask,omega,fn):
        fn(flow,state,mask)
        return state[:,None].expand(-1,100,-1,-1,-1)
    def chunk(aero,states,h1,mask,omega,target,pre,actions,fn,obj,*,backward):
        assert backward is False and pre.shape==(1,0,3,2,2) and actions.shape==(1,0,1)
        assert target.shape==(1,100,4)
        fn(aero,states,mask)
        return dict(total=0.,h1_balanced=0.,ar_balanced=0.,normalized_predictions={'h1':target,'ar':target})
    def metrics(helper,preds,target,std,result):
        assert set(preds)=={'h1','ar'} and preds['h1'].shape==(100,4)
        return {'objective':{k:result[k] for k in ('total','h1_balanced','ar_balanced')}}
    kw=dict(flow=torch.nn.Linear(2,2),aero=torch.nn.Linear(2,2),dataset=object(),base=NS(force_std=torch.ones(4)),device='cpu',
            trainer=NS(diagnostic_windows=lambda _:items),
            original_runner=NS(preceding=preceding,grouped_panel=lambda rows:{'count':len(rows)}),
            objective=NS(tensor_state_sha256=msha,tensor_sha256=tsha,frozen_flow_states=frozen,
                         true_state_inputs=lambda state,target:torch.cat((state[:,None],target[:,:-1]),dim=1)),
            chunk=NS(chunk_force_objective=chunk),history=object(),
            p020=NS(metrics=metrics,aggregate=lambda rows:{'count':len(rows)}),helper=object(),predict=predict,guard=lambda:None)
    return kw,items,calls

def test_original_six_calls_readonly_and_precision_restored():
    kw,items,calls=fixtures()
    before=torch.get_rng_state().clone()
    precision=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    out=six.evaluate_fixed_six(**kw)
    assert len(out['rows'])==6 and len(calls)==12 and out['aggregate']=={'count':6}
    assert torch.equal(before,torch.get_rng_state())
    assert precision==(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    assert out['optimizer_steps']==0 and not out['candidate_saved']
    arrays=out['rows'][0]['arrays']
    assert set(arrays)=={'h1','ar','target_force','force_std'}
    assert torch.tensor(arrays['h1']).shape==(100,4)
    assert arrays['target_force']==arrays['h1'] and arrays['force_std']==[1.]*4

def test_wrong_window_rejected_before_forward():
    kw,items,calls=fixtures();items[0]['identity']['start']+=1
    with pytest.raises(ValueError,match='identity/order'):six.evaluate_fixed_six(**kw)
    assert not calls

def test_prediction_exception_restores_precision():
    kw,_,_=fixtures()
    precision=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    def failure(*args):raise RuntimeError('synthetic forward exception')
    kw['predict']=failure
    with pytest.raises(RuntimeError,match='synthetic'):six.evaluate_fixed_six(**kw)
    assert precision==(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
