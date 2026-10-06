import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

STAGE = Path(__file__).resolve().parents[1]
REPO = STAGE
sys.path.insert(0, str(STAGE / 'scripts'))
import p064_response_aux as aux


def function(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def test_delta_shared_zero_gradient():
    p = torch.zeros(12, 4, dtype=torch.float64, requires_grad=True)
    y = torch.arange(48, dtype=torch.float64).reshape(12, 4) / 10
    loss = aux.response_loss(p, y)
    loss.backward()
    residual = (p.detach().reshape(4, 3, 4)[:, (0, 2)]-p.detach().reshape(4, 3, 4)[:, 1:2])-(y.reshape(4, 3, 4)[:, (0, 2)]-y.reshape(4, 3, 4)[:, 1:2])
    expected = torch.zeros(4, 3, 4, dtype=torch.float64)
    expected[:, 0] = 2*residual[:, 0]/32
    expected[:, 2] = 2*residual[:, 1]/32
    expected[:, 1] = -2*residual.sum(1)/32
    torch.testing.assert_close(p.grad.reshape(4, 3, 4), expected)


def test_normalized_delta_equals_physical_std_scaled():
    p = torch.arange(48.).reshape(12, 4)
    y = p.flip(0)
    mean, std = torch.tensor([1., 2., 3., 4.]), torch.tensor([2., 3., 4., 5.])
    physical = ((p-y).reshape(4, 3, 4)[:, (0, 2)]-(p-y).reshape(4, 3, 4)[:, 1:2])/std
    torch.testing.assert_close(aux.response_loss((p-mean)/std, (y-mean)/std), physical.square().mean())


def test_real_accumulator_eight_compensation_and_single_adam():
    step = function(REPO/'scripts/train_fcp015_window_accumulation.py', 'accumulation_step', {'GROUP': 8})
    model = torch.nn.Linear(6, 4, bias=False).double()
    reference = torch.nn.Linear(6, 4, bias=False).double()
    reference.load_state_dict(model.state_dict())
    x = torch.arange(72, dtype=torch.float64).reshape(12, 6)/100
    target = torch.zeros(12, 4, dtype=torch.float64)
    masks = torch.ones(12, 1, 1, 1, dtype=torch.float64)
    packed = (x, masks, target, [])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5625e-7, weight_decay=1e-4)
    expected = sum((reference(x[:1])-(i/10)).square().mean() for i in range(8))/8 + aux.response_loss(reference(x), target)
    expected.backward()
    captured = {}
    def run(i):
        loss = (model(x[:1])-(i/10)).square().mean()
        loss.backward()
        if i == 7:
            aux.backward_once(model, packed, lambda m, inp, mask:(None, m(inp)))
        return dict(total=float(loss.detach()), h1_balanced=0., ar_balanced=0.)
    def audit(m):
        captured['grad'] = m.weight.grad.clone()
        return {}
    result = step(model, optimizer, range(8), run, audit)
    torch.testing.assert_close(captured['grad'], reference.weight.grad, rtol=1e-12, atol=1e-12)
    assert result['optimizer_steps'] == 1
    assert all(int(s['step']) == 1 for s in optimizer.state.values())


def test_actual_scope_keeps_two_biases_and_flow_frozen():
    path = REPO/'scripts/probe_fcp026_history_resource.py'
    tree = ast.parse(path.read_text())
    frozen = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='FROZEN' for t in n.targets))
    scope = function(path, 'trainable_scope', {'FROZEN': frozen})
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(())) for _ in range(30)])
        def named_parameters(self, *args, **kwargs):
            return iter(zip([f'p{i}' for i in range(28)] + list(frozen), self.values))
    model = Fixture()
    scope(model)
    assert sum(p.requires_grad for p in model.parameters()) == 28
    assert all(not p.requires_grad for name,p in model.named_parameters() if name in frozen)
    flow = torch.nn.Linear(2, 2).requires_grad_(False)
    before = {k:v.clone() for k,v in flow.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    sum(p.square() for p in model.parameters() if p.requires_grad).backward()
    optimizer.step()
    assert all(p.grad is None for name,p in model.named_parameters() if name in frozen)
    assert all(torch.equal(v,before[k]) for k,v in flow.state_dict().items())


def test_fixed_protocol_no_future_field_input():
    p=aux.protocol()
    assert (p['coefficient'],p['updates'],p['extra_aerodynamic_samples'],p['state_count']) == (1.,32,384,4)
    assert aux.ROLES == ('m075','zero','p075')
    source=(STAGE/'scripts/p064_response_aux.py').read_text()
    assert "q1['state']" not in source and "q1['force']" in source
    with pytest.raises(ValueError):
        aux.response_loss(torch.zeros(16,4),torch.zeros(16,4))


def test_new_consumer_protocol_and_old_profile_isolation(tmp_path):
    import copy
    import json
    import hashlib
    path=STAGE/'src/fluid_control/dual_fno_response_aux.py'
    spec=importlib.util.spec_from_file_location('response_consumer_fixture',path)
    loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)
    old=loader._experiment_contract(loader.P064_SYSTEM_KIND['B'])
    assert old.get('response_aux') is None and old['extra']['training_experiment']=='FC-P064'
    contract=loader._experiment_contract('FC_P064_RESPONSE_AUX_K1_FRESH_FORCE_FNO')
    assert contract['p064_arm']=='B' and contract['response_aux'] and contract['flow_frozen']
    root=REPO/'artifacts/fcp064_controlled_aero_arm_b_20261006'
    payload=json.loads((root/'dual_model_manifest.json').read_text())
    protocol=json.loads((root/'training_protocol.json').read_text())
    protocol.update(training_experiment='FC-P064-RESPONSE-AUX', auxiliary_response=aux.protocol(), candidate_profile='FC_P064_RESPONSE_AUX_K1_FRESH')
    def write():
        file=tmp_path/'training_protocol.json';file.write_text(json.dumps(protocol))
        payload.update(training_semantics=copy.deepcopy(protocol),training_protocol_file=file.name,training_protocol_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    write()
    loader._validate_p026_protocol(tmp_path,payload,contract)
    with pytest.raises(ValueError):loader._validate_p026_protocol(tmp_path,payload,old)
    protocol['auxiliary_response']['coefficient']=.5;write()
    with pytest.raises(ValueError):loader._validate_p026_protocol(tmp_path,payload,contract)
