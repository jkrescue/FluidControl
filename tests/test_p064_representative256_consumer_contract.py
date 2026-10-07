"""Independent CPU metadata roundtrip; no scientific FNO/data execution."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from physicsnemo.utils import load_checkpoint, save_checkpoint

STAGE = (Path(__file__).resolve().parents[1]
         / 'artifacts/p064_representative256_source_20261007_immutable')
PRODUCER = STAGE / 'train_representative256.py'
CONSUMER = STAGE / 'dual_fno_representative256.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


consumer = load(CONSUMER, 'rep256_consumer_independent')


def producer_metadata(*, protocol_sha='1' * 64, closures=137):
    """Evaluate the producer's literal `info=dict(...)` expression itself."""
    tree = ast.parse(PRODUCER.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'save_candidate')
    assignment = next(node for node in ast.walk(function)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'info'
                              for target in node.targets))
    expression = ast.Expression(assignment.value)
    identity = type('Identity', (), {'manifest_sha256':consumer.REP256_PARENT})()
    spec = {'panel_receipt':{'sha256':consumer.REP256_PANEL}}
    return eval(compile(ast.fix_missing_locations(expression), str(PRODUCER), 'eval'),
                {'dict':dict, 'protocol_sha':protocol_sha, 'result':{'closures':closures},
                 'identity':identity, 's':spec})


def manifest_payload(root, *, closures=137):
    protocol_path = root / 'training_protocol.json'
    protocol_path.write_text(json.dumps(consumer.REP256_PROTOCOL,
                                        sort_keys=True, separators=(',', ':')))
    payload = dict(
        kind=consumer.REP256_KIND,
        status='P064_REPRESENTATIVE256_DUAL_FNO_MANIFEST_VERIFIED',
        precision_protocol=consumer.REP256_PRECISION,
        training_experiment='P064-REPRESENTATIVE256', training_points=256,
        training_protocol_file='training_protocol.json',
        training_protocol_sha256=consumer.sha256(protocol_path),
        training_semantics=consumer.REP256_PROTOCOL,
        optimizer_closures=closures, parent_manifest_sha256=consumer.REP256_PARENT,
        panel_receipt_sha256=consumer.REP256_PANEL,
        history_input=consumer._p026_history_input(1),
        parent_history_inventory=consumer._p026_inventory(),
        history_state_module_sha256=consumer.P026_HISTORY_STATE_SHA256,
        history_inference_module_sha256=consumer.P026_HISTORY_INFERENCE_SHA256,
        flow_architecture=consumer.ARCHITECTURE,
        aerodynamic_architecture=consumer.ARCHITECTURE)
    return payload


def actual_producer_manifest(tmp_path):
    """Execute the producer's actual manifest-construction AST, then validate it."""
    repo = Path('/workspace/fluid_control')
    parent_path = repo / 'artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json'
    parent = json.loads(parent_path.read_text())
    candidate = tmp_path / 'candidate'
    (candidate / 'flow').mkdir(parents=True)
    (candidate / 'aerodynamic').mkdir()
    protocol_path = candidate / 'training_protocol.json'
    protocol_path.write_text(json.dumps(consumer.REP256_PROTOCOL,
                                        sort_keys=True, separators=(',', ':')))
    for key in ('model_file', 'state_file'):
        source = parent_path.parent / parent['flow']['checkpoint_relative_directory'] / parent['flow'][key]
        os.link(source, candidate / 'flow' / source.name)
    (candidate / 'aerodynamic/FNO.0.1.mdlus').write_bytes(b'tiny-official-name-model')
    (candidate / 'aerodynamic/checkpoint.0.1.pt').write_bytes(b'tiny-official-name-state')
    function = next(node for node in ast.parse(PRODUCER.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'save_candidate')
    start = next(i for i, node in enumerate(function.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'manifest'
                         for target in node.targets))
    end = next(i for i, node in enumerate(function.body[start:], start)
               if isinstance(node, ast.Expr)
               and 'dual_model_manifest.json' in ast.unparse(node))
    identity = type('Identity', (), {'manifest_sha256':consumer.REP256_PARENT})()
    spec = dict(panel_receipt={'sha256':consumer.REP256_PANEL},
                **{key:{'sha256':hashlib.sha256(key.encode()).hexdigest()}
                   for key in ('driver','core','panel_adapter','consumer')})
    info = producer_metadata(protocol_sha=consumer.sha256(protocol_path), closures=137)
    namespace = dict(copy=copy, parent=parent, result={'closures':137},
                     protocol_sha=consumer.sha256(protocol_path), identity=identity,
                     s=spec, candidate=candidate, json=json, sha=consumer.sha256,
                     PROTOCOL=consumer.REP256_PROTOCOL, info=info)
    exec(compile(ast.fix_missing_locations(ast.Module(body=function.body[start:end+1],
                                                       type_ignores=[])),
                 str(PRODUCER), 'exec'), namespace)
    return candidate / 'dual_model_manifest.json'


def test_actual_producer_metadata_survives_official_cpu_roundtrip(tmp_path):
    payload = manifest_payload(tmp_path)
    expected = consumer.rep256_checkpoint_metadata(payload)
    assert producer_metadata(protocol_sha=payload['training_protocol_sha256']) == expected
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = tmp_path / 'aerodynamic'
    save_checkpoint(checkpoint, models=model, optimizer=optimizer, epoch=1,
                    metadata=expected)
    fresh = torch.nn.Linear(2, 3)
    observed = {}
    assert load_checkpoint(checkpoint, models=fresh, metadata_dict=observed,
                           device='cpu') == 1
    assert observed == expected
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), fresh.parameters()))


def test_new_profile_is_exact_and_does_not_relax_parent(tmp_path):
    payload = manifest_payload(tmp_path)
    consumer._validate_rep256_protocol(tmp_path, payload)
    new_contract = consumer._experiment_contract(consumer.REP256_KIND)
    parent_contract = consumer._experiment_contract(consumer.P064_SYSTEM_KIND['B'])
    assert new_contract['rep256'] is True
    assert 'rep256' not in parent_contract
    assert new_contract['status'] == 'P064_REPRESENTATIVE256_DUAL_FNO_MANIFEST_VERIFIED'
    assert new_contract['aero_kind'] == 'P064_REPRESENTATIVE256_AERODYNAMIC_CHECKPOINT'


def test_zero_closure_initial_fit_is_valid_but_negative_is_rejected(tmp_path):
    payload = manifest_payload(tmp_path, closures=0)
    consumer._validate_rep256_protocol(tmp_path, payload)
    assert producer_metadata(protocol_sha=payload['training_protocol_sha256'], closures=0) \
        == consumer.rep256_checkpoint_metadata(payload)
    payload['optimizer_closures'] = -1
    with pytest.raises(ValueError, match='closure count'):
        consumer._validate_rep256_protocol(tmp_path, payload)


def test_actual_producer_manifest_passes_new_consumer_and_mutations_fail(tmp_path):
    manifest = actual_producer_manifest(tmp_path)
    identity = consumer.validate_dual_fno_manifest(
        manifest, expected_sha256=consumer.sha256(manifest))
    assert identity.payload['kind'] == consumer.REP256_KIND
    assert identity.payload['precision_protocol'] == consumer.REP256_PRECISION
    assert identity.aerodynamic.metadata_kind == 'P064_REPRESENTATIVE256_AERODYNAMIC_CHECKPOINT'
    payload = json.loads(manifest.read_text())
    payload['precision_protocol'] = consumer.PRECISION_PROTOCOL
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='precision'):
        consumer.validate_dual_fno_manifest(manifest)


@pytest.mark.parametrize('mutation', ['old_adam', 'wrong_panel', 'wrong_protocol',
                                      'string_closures'])
def test_manifest_mutations_fail_closed(tmp_path, mutation):
    payload = manifest_payload(tmp_path)
    if mutation == 'old_adam':
        payload['optimizer_steps'] = 32
    elif mutation == 'wrong_panel':
        payload['panel_receipt_sha256'] = '0' * 64
    elif mutation == 'wrong_protocol':
        payload['training_semantics'] = dict(payload['training_semantics'], points=255)
    else:
        payload['optimizer_closures'] = '137'
    with pytest.raises(ValueError):
        consumer._validate_rep256_protocol(tmp_path, payload)


def test_runtime_precision_is_per_kind_and_restored():
    before = (torch.get_float32_matmul_precision(),
              torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert consumer.validate_runtime_precision(consumer.REP256_PRECISION) == consumer.REP256_PRECISION
        with pytest.raises(ValueError):
            consumer.validate_runtime_precision()
    finally:
        torch.set_float32_matmul_precision(before[0])
        torch.backends.cuda.matmul.allow_tf32 = before[1]
        torch.backends.cudnn.allow_tf32 = before[2]
