import hashlib,importlib.util,json,sys
from pathlib import Path
import pytest
R=Path('/workspace/fluid_control');S=Path(__file__).resolve().parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def test_actual_producer_protocol_consumption_and_reject_mutations(tmp_path):
    pytest.importorskip('physicsnemo', reason='Run this fixture in the pinned Curator/PhysicsNeMo environment')
    producer=load(S/'scripts/train_p064_temporal_increment.py','producer')
    consumer=load(S/'scripts/dual_fno_temporal_increment.py','consumer')
    source=R/'artifacts/fcp064_training_source_20261006_immutable'
    schedule=load(source/'src/fluid_control/p064_controlled_aero_ab.py','schedule')
    order=json.loads((source/'parent_order.json').read_text())
    p=producer.protocol('B',schedule,order)
    payload=json.loads((R/'artifacts/fcp064_controlled_aero_arm_b_20261006/dual_model_manifest.json').read_text())
    contract=consumer._experiment_contract('FC_P064_TEMPORAL_INCREMENT_AUX_K1_FRESH_FORCE_FNO')
    def write():
        target=tmp_path/'training_protocol.json';target.write_text(json.dumps(p))
        payload.update(training_semantics=p,training_protocol_file=target.name,training_protocol_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    write();consumer._validate_p026_protocol(tmp_path,payload,contract)
    with pytest.raises(ValueError):consumer._validate_p026_protocol(tmp_path,payload,consumer._experiment_contract(consumer.P064_SYSTEM_KIND['B']))
    p['temporal_increment']['weight']=.5;write()
    with pytest.raises(ValueError):consumer._validate_p026_protocol(tmp_path,payload,contract)
    p['temporal_increment']['weight']=1.;p['history_objective_sha256']='0'*64;write()
    with pytest.raises(ValueError):consumer._validate_p026_protocol(tmp_path,payload,contract)
