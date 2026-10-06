import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_fcp028_formal_approval.py"
spec = importlib.util.spec_from_file_location("prepare_fcp028_formal_approval", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_generator_cannot_authorize():
    assert module.PENDING_STATUS != module.FINAL_STATUS
    source = SCRIPT.read_text()
    assert '"formal_evaluation_authorized": False' in source
    assert '"reviewed_by_lead": False' in source
    assert '"generator_authorizes_execution": False' in source


def test_exact_candidate_file_contract():
    assert module.FILES == {
        "result.json",
        "training_protocol.json",
        "dual_model_manifest.json",
        "flow/FNO.0.1.mdlus",
        "flow/checkpoint.0.1.pt",
        "aerodynamic/FNO.0.1.mdlus",
        "aerodynamic/checkpoint.0.1.pt",
    }


def test_runtime_pins_are_exact():
    assert module.SOURCE_RECEIPT_SHA == "e4645a4d49359fabe22c6040de0d0ab76667b138f9879d3a24487dfc2a782152"
    assert module.RUNNER_SHA == "7ba31bc02e892b860a2ec9edd9e7187661bedc36c7437c44aa7a078ed221bbd4"
    assert module.NUMERICAL_RUNNER_SHA == "03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3"
