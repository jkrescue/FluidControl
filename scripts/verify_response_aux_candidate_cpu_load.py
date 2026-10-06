"""CUDA-hidden official dual-checkpoint load for the terminal response-aux candidate."""

import hashlib
import inspect
import json
import os
from pathlib import Path


ROOT = Path("/workspace/fluid_control")
SOURCE = ROOT / "artifacts/p064_response_aux_development_source_20261007_immutable"
OUTPUT = ROOT / "artifacts/p064_response_aux_candidate_e_cpu_load_20261007"
INPUTS = {
    "manifest": (
        ROOT / "artifacts/p064_response_aux_candidate_e_20261007/dual_model_manifest.json",
        "50a87f0129f3372b1d6d3c2a48d38f6507e8e7822c61e39ad027ecdba76bc92a",
    ),
    "result": (
        ROOT / "artifacts/p064_response_aux_candidate_e_20261007/result.json",
        "a489e6a5ba221f7afc5acdac251dba8a7d20b12a447386a14a677320deeba3cd",
    ),
    "approval": (
        ROOT / "docs/P064_RESPONSE_AUX_E_TRAINING_APPROVAL_20261007.json",
        "93efb40cf57c3e260c4b31b86b767601ed9562f46102e4b8d2b4eef75ef96e39",
    ),
    "review": (
        ROOT / "docs/P064_RESPONSE_AUX_E_TERMINAL_REVIEW_20261007.md",
        "ac4c4ede971b7db0a475fed25956da63ebda55d04284498f627f50a5f066ba61",
    ),
}
DUAL_SHA = "d5e3ac6bfe893ca14f24eea89b28db578f11c4a182d4d042b45cedd50da9e5e1"
TRAINER_SHA = "9e5bbebd338af02b1d73530c9cb74fc2f840455f56b7bf34ed2bb517be19a22a"
CHECKPOINT_SHA = "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
KIND = "FC_P064_RESPONSE_AUX_K1_FRESH_FORCE_FNO"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU proof must hide CUDA")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, expected in INPUTS.values():
        if sha(path) != expected:
            raise ValueError("terminal input differs: " + str(path))

    import torch
    from omegaconf import OmegaConf
    from physicsnemo.utils import load_checkpoint
    from fluid_control import dual_fno
    import train_tandem_fno

    if (
        sha(inspect.getfile(dual_fno)) != DUAL_SHA
        or sha(inspect.getfile(train_tandem_fno)) != TRAINER_SHA
        or sha(inspect.getfile(load_checkpoint)) != CHECKPOINT_SHA
    ):
        raise ValueError("actual import source differs")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest_path = INPUTS["manifest"][0]
    adapter, identity = dual_fno.load_dual_fno(
        manifest_path,
        OmegaConf.load(SOURCE / "training_config.yaml"),
        "cpu",
        build_model=train_tandem_fno.build_model,
        load_checkpoint=load_checkpoint,
        expected_manifest_sha256=INPUTS["manifest"][1],
    )
    if identity.payload["kind"] != KIND:
        raise ValueError("loaded candidate kind differs")
    result = json.loads(INPUTS["result"][0].read_text())
    flow_sha = tensor_state_sha256(adapter.flow_model)
    aero_sha = tensor_state_sha256(adapter.aerodynamic_model)
    if flow_sha != result["flow_tensor_sha256"]:
        raise ValueError("loaded flow tensor differs")
    if aero_sha != result["aerodynamic_terminal_tensor_sha256"]:
        raise ValueError("loaded aerodynamic tensor differs")
    if any(parameter.grad is not None for model in (adapter.flow_model, adapter.aerodynamic_model)
           for parameter in model.parameters()):
        raise RuntimeError("CPU load unexpectedly created gradients")
    OUTPUT.mkdir()
    receipt = {
        "status": "P064_RESPONSE_AUX_OFFICIAL_CPU_LOAD_COMPLETE_NO_FORWARD",
        "inputs": {name: {"path": str(path), "sha256": expected}
                   for name, (path, expected) in INPUTS.items()},
        "kind": KIND,
        "manifest_sha256": identity.manifest_sha256,
        "flow_tensor_sha256": flow_sha,
        "aerodynamic_tensor_sha256": aero_sha,
        "dual_source": {"path": inspect.getfile(dual_fno), "sha256": DUAL_SHA},
        "trainer_source": {"path": inspect.getfile(train_tandem_fno), "sha256": TRAINER_SHA},
        "checkpoint_source": {"path": inspect.getfile(load_checkpoint), "sha256": CHECKPOINT_SHA},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "forward_calls": 0,
        "optimizer_steps": 0,
    }
    path = OUTPUT / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(path), "sha256": sha(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
