"""CUDA-hidden official dual-checkpoint load for terminal H1-only candidate F."""

import hashlib
import inspect
import json
import os
from pathlib import Path


ROOT = Path("/workspace/fluid_control")
SOURCE = ROOT / "artifacts/p064_h1_only_development_source_20261007_immutable"
OUTPUT = ROOT / "artifacts/p064_h1_only_candidate_f_cpu_load_20261007"
INPUTS = {
    "manifest": (
        ROOT / "artifacts/p064_h1_only_candidate_f_20261007/dual_model_manifest.json",
        "cabd2794ba187e8f303980543fc32f1386ec26da97c02819ba087b583692cf3b",
    ),
    "result": (
        ROOT / "artifacts/p064_h1_only_candidate_f_20261007/result.json",
        "9e00f1d54a422385782eb955c7217086bf0a59e6b64dcb5f20191f480a4422de",
    ),
    "approval": (
        ROOT / "docs/P064_H1_ONLY_F_TRAINING_APPROVAL_20261007.json",
        "d157e311d70261511144c9fc22a27a468730b4ea2b5ea37705c773b43aea8ef2",
    ),
    "review": (
        ROOT / "docs/P064_H1_ONLY_F_TERMINAL_REVIEW_20261007.md",
        "e40fc80981b5af2457f5e9e1091d62e317219daf5d9f0a02a4121f5b0a6a1413",
    ),
}
DUAL_SHA = "f326cffe5d89d5d336dcc10c9a90fed13e0d61470947d2bc54d79cf2fe761131"
TRAINER_SHA = "9e5bbebd338af02b1d73530c9cb74fc2f840455f56b7bf34ed2bb517be19a22a"
CHECKPOINT_SHA = "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
KIND = "FC_P064_H1_ONLY_K1_FRESH_FORCE_FNO"


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
        "status": "P064_H1_ONLY_OFFICIAL_CPU_LOAD_COMPLETE_NO_FORWARD",
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
