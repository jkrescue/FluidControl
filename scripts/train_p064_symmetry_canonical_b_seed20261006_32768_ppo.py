"""Fresh-init 32768 PPO with a fixed symmetry-canonical coordinate adapter.

The PPO algorithm, B candidate, reset panel, reward, optimization budget, and
resources are retained. The sole scientific change is the coordinate adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

CONFIG_SHA = "07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9"
NORM_SHA = "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
BASELINE_SHA = "b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7"
STATUS = "P064_B_SYMMETRY_CANONICAL_H5_32768_PPO_EXECUTION_APPROVED"
P064_KINDS = {
    "A": "FC_P064_ARM_A_CONTROLLED_AERO_FORCE_FNO",
    "B": "FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO",
}
REQUIRED_IMPORTS = {
    "exploratory_h5_hydrogym",
    "exploratory_diverse_h5_resets",
    "physicsnemo.datapipes.readers.hdf5",
    "fluid_control.openfoam_force_history",
    "train_tandem_fno",
    "train_full40_hydrogym_ppo_canonical",
    "fluid_control.dual_fno",
    "fluid_control.dual_control_contract",
    "symmetry_canonical_wrapper",
    "fluid_control.full40_canonical_hydrogym",
    "fluid_control.tandem_hydrogym",
    "hydrogym",
    "hydrogym.core",
    "gymnasium",
    "stable_baselines3",
    "stable_baselines3.common.on_policy_algorithm",
    "stable_baselines3.common.vec_env.dummy_vec_env",
    "physicsnemo.models.fno.fno",
    "physicsnemo.utils.checkpoint",
}
PROTOCOL = {
    "schema_version": 1,
    "reset_panel": "four_phases_each_zero_frame0_then_five_train_cases_frame62",
    "reset_count": 24,
    "reset_order": ["zero:0", "m075:62", "m0375:62", "zero:62", "p0375:62", "p075:62"],
    "reset_selection": "deterministic_phase_cycle_no_reward_selection",
    "timesteps": 32768,
    "episode_steps": 5,
    "environments": 4,
    "seed": 20261006,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 4,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "observation_dimension": 69,
    "control_dt": 0.1,
    "action_limit": 0.75,
    "action_delta_limit": 0.1,
    "force_window_samples": 62,
    "reward": "canonical_joint_v1",
    "policy": "MlpPolicy",
    "norm_obs": False,
    "norm_reward": False,
    "final_policy_only": True,
    "device": "cuda:0",
    "gpu_memory_fraction": 0.06,
    "startup_available_gib": 50,
    "runtime_available_gib": 22,
    "physical_reserve_gib": 20,
    "deadline_seconds": 1800,
    "scientific_admission": False,
    "cfd_execution": False,
    "symmetry_adapter": "physical69_max_abs_first_tie_fixed_plus_v1",
    "symmetry_training_and_deployment_same_bytes": True,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def same_json(actual, expected):
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def fixed_panel():
    return [
        {"case": f"matched_start_acquisition_train_b{phase}_{family}", "frame": frame}
        for phase in ("00", "02", "04", "06")
        for family, frame in (("zero", 0), ("m075", 62), ("m0375", 62),
                              ("zero", 62), ("p0375", 62), ("p075", 62))
    ]


def read_packet_verification(spec):
    binding = spec["packet_verification"]
    require(sha(binding["path"]) == binding["sha256"], "packet verification changed")
    proof = json.loads(Path(binding["path"]).read_text())
    require(proof["status"] == "DIVERSE_H5_REAL_RESET_PACKETS_VERIFIED_CPU_NOT_TRAINING",
            "actual CPU packet verification required")
    require(proof["scientific_admission"] is False and len(proof["packets"]) == 24,
            "packet verification scope")
    require(proof["packet_count"] == 24 and proof["no_model_loaded"] is True
            and proof["no_optimizer"] is True and proof["no_gpu"] is True
            and proof["no_cfd"] is True, "CPU-only proof required")
    adapter_path = spec["import_bindings"]["exploratory_diverse_h5_resets"]
    require(spec["source_files"][adapter_path] in proof["source_sha256"].values(),
            "CPU packet proof used different adapter")
    observed = [{"case": p["case"], "frame": p["frame"]} for p in proof["packets"]]
    require(observed == fixed_panel(), "packet proof membership/order differs")
    return proof


def validate_candidate_proofs(spec):
    arm = spec["candidate_arm"]
    result_binding = spec["candidate_terminal_result"]
    review_binding = spec["candidate_terminal_review"]
    for label, binding in (("terminal result", result_binding), ("terminal review", review_binding)):
        require(set(binding) == {"path", "sha256"}, f"candidate {label} binding")
        require(sha(binding["path"]) == binding["sha256"], f"candidate {label} changed")
    result = json.loads(Path(result_binding["path"]).read_text())
    require(
        result.get("status") == f"FC_P064_ARM_{arm}_TRAINING_COMPLETE_NOT_ADMISSION"
        and result.get("arm") == arm
        and result.get("optimizer_steps") == 32
        and result.get("training_windows") == 256
        and result.get("official_fresh_reload_verified") is True
        and result.get("scientific_admission") is False,
        "reviewed P064 candidate terminal proof differs",
    )
    manifest_path = Path(spec["inputs"]["manifest"]["path"]).resolve()
    result_path = Path(result_binding["path"]).resolve()
    require(manifest_path.parent == result_path.parent, "candidate result/manifest directory differs")
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("status") == f"FC_P064_ARM_{arm}_DUAL_FNO_MANIFEST_VERIFIED"
        and manifest.get("kind") == spec["candidate_manifest_kind"]
        and manifest.get("arm") == arm
        and manifest.get("training_windows") == 256
        and manifest.get("optimizer_steps") == 32
        and isinstance(manifest.get("aerodynamic", {}).get("model_sha256"), str)
        and len(manifest["aerodynamic"]["model_sha256"]) == 64,
        "candidate result/manifest relationship differs",
    )
    review = json.loads(Path(review_binding["path"]).read_text())
    require(
        review.get("status") == "P064_TERMINAL_ENGINEERING_REVIEW_NOT_ADMISSION"
        and review.get("result_sha256") == result_binding["sha256"]
        and review.get("records") == 32
        and review.get("consumed") == 256
        and review.get("producer_official_reload") is True
        and review.get("unit", {}).get("Result") == "success"
        and review.get("unit", {}).get("ExecMainStatus") == "0",
        "independent candidate terminal review differs",
    )
    return result


def validate_spec(spec):
    require(
        spec.get("status") == STATUS and spec.get("execution_authorized") is True,
        "separate exploratory PPO execution approval required",
    )
    require(same_json(spec.get("protocol"), PROTOCOL), "fixed protocol differs")
    expected = {
        "config": CONFIG_SHA,
        "normalization": NORM_SHA,
        "baseline": BASELINE_SHA,
    }
    for name, digest in expected.items():
        require(
            spec["inputs"][name]["sha256"] == digest, f"fixed {name} identity differs"
        )
    arm = spec.get("candidate_arm")
    require(arm in P064_KINDS, "explicit P064 candidate arm required")
    candidate_sha = spec.get("candidate_manifest_sha256")
    require(
        isinstance(candidate_sha, str)
        and len(candidate_sha) == 64
        and spec["inputs"]["manifest"]["sha256"] == candidate_sha,
        "candidate manifest binding differs",
    )
    require(
        spec.get("candidate_manifest_kind") == P064_KINDS[arm],
        "P064 candidate kind differs",
    )
    validate_candidate_proofs(spec)
    cases = [
        f"matched_start_acquisition_train_b{p}_zero" for p in ("00", "02", "04", "06")
    ]
    require(spec["train_cases"] == cases, "four fixed train starts required")
    all_cases = [
        f"matched_start_acquisition_train_b{phase}_{family}"
        for phase in ("00", "02", "04", "06")
        for family in ("m075", "m0375", "zero", "p0375", "p075")
    ]
    require(len(spec["train_hdf"]) == 20, "exact twenty base-train files required")
    require(spec["reset_panel"] == fixed_panel(), "fixed24 reset order differs")
    require(set(spec["packet_verification"]) == {"path", "sha256"}, "packet proof binding")
    data = Path(spec["data_root"]).resolve()
    for row, case in zip(spec["train_hdf"], all_cases, strict=True):
        require(
            Path(row["path"]).resolve() == data / "train" / f"{case}.h5",
            "non-train or different HDF path",
        )
    require(
        spec["runtime_packages"]["stable-baselines3"] == "2.7.1"
        and spec["runtime_packages"]["gymnasium"] == "1.2.3",
        "SB3/Gymnasium pins differ",
    )
    require(
        set(spec["import_bindings"]) == REQUIRED_IMPORTS,
        "required import closure differs",
    )
    require(
        Path(spec["inputs"]["normalization"]["path"]).resolve()
        == data / "normalization.json",
        "environment normalization is not the pinned normalization",
    )
    return spec


def validate_files(spec):
    """Approval-bound small source/runtime metadata first; dataset reads only in worker."""
    for table in (spec["source_files"], spec["runtime_sources"]):
        require(table, "empty source closure")
        for name, digest in table.items():
            path = Path(name)
            require(
                path.is_absolute() and path.is_file() and not path.is_symlink(),
                f"missing or symlink source: {path}",
            )
            require(sha(path) == digest, f"source changed: {path}")
    require(
        str(Path(__file__).resolve()) in spec["source_files"], "runner not source-bound"
    )
    for package, version in spec["runtime_packages"].items():
        require(
            importlib.metadata.version(package) == version,
            f"runtime changed: {package}",
        )
    for row in [
        *spec["inputs"].values(),
        *spec["train_hdf"],
        *spec["train_provenance_files"],
    ]:
        require(sha(row["path"]) == row["sha256"], f"input changed: {row['path']}")


def model_digest(network):
    h = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen(network):
    require(not network.training, "FNO must remain eval")
    require(
        all(not p.requires_grad and p.grad is None for p in network.parameters()),
        "FNO parameter trainability/grad changed",
    )


def precision(torch):
    return {
        "matmul": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def execute(spec, output):
    start = time.monotonic()
    output.mkdir(parents=False, exist_ok=False)
    write(output / "source_spec.json", spec)
    # External supervisor monitors MemAvailable before this includes heavy imports/load.
    validate_files(spec)
    proof = read_packet_verification(spec)
    import numpy as np
    import torch
    from fluid_control.dual_control_contract import p026_runtime_binding
    from fluid_control.dual_fno import load_dual_fno
    from symmetry_canonical_wrapper import CanonicalSymmetryWrapper
    from omegaconf import OmegaConf
    from physicsnemo.utils.checkpoint import load_checkpoint
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train_full40_hydrogym_ppo_canonical import (
        TrainingDiagnosticsAccumulator,
        validate_train20_baselines,
    )
    from train_tandem_fno import build_model

    from exploratory_h5_hydrogym import make_exploratory_env
    from exploratory_diverse_h5_resets import load_packet, packet_identity, wrap_phase_cycle

    # Verify actual imported files, not just an unrelated file named in an approval.
    for module_name, expected_path in spec["import_bindings"].items():
        module = importlib.import_module(module_name)
        actual = Path(module.__file__).resolve()
        require(
            actual == Path(expected_path).resolve(), f"unexpected import: {module_name}"
        )
        table = {**spec["source_files"], **spec["runtime_sources"]}
        require(
            str(actual) in table and sha(actual) == table[str(actual)],
            "import hash differs",
        )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "one CUDA device required",
    )
    torch.cuda.set_per_process_memory_fraction(0.06, 0)
    random.seed(PROTOCOL["seed"])
    np.random.seed(PROTOCOL["seed"])
    torch.manual_seed(PROTOCOL["seed"])
    torch.cuda.manual_seed_all(PROTOCOL["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    inputs = spec["inputs"]
    cfg = OmegaConf.load(inputs["config"]["path"])
    network, identity = load_dual_fno(
        Path(inputs["manifest"]["path"]),
        cfg,
        torch.device("cuda:0"),
        build_model=build_model,
        load_checkpoint=load_checkpoint,
        expected_manifest_sha256=spec["candidate_manifest_sha256"],
    )
    before_precision = precision(torch)
    network.eval().requires_grad_(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    effective = {"matmul": "highest", "cuda_tf32": False, "cudnn_tf32": False}
    require(precision(torch) == effective, "post-load precision differs")
    runtime = p026_runtime_binding(identity)
    require(runtime and runtime["profile"] == "p026_k1", "K1 history binding required")
    require(
        runtime["manifest_kind"] == spec["candidate_manifest_kind"],
        "P064 runtime candidate identity differs",
    )
    initial = model_digest(network)
    frozen(network)
    baselines = validate_train20_baselines(Path(inputs["baseline"]["path"]))
    phases = ("00", "02", "04", "06")
    packets = [
        load_packet(spec["data_root"], spec["cases_root"], row["case"], row["frame"])
        for row in fixed_panel()
    ]
    observed_packets = [packet_identity(packet) for packet in packets]
    require(same_json(observed_packets, proof["packets"]), "reset packets differ from CPU verification")
    write(output / "reset_packets.json", {"packets": observed_packets, "verification": spec["packet_verification"]})
    phase_wrappers = []

    def make(case, phase):
        anchor = make_exploratory_env(
                data=spec["data_root"],
                case=case,
                network=network,
                checkpoint_epoch=identity.aerodynamic.epoch,
                baseline=baselines[f"b{phase}"],
                device="cuda:0",
                cases_root=spec["cases_root"],
                fno_history_runtime=runtime,
            )
        offset = phases.index(phase) * 6
        wrapped = wrap_phase_cycle(anchor, packets[offset:offset + 6])
        phase_wrappers.append(wrapped)
        return Monitor(CanonicalSymmetryWrapper(wrapped))

    env = VecNormalize(
        DummyVecEnv(
            [
                lambda c=c, p=p: make(c, p)
                for c, p in zip(spec["train_cases"], phases, strict=True)
            ]
        ),
        norm_obs=False,
        norm_reward=False,
    )
    diagnostics = TrainingDiagnosticsAccumulator()

    class Audit(BaseCallback):
        def _on_step(self):
            frozen(network)
            require(precision(torch) == effective, "effective precision changed")
            require(
                time.monotonic() - start < PROTOCOL["deadline_seconds"],
                "training deadline",
            )
            infos = self.locals["infos"]
            diagnostics.record(infos)
            with (output / "transitions.jsonl").open("a") as stream:
                for index, info in enumerate(infos):
                    row = {
                        "num_timesteps": self.num_timesteps,
                        "env_index": index,
                        **{
                            k: v for k, v in info.items() if k != "terminal_observation"
                        },
                    }
                    if "terminal_observation" in info:
                        obs = np.asarray(info["terminal_observation"])
                        require(
                            obs.shape == (69,) and np.isfinite(obs).all(),
                            "bad terminal observation",
                        )
                        row["terminal_observation"] = obs.tolist()
                    json.dump(row, stream, allow_nan=False)
                    stream.write("\n")
            return True

    try:
        policy = PPO(
            "MlpPolicy",
            env,
            device="cuda:0",
            seed=PROTOCOL["seed"],
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
        )
        fno_ids = {id(p) for p in network.parameters()}
        require(
            not any(
                id(p) in fno_ids
                for g in policy.policy.optimizer.param_groups
                for p in g["params"]
            ),
            "FNO leaked into PPO optimizer",
        )
        policy_initial = model_digest(policy.policy)
        optimizer_steps = []

        def after_optimizer_step(optimizer, positional, keyword):
            require(
                all(
                    torch.isfinite(p).all()
                    and (p.grad is None or torch.isfinite(p.grad).all())
                    for p in policy.policy.parameters()
                ),
                "nonfinite PPO parameter/gradient",
            )
            optimizer_steps.append(
                {
                    "optimizer_step": len(optimizer_steps) + 1,
                    "num_timesteps": policy.num_timesteps,
                }
            )

        hook = policy.policy.optimizer.register_step_post_hook(after_optimizer_step)
        policy.set_logger(configure(str(output), ["json"]))
        policy.learn(total_timesteps=32768, callback=Audit(), progress_bar=False)
        policy.logger.dump(
            step=policy.num_timesteps
        )  # retain final update metrics too.
        hook.remove()
        require(policy.num_timesteps == 32768, "training budget differs")
        require(
            policy._n_updates == 256 and len(optimizer_steps) == 512,
            "PPO epoch/optimizer update count differs",
        )
        frozen(network)
        require(model_digest(network) == initial, "FNO tensors changed")
        require(
            all(torch.isfinite(p).all() for p in policy.policy.parameters()),
            "nonfinite policy",
        )
        policy_terminal = model_digest(policy.policy)
        require(policy_initial != policy_terminal, "PPO parameters did not update")
        policy.save(output / "ppo_final.zip")
        env.training = False
        env.save(output / "vecnormalize.pkl")
        write(
            output / "result.json",
            {
                "status": "P064_B_SYMMETRY_CANONICAL_H5_32768_PPO_TRAINING_COMPLETE_NOT_ADMISSION",
                "candidate_arm": spec["candidate_arm"],
                "candidate_manifest_sha256": spec["candidate_manifest_sha256"],
                "candidate_manifest_kind": spec["candidate_manifest_kind"],
                "protocol": PROTOCOL,
                "timesteps": policy.num_timesteps,
                "fno_runtime": runtime,
                "reset_packet_verification": spec["packet_verification"],
                "reset_counts_by_phase": {p: list(w.reset_counts) for p, w in zip(phases, phase_wrappers, strict=True)},
                "fno_tensors_unchanged": True,
                "fno_tensor_sha256": initial,
                "precision_before_override": before_precision,
                "policy_tensor_sha256_before": policy_initial,
                "policy_tensor_sha256_after": policy_terminal,
                "ppo_n_updates": policy._n_updates,
                "optimizer_steps": optimizer_steps,
                "precision_effective": precision(torch),
                "diagnostics": diagnostics.snapshot(),
                "artifacts": {
                    p: sha(output / p)
                    for p in (
                        "ppo_final.zip",
                        "vecnormalize.pkl",
                        "transitions.jsonl",
                        "source_spec.json",
                        "progress.json",
                        "reset_packets.json",
                    )
                },
                "scientific_admission": False,
                "cfd_executed": False,
                "symmetry_adapter": PROTOCOL["symmetry_adapter"],
                "wall_seconds": time.monotonic() - start,
            },
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(args.approval) == args.approval_sha256, "approval changed")
    spec = validate_spec(json.loads(args.approval.read_text()))
    require(args.output.resolve() == Path(spec["output"]).resolve(), "output differs")
    require(
        os.environ.get("P064_B_SYMMETRY_CANONICAL_H5_32768_SUPERVISED") == args.approval_sha256,
        "approved memory supervisor required",
    )
    if args.execute:
        execute(spec, args.output)
    else:
        print(json.dumps({"status": "PREPARATION_ONLY", "protocol": PROTOCOL}))


if __name__ == "__main__":
    main()
