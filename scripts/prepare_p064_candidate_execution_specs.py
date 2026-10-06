"""Build full-schema, non-authorizing P064 PPO/CFD specs from reviewed executions."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


PPO_TEMPLATE_SHA = "1cd1d5182fd7e7a3eed11060e7a6ffe9fad840c776f021f239f059525a515132"
CFD_TEMPLATE_SHA = "87944e807a68caab6ce7a46e01207e1d7ba432b86ccd639b6f47424de481c64c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pending_ppo(template, args):
    value = copy.deepcopy(template)
    old_runner = value["runner"]
    old_supervisor = next(
        path for path in value["source_files"] if path.endswith("supervise_exploratory_diverse_h5_32768_ppo.py")
    )
    value["source_files"].pop(old_runner)
    value["source_files"].pop(old_supervisor)
    for path in (args.ppo_runner, args.ppo_supervisor):
        value["source_files"][str(path.resolve())] = sha(path)
    package_root = args.runtime_package_root.resolve()
    package = package_root / "fluid_control"
    require((package / "__init__.py").is_file(), "complete fluid_control package required")
    package_files = sorted(package.rglob("*.py"))
    require(package_files, "empty runtime fluid_control package")
    for path in package_files:
        value["source_files"][str(path.resolve())] = sha(path)
    value["runner"] = str(args.ppo_runner.resolve())
    value["supervisor"] = str(args.ppo_supervisor.resolve())
    value["import_bindings"]["fluid_control.dual_control_contract"] = str(
        (package / "dual_control_contract.py").resolve()
    )
    for module in tuple(value["import_bindings"]):
        if module.startswith("fluid_control."):
            path = package.joinpath(*module.split(".")[1:]).with_suffix(".py")
            require(path.is_file(), f"missing runtime package module: {module}")
            value["import_bindings"][module] = str(path.resolve())
    value["pythonpath"] = [str(package_root), *value["pythonpath"]]
    value.update(
        status="P064_CANDIDATE_DIVERSE_H5_32768_PPO_PREPARATION_ONLY_NOT_APPROVED",
        intended_execution_status="P064_CANDIDATE_DIVERSE_H5_32768_PPO_EXECUTION_APPROVED",
        execution_authorized=False,
        reviewed_by_lead=False,
        candidate_arm=None,
        candidate_manifest_kind=None,
        candidate_manifest_sha256=None,
        candidate_terminal_result={"path": None, "sha256": None},
        candidate_terminal_review={"path": None, "sha256": None},
        supervision_output=None,
        output=None,
        lead_statement=None,
    )
    value["inputs"]["manifest"] = {"path": None, "sha256": None}
    value["interpretation"] = (
        "Full fixed 32768/24-reset schema retained; candidate/proofs/output are deliberately null. "
        "This pending file cannot authorize or execute PPO."
    )
    bindings = (
        args.candidate_arm,
        args.candidate_manifest,
        args.candidate_result,
        args.candidate_review,
    )
    require(all(item is None for item in bindings) or all(item is not None for item in bindings),
            "candidate binding must be wholly omitted or wholly supplied")
    if args.candidate_arm is not None:
        require(args.candidate_arm in ("A", "B"), "candidate arm must be A or B")
        manifest = args.candidate_manifest.resolve()
        result = args.candidate_result.resolve()
        review = args.candidate_review.resolve()
        require(manifest.parent == result.parent, "candidate manifest/result directory differs")
        value["candidate_arm"] = args.candidate_arm
        value["candidate_manifest_kind"] = (
            f"FC_P064_ARM_{args.candidate_arm}_CONTROLLED_AERO_FORCE_FNO"
        )
        value["candidate_manifest_sha256"] = sha(manifest)
        value["inputs"]["manifest"] = {"path": str(manifest), "sha256": sha(manifest)}
        value["candidate_terminal_result"] = {"path": str(result), "sha256": sha(result)}
        value["candidate_terminal_review"] = {"path": str(review), "sha256": sha(review)}
        value["interpretation"] = (
            f"Full fixed 32768/24-reset schema retained and actual P064 arm {args.candidate_arm} "
            "candidate/proofs are bound; output and execution authorization remain deliberately null. "
            "This pending file cannot authorize or execute PPO."
        )
    return value


def pending_cfd(template, args):
    value = copy.deepcopy(template)
    value.update(
        status="P064_CANDIDATE_PROJECTED_32768_PPO_LONG_CFD_PREPARATION_ONLY_NOT_APPROVED",
        intended_execution_status="P064_CANDIDATE_PROJECTED_32768_PPO_LONG_CFD_EXECUTION_APPROVED",
        execution_authorized=False,
        reviewed_by_lead=False,
        immutable_driver=str(args.cfd_runner.resolve()),
        driver_sha256=sha(args.cfd_runner),
        candidate_arm=None,
        candidate_manifest_sha256=None,
        training_payload=None,
        training_execution={"unit": None, "invocation_id": None},
        output=None,
        lead_statement=None,
    )
    for key in ("training_result", "policy", "vecnormalize", "training_approval", "training_review"):
        value["inputs"][key] = {"path": None, "sha256": None}
    value["inputs"].pop("original_long_result")
    value["interpretation"] = (
        "Full retained paired800 physics/resources/source schema; future P064 PPO/candidate bindings are null. "
        "No 124-cycle screen, execution, or admission is authorized."
    )
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-template", type=Path, required=True)
    parser.add_argument("--cfd-template", type=Path, required=True)
    parser.add_argument("--ppo-runner", type=Path, required=True)
    parser.add_argument("--ppo-supervisor", type=Path, required=True)
    parser.add_argument("--cfd-runner", type=Path, required=True)
    parser.add_argument("--runtime-package-root", type=Path, required=True)
    parser.add_argument("--candidate-arm")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--candidate-result", type=Path)
    parser.add_argument("--candidate-review", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.ppo_template) == PPO_TEMPLATE_SHA, "PPO template changed")
    require(sha(args.cfd_template) == CFD_TEMPLATE_SHA, "CFD template changed")
    ppo = json.loads(args.ppo_template.read_text())
    cfd = json.loads(args.cfd_template.read_text())
    args.output_dir.mkdir(parents=False, exist_ok=False)
    write(args.output_dir / "PPO_PENDING.json", pending_ppo(ppo, args))
    write(args.output_dir / "CFD_PENDING.json", pending_cfd(cfd, args))


if __name__ == "__main__":
    main()
