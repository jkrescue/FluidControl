"""Bind terminal E089 evidence into a conversion spec without authorizing it."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path("/workspace/fluid_control")
SOURCE = ROOT / "artifacts/p064_b_symmetry_canonical_ppo_b02_train_acquisition_20261007"
EXPECTED_STATUS = "P064_B_SYMMETRY_CANONICAL_32768_PPO_LONG_CFD_COMPLETE_NOT_ADMISSION"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path, expected: str | None = None) -> Path:
    path = path.absolute()
    require(path.is_file() and not path.is_symlink(), f"regular file: {path}")
    for parent in path.parents:
        require(not parent.is_symlink(), f"symlink ancestor: {parent}")
    path = path.resolve()
    if expected is not None:
        require(sha(path) == expected, f"SHA: {path}")
    return path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_terminal(result: dict, rows: dict) -> None:
    require(result.get("status") == EXPECTED_STATUS, "terminal result status")
    require(result.get("cycles") == 800, "terminal result cycles")
    require(result.get("owned_containers_cleaned") is True, "terminal cleanup")
    require(result.get("source_restart_unchanged") is True, "restart unchanged")
    require(rows.get("completed_cycles") == 800, "progress cycles")
    require(isinstance(rows.get("rows"), list) and len(rows["rows"]) == 800,
            "progress rows")


def validate_unit(unit: str, invocation: str) -> dict:
    raw = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-p", "MainPID", "-p",
         "ExecMainStatus", "-p", "ActiveState", "-p", "SubState", "-p",
         "InvocationID"], text=True, timeout=10)
    state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    require(state.get("MainPID") == "0" and state.get("ExecMainStatus") == "0",
            "source unit not successful terminal")
    require(state.get("InvocationID") == invocation, "source invocation")
    require(state.get("SubState") in ("exited", "dead"), "source unit substate")
    return state


def validate_review(text: str, result_sha: str, invocation: str) -> None:
    require(result_sha in text, "terminal review result SHA")
    require(invocation in text, "terminal review invocation")
    require("conversion" in text.lower() and
            any(word in text.lower() for word in ("ready", "suitable", "可转换")),
            "terminal review conversion disposition")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--terminal-review", type=Path, required=True)
    parser.add_argument("--source-unit", required=True)
    parser.add_argument("--source-invocation", required=True)
    parser.add_argument("--conversion-unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(not args.output.exists(), "exclusive output spec")
    pending_path = checked(args.pending)
    driver = checked(args.driver)
    core_path = checked(args.core)
    review = checked(args.terminal_review)
    result_path = checked(SOURCE / "result.json")
    progress_path = checked(SOURCE / "progress.json")
    result_sha = sha(result_path)
    result = json.loads(result_path.read_text())
    progress = json.loads(progress_path.read_text())
    validate_terminal(result, progress)
    unit_state = validate_unit(args.source_unit, args.source_invocation)
    validate_review(review.read_text(), result_sha, args.source_invocation)

    core = load(core_path, "b02_finalizer_core")
    converter = load(driver, "b02_finalizer_converter")
    base = converter.load_base()
    selected, _, _ = converter.selection(progress, core)
    inventory = base.selected_inventory(SOURCE, selected, lambda: None)
    require(len([key for key in inventory if key.endswith(("/U", "/p"))]) == 1602,
            "selected U/p inventory")

    spec = json.loads(pending_path.read_text())
    require(spec["execution_authorized"] is False, "pending must remain unauthorized")
    spec["status"] = "B02_CONVERSION_REAL_INPUTS_BOUND_AWAITING_LEAD_APPROVAL"
    require(args.conversion_unit ==
            "fluid-control-p064-b02-controlled-train-conversion-20261007.service",
            "conversion unit")
    spec["unit"] = args.conversion_unit
    spec["driver"] = {"path": str(driver.relative_to(ROOT)), "sha256": sha(driver)}
    spec["core"] = {"path": str(core_path.relative_to(ROOT)), "sha256": sha(core_path)}
    spec["source_result"]["sha256"] = result_sha
    spec["progress"]["sha256"] = sha(progress_path)
    spec["terminal_review"] = {
        "path": str(review.relative_to(ROOT)), "sha256": sha(review)}
    spec["source_execution"] = {
        "unit": args.source_unit, "invocation_id": args.source_invocation,
        "terminal_state": unit_state}
    spec["selected_source_inventory"] = inventory
    spec["source_inventory_sha256"] = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": spec["status"], "spec_sha256": sha(args.output),
                      "source_files": len(inventory)}, sort_keys=True))


if __name__ == "__main__":
    main()
