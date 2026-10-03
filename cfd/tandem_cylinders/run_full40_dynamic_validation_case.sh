#!/usr/bin/env bash
# Run one reviewed real-OpenFOAM dynamic validation case. Preflight is default.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd "$root/../.." && pwd -P)"
expected_repo="/workspace/fluid_control"
name="${1:-}"
mode="${2:---preflight-only}"
predeclaration="$repo/artifacts/tandem_cylinders/full40_dynamic_validation_predeclared_20261003.json"
predeclaration_sha="0478c8532bd2ded504ccd5f89303001eb8359f69b3036f296e31428a085d1272"
execute_token="EXECUTE_REVIEWED_FULL40_DYNAMIC_VALIDATION"

[[ "$repo" == "$expected_repo" ]] || {
  echo "dynamic validation is Spark-local; Worker temporary copies require a separate reviewed transfer protocol" >&2
  exit 2
}

[[ "$name" =~ ^full40_dynamic_validation_b(01|05)_(minus|zero|plus)$ ]] \
  || { echo "case is outside the six-case dynamic validation panel" >&2; exit 2; }
[[ ${#predeclaration_sha} -eq 64 ]] \
  || { echo "execution disabled until predeclaration SHA is reviewed" >&2; exit 2; }
[[ -f "$predeclaration" ]] || { echo "predeclaration missing" >&2; exit 2; }
[[ "$(sha256sum "$predeclaration" | awk '{print $1}')" == "$predeclaration_sha" ]] \
  || { echo "predeclaration SHA differs" >&2; exit 2; }

case_dir="$root/cases/$name"
config="$case_dir/case_config.json"
[[ -f "$config" ]] || { echo "staged case is missing" >&2; exit 2; }
readarray -t contract < <(python3 - "$predeclaration" "$config" "$name" "$predeclaration_sha" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

predecl_path, config_path = map(Path, sys.argv[1:3])
name, expected_sha = sys.argv[3:]
if hashlib.sha256(predecl_path.read_bytes()).hexdigest() != expected_sha:
    raise SystemExit("predeclaration SHA differs")
predecl = json.loads(predecl_path.read_text(encoding="utf-8"))
config = json.loads(config_path.read_text(encoding="utf-8"))
expected = predecl.get("cases", {}).get(name)
if (
    predecl.get("status") != "FULL40_DYNAMIC_VALIDATION_PREDECLARED_NOT_EXECUTED"
    or predecl.get("frozen_test_accessed") is not False
    or expected is None
    or config.get("case") != name
    or config.get("panel") != "full40_dynamic_validation_v1"
    or config.get("predeclaration_sha256") != expected_sha
):
    raise SystemExit("case/predeclaration identity differs")
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"case config differs: {key}")
source = config_path.parent / "source_restart_provenance"
for field, digest in config["source_state_sha256"].items():
    if hashlib.sha256((source / field).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"source-state SHA differs: {field}")
velocity = (config_path.parent / f'{config["source_restart_time"]:g}' / "U").read_text()
patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", velocity, re.S)
table = patch and re.search(r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.S)
actual = [] if not table else [[float(a), float(b)] for a, b in re.findall(
    r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows"))]
if actual != config["action_points"]:
    raise SystemExit("OpenFOAM action table differs")
print(f'{config["start_time"]:g}')
print(f'{config["end_time"]:g}')
print(config["expected_solver_steps"])
PY
)
start="${contract[0]}"
end="${contract[1]}"
steps="${contract[2]}"

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 40 * 1024 * 1024 )) \
  || { echo "MemAvailable is below 40 GiB" >&2; exit 1; }
if pgrep -af '[p]impleFoam|[c]urate_matched_start' >/dev/null; then
  echo "CFD/Curator activity detected; dynamic validation is deferred" >&2
  exit 1
fi
python3 "$repo/scripts/check_dynamic6_qs1_coexistence.py"
if [[ "$mode" == "--preflight-only" ]]; then
  echo "FULL40_DYNAMIC_VALIDATION_PREFLIGHT_OK $name"
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "mode must be --preflight-only or --execute" >&2; exit 2; }
[[ "${FULL40_DYNAMIC_VALIDATION_APPROVAL_TOKEN:-}" == "$execute_token" ]] \
  || { echo "explicit execution token is required" >&2; exit 2; }

log="$case_dir/log.pimpleFoam.full40_dynamic_validation"
qc="$case_dir/solver_log_qc.full40_dynamic_validation.json"
marker="$case_dir/solver_complete.full40_dynamic_validation.json"
[[ ! -e "$log" && ! -e "$qc" && ! -e "$marker" ]] \
  || { echo "refusing existing solver output" >&2; exit 1; }
if find "$case_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | awk -v start="$start" '$1 ~ /^[0-9.]+$/ && $1 > start {found=1} END {exit !found}'; then
  echo "refusing case with solver time directories" >&2
  exit 1
fi

if ! bash "$root/run_openfoam.sh" pimpleFoam -case "/case/cases/$name" >"$log" 2>&1; then
  echo "OpenFOAM failed; log retained" >&2
  exit 1
fi
python3 "$root/check_matched_start_solver_log.py" \
  --log "$log" --expected-steps "$steps" --expected-end "$end" >"$qc"
python3 - "$marker" "$name" "$predeclaration_sha" "$log" "$qc" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

marker, log, qc = Path(sys.argv[1]), Path(sys.argv[4]), Path(sys.argv[5])
payload = {
    "status": "FULL40_DYNAMIC_VALIDATION_SOLVER_COMPLETED_PENDING_PANEL_QC",
    "case": sys.argv[2],
    "predeclaration_sha256": sys.argv[3],
    "solver_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "solver_qc_sha256": hashlib.sha256(qc.read_bytes()).hexdigest(),
}
temporary = marker.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.link(temporary, marker)
temporary.unlink()
PY
