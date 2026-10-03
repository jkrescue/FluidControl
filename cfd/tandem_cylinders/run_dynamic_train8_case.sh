#!/usr/bin/env bash
# Fail-closed one-case runner.  Execution stays disabled until reviewed SHA binding.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd "$root/../.." && pwd -P)"
name="${1:-}"
mode="${2:---preflight-only}"
predecl="$repo/artifacts/tandem_cylinders/dynamic_train8_predeclared_20261003.json"
predecl_sha="4cf4e7c9b9da27b71e58db2e94b0750736b7f79f09a2aebc3ffa97729e882c5a"
token="EXECUTE_REVIEWED_DYNAMIC_TRAIN8"

[[ "$name" =~ ^dynamic_train8_b(00|02|04|06)_(prbs|multisine)$ ]] \
  || { echo "case is outside train8" >&2; exit 2; }
[[ ${#predecl_sha} -eq 64 ]] \
  || { echo "execution disabled until predeclaration SHA is reviewed" >&2; exit 2; }
[[ -f "$predecl" ]] || { echo "predeclaration missing" >&2; exit 2; }
[[ "$(sha256sum "$predecl" | awk '{print $1}')" == "$predecl_sha" ]] \
  || { echo "predeclaration SHA differs" >&2; exit 2; }

case_dir="$root/cases/$name"
config="$case_dir/case_config.json"
[[ -f "$config" ]] || { echo "staged case missing" >&2; exit 2; }
readarray -t contract < <(python3 - "$predecl" "$config" "$name" "$predecl_sha" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
pre, cfg = map(Path, sys.argv[1:3]); name, digest = sys.argv[3:]
if hashlib.sha256(pre.read_bytes()).hexdigest() != digest:
    raise SystemExit("predeclaration SHA differs")
p, c = json.loads(pre.read_text()), json.loads(cfg.read_text())
e = p.get("cases", {}).get(name)
if p.get("status") != "DYNAMIC_TRAIN8_PREDECLARED_NOT_EXECUTED" or e is None:
    raise SystemExit("predeclaration identity differs")
if c.get("case") != name or c.get("panel") != "dynamic_train8_v1" or c.get("predeclaration_sha256") != digest:
    raise SystemExit("case identity differs")
for key, value in e.items():
    if c.get(key) != value:
        raise SystemExit(f"case contract differs: {key}")
if c.get("split") != "train" or c.get("phase_bin") not in [0, 2, 4, 6]:
    raise SystemExit("case is not train-only")
source = cfg.parent / "source_restart_provenance"
for field, expected in c["source_state_sha256"].items():
    actual = hashlib.sha256((source / field).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"source state differs: {field}")
text = (cfg.parent / f'{c["source_restart_time"]:g}' / "U").read_text()
patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", text, re.S)
table = patch and re.search(r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.S)
actual = [] if not table else [[float(a), float(b)] for a, b in re.findall(
    r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows"))]
if actual != c["action_points"]:
    raise SystemExit("OpenFOAM omega table differs")
print(f'{c["run_window"][0]:g}')
print(f'{c["run_window"][1]:g}')
print(c["expected_solver_steps"])
PY
)
start="${contract[0]}"; end="${contract[1]}"; steps="${contract[2]}"

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 40 * 1024 * 1024 )) || { echo "MemAvailable below 40 GiB" >&2; exit 1; }
free_kib="$(df -Pk "$repo" | awk 'NR==2 {print $4}')"
(( free_kib >= 230 * 1024 * 1024 )) || { echo "Spark free disk below 230 GiB launch floor" >&2; exit 1; }
active="$({ pgrep -x pimpleFoam || true; } | wc -l | tr -d ' ')"
(( active < 4 )) || { echo "Spark already has four pimpleFoam cases" >&2; exit 1; }

if [[ "$mode" == "--preflight-only" ]]; then
  echo "DYNAMIC_TRAIN8_PREFLIGHT_OK $name $start $end"
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "mode must be --preflight-only or --execute" >&2; exit 2; }
[[ "${DYNAMIC_TRAIN8_APPROVAL_TOKEN:-}" == "$token" ]] || { echo "explicit token required" >&2; exit 2; }

lock="$case_dir/.dynamic_train8_solver.lock"
mkdir "$lock" 2>/dev/null || { echo "case lock exists" >&2; exit 1; }
log="$case_dir/log.pimpleFoam.dynamic_train8"
qc="$case_dir/solver_log_qc.dynamic_train8.json"
marker="$case_dir/solver_complete.dynamic_train8.json"
[[ ! -e "$log" && ! -e "$qc" && ! -e "$marker" ]] || { echo "refusing existing output" >&2; exit 1; }
image='opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319'
if ! docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m --cpus 1 --memory 8g \
  --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${root},dst=/case" --workdir /case \
  "$image" pimpleFoam -case "/case/cases/$name" >"$log" 2>&1; then
  echo FAILED >"$lock/status"; exit 1
fi
python3 "$root/check_matched_start_solver_log.py" \
  --log "$log" --expected-steps "$steps" --expected-end "$end" >"$qc"
python3 - "$marker" "$name" "$predecl_sha" "$log" "$qc" <<'PY'
import hashlib, json, os, sys, tempfile
from pathlib import Path
marker, name, pre_sha, log, qc = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
payload = {"status": "DYNAMIC_TRAIN8_SOLVER_COMPLETE_PENDING_AGGREGATE_QC", "case": name,
           "predeclaration_sha256": pre_sha,
           "solver_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
           "solver_qc_sha256": hashlib.sha256(qc.read_bytes()).hexdigest()}
with tempfile.NamedTemporaryFile("w", dir=marker.parent, delete=False) as f:
    tmp=Path(f.name); json.dump(payload,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
try: os.link(tmp, marker)
finally: tmp.unlink(missing_ok=True)
PY
echo COMPLETE >"$lock/status"
