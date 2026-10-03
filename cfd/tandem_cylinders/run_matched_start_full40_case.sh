#!/usr/bin/env bash
# Run one independently authorized full40 remainder case; never the original nine.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd "${root}/../.." && pwd -P)"
name="${1:-}"
phase_manifest="${2:-}"
predeclaration="${3:-}"
authorization="${4:-}"
mode="${5:-run}"
phase_sha='6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603'
predeclaration_sha='d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b'
# Deliberately non-SHA until the post-commissioning authorization is committed.
authorization_sha='REVIEW_REQUIRED_AFTER_NINE_CASE_AGGREGATE'

case_pattern='^matched_start_acquisition_(train|validation|frozen_test)_b0[0-7]_'
case_pattern+='(m075|m0375|zero|p0375|p075)$'
[[ "${name}" =~ ${case_pattern} ]] \
    || { echo "Invalid full40 case name" >&2; exit 2; }
[[ -f "${phase_manifest}" && -f "${predeclaration}" && -f "${authorization}" ]] \
    || { echo "Missing reviewed provenance artifact" >&2; exit 2; }
[[ ${#authorization_sha} -eq 64 ]] \
    || { echo "Extension execution is disabled pending authorization SHA binding" >&2; exit 2; }
[[ "$(sha256sum "${phase_manifest}" | awk '{print $1}')" == "${phase_sha}" ]]
[[ "$(sha256sum "${predeclaration}" | awk '{print $1}')" == "${predeclaration_sha}" ]]
[[ "$(sha256sum "${authorization}" | awk '{print $1}')" == "${authorization_sha}" ]]

target="${root}/cases/${name}"
config="${target}/case_config.json"
[[ -f "${config}" ]] || { echo "Missing generated full40 case config" >&2; exit 1; }

readarray -t bounds < <(python3 - "${config}" "${predeclaration}" "${authorization}" "${name}" \
    "${phase_sha}" "${predeclaration_sha}" "${authorization_sha}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

config_path, predeclaration_path, authorization_path = map(Path, sys.argv[1:4])
requested, phase_sha, predecl_sha, auth_sha = sys.argv[4:]
cfg = json.loads(config_path.read_text(encoding="utf-8"))
predecl = json.loads(predeclaration_path.read_text(encoding="utf-8"))
auth = json.loads(authorization_path.read_text(encoding="utf-8"))
if hashlib.sha256(authorization_path.read_bytes()).hexdigest() != auth_sha:
    raise SystemExit("authorization SHA differs")
if auth.get("status") != "MATCHED_START_FULL40_EXTENSION_AUTHORIZED":
    raise SystemExit("extension is not authorized")
if requested not in auth.get("authorized_cases", []) or len(auth["authorized_cases"]) != 31:
    raise SystemExit("case is not in the exact authorized 31-case remainder")
if cfg.get("case") != requested or cfg.get("panel") != "matched_start_acquisition_full40_v1":
    raise SystemExit("case config identity differs")
if cfg.get("phase_manifest_sha256") != phase_sha:
    raise SystemExit("phase manifest provenance differs")
if cfg.get("full40_predeclaration_sha256") != predecl_sha:
    raise SystemExit("full40 predeclaration provenance differs")
if cfg.get("full40_extension_authorization_sha256") != auth_sha:
    raise SystemExit("extension authorization provenance differs")
expected = predecl.get("cases", {}).get(requested)
if not expected or expected.get("disposition") != "planned_new_remainder_case":
    raise SystemExit("case is absent from the reviewed remainder")
checks = {
    "split": expected["split"], "phase_bin": expected["phase_bin"],
    "source_restart_time": expected["source_restart_time"],
    "source_state_sha256": expected["source_state_sha256"],
    "action_target": expected["action_target"], "action_points": expected["action_points"],
    "start_time": expected["run_window"][0], "end_time": expected["run_window"][1],
    "analysis_window": expected["analysis_window"],
}
for key, value in checks.items():
    if cfg.get(key) != value: raise SystemExit(f"case config differs: {key}")
baseline = auth.get("baseline_force_source_sha256")
if not isinstance(baseline, dict) or set(baseline) != {"forceFront", "forceRear"}:
    raise SystemExit("authorization lacks baseline force provenance")
if cfg.get("source_force_sha256") != baseline:
    raise SystemExit("case baseline force provenance differs")
source_case = config_path.parent.parent / cfg["source_restart_case"]
for force_name, expected_sha in baseline.items():
    force_path = source_case / "postProcessing" / force_name / "0" / "coefficient.dat"
    if hashlib.sha256(force_path.read_bytes()).hexdigest() != expected_sha:
        raise SystemExit(f"baseline force changed: {force_name}")
provenance = config_path.parent / cfg["source_state_provenance_dir"]
for field in ("U", "U_0", "p", "phi", "phi_0"):
    digest = hashlib.sha256((provenance / field).read_bytes()).hexdigest()
    if digest != cfg["source_state_sha256"][field]:
        raise SystemExit(f"source-state provenance differs: {field}")
    staged = config_path.parent / f'{cfg["source_restart_time"]:g}' / field
    if field != "U" and hashlib.sha256(staged.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"staged source state differs: {field}")
velocity = (config_path.parent / f'{cfg["source_restart_time"]:g}' / "U").read_text()
patch = re.search(r"rearCylinder\s*\{(?P<body>.*?)\n\s*\}", velocity, re.S)
table = patch and re.search(r"omega\s+table\s*\((?P<rows>.*?)\)\s*;", patch.group("body"), re.S)
actual = [] if not table else [[float(a), float(b)] for a, b in re.findall(
    r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", table.group("rows"))]
if actual != cfg["action_points"]:
    raise SystemExit("OpenFOAM action table differs from case config")
print(f'{cfg["start_time"]:g}')
print(f'{cfg["end_time"]:g}')
print(cfg["expected_solver_steps"])
PY
)
start_time="${bounds[0]}"
end_time="${bounds[1]}"
expected_steps="${bounds[2]}"
if [[ "${mode}" == "--preflight-only" ]]; then
    echo "FULL40_PREFLIGHT_OK ${name}"
    exit 0
fi
[[ "${mode}" == "run" ]] || { echo "Invalid runner mode" >&2; exit 2; }
lock="${target}/.matched_start_full40_solver_lock"
python3 "${root}/matched_start_case_lock.py" acquire --lock "${lock}" --case "${name}"
finalized=false
on_exit() {
    result=$?
    if [[ "${finalized}" != true ]]; then
        python3 "${root}/matched_start_case_lock.py" update \
            --lock "${lock}" --status FAILED --exit-code "${result}" || true
    fi
    exit "${result}"
}
trap on_exit EXIT

log="${target}/log.pimpleFoam.matched_start_full40"
marker="${target}/solver_complete.full40.json"
[[ ! -e "${log}" && ! -e "${marker}" ]] \
    || { echo "Refusing to overwrite full40 solver output" >&2; exit 1; }
# shellcheck disable=SC2016  # Literal awk program; $1 is an awk field.
numeric_output='$1 ~ /^[0-9.]+$/ && $1 > start {found=1} END {exit !found}'
if find "${target}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk -v start="${start_time}" "${numeric_output}"; then
    echo "Refusing case with existing solver time directories" >&2
    exit 1
fi
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 40 * 1024 * 1024 )) \
    || { echo "Running MemAvailable is below 40 GiB" >&2; exit 1; }

if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"${log}" 2>&1; then
    qc="${target}/solver_log_qc.full40.json"
    [[ ! -e "${qc}" ]] || { echo "Refusing to overwrite solver QC" >&2; exit 1; }
    python3 "${root}/check_matched_start_solver_log.py" \
        --log "${log}" --expected-steps "${expected_steps}" --expected-end "${end_time}" \
        >"${qc}"
    transfer_dir="${repo}/artifacts/matched_start_full40_extension/worker_transfer_manifests"
    mkdir -p "${transfer_dir}"
    transfer_manifest="${transfer_dir}/${name}.sha256"
    [[ ! -e "${transfer_manifest}" ]] \
        || { echo "Refusing to overwrite worker transfer manifest" >&2; exit 1; }
    python3 - "${target}" "${root}/cases" "${transfer_manifest}" "${name}" \
        "${phase_sha}" "${predeclaration_sha}" "${authorization_sha}" \
        "${expected_steps}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

case, cases, manifest = map(Path, sys.argv[1:4])
name, phase_sha, predecl_sha, auth_sha = sys.argv[4:8]
steps = int(sys.argv[8])
def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): result.update(block)
    return result.hexdigest()
marker=case/'solver_complete.full40.json'
if marker.exists(): raise SystemExit('refusing to overwrite solver completion marker')
payload={
    'status':'FULL40_SOLVER_COMPLETED_PENDING_TRANSFER_QC', 'case':name,
    'phase_manifest_sha256':phase_sha, 'full40_predeclaration_sha256':predecl_sha,
    'full40_extension_authorization_sha256':auth_sha, 'solver_steps':steps,
    'solver_log_sha256':digest(case/'log.pimpleFoam.matched_start_full40'),
    'solver_log_qc_sha256':digest(case/'solver_log_qc.full40.json'),
}
tmp=marker.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload,indent=2)+'\n'); os.replace(tmp,marker)
rows=[]
files = (
    item for item in case.rglob('*')
    if item.is_file()
    and not any(part.startswith('.matched_start_') for part in item.parts)
)
for path in sorted(files):
    rows.append(f'{digest(path)}  {path.relative_to(cases)}')
tmp=manifest.with_name(f'.{manifest.name}.tmp')
tmp.write_text('\n'.join(rows)+'\n'); os.replace(tmp,manifest)
PY
    python3 "${root}/matched_start_case_lock.py" update \
        --lock "${lock}" --status COMPLETED --exit-code 0
    finalized=true
else
    result=$?
    tail -n 50 "${log}" >&2
    exit "${result}"
fi
