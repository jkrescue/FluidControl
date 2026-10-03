#!/usr/bin/env bash
# Run one generated matched-start commissioning case with fail-closed provenance.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd "${root}/../.." && pwd -P)"
name="${1:-}"
phase_manifest="${2:-}"
approved_sha256='6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603'
approved_manifest_name='matched_start_phase_restart_predeclared_v3_20261003.json'

[[ "${name}" =~ ^matched_start_acquisition_train_b(00|02|04)_(m075|zero|p075)$ ]] \
    || { echo "Invalid commissioning case name" >&2; exit 2; }
[[ -f "${phase_manifest}" ]] || { echo "Missing phase manifest" >&2; exit 2; }
[[ "$(basename "${phase_manifest}")" == "${approved_manifest_name}" ]] \
    || { echo "Only the reviewed v3 phase manifest is accepted" >&2; exit 2; }

target="${root}/cases/${name}"
config="${target}/case_config.json"
log="${target}/log.pimpleFoam.matched_start_acquisition"
[[ -f "${config}" ]] || { echo "Missing generated case config" >&2; exit 1; }
lock="${target}/.matched_start_solver_lock"
python3 "${root}/matched_start_case_lock.py" acquire --lock "${lock}" --case "${name}"
run_finalized=false
finalize_failed() {
    result=$?
    if [[ "${run_finalized}" != true ]]; then
        python3 "${root}/matched_start_case_lock.py" update \
            --lock "${lock}" --status FAILED --exit-code "${result}" || true
    fi
    exit "${result}"
}
trap finalize_failed EXIT
[[ ! -e "${log}" ]] || { echo "Refusing to overwrite solver log" >&2; exit 1; }
[[ ! -e "${target}/solver_complete.json" ]] \
    || { echo "Refusing to overwrite solver completion marker" >&2; exit 1; }

readarray -t bounds < <(python3 - "${config}" "${phase_manifest}" "${approved_sha256}" "${name}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
approved = sys.argv[3]
requested = sys.argv[4]
actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if actual != approved:
    raise SystemExit("phase manifest SHA-256 is not approved")
cfg = json.loads(config_path.read_text(encoding="utf-8"))
if cfg["case"] != requested or cfg["panel"] != "matched_start_acquisition_commissioning_v1":
    raise SystemExit("case config identity mismatch")
if cfg["phase_manifest_sha256"] != approved:
    raise SystemExit("case was generated from another phase manifest")
if cfg["split"] != "train" or cfg["phase_bin"] not in (0, 2, 4):
    raise SystemExit("case is not in the commissioning train subset")
if cfg["action_target"] not in (-0.75, 0.0, 0.75):
    raise SystemExit("case action differs from commissioning contract")
provenance = config_path.parent / cfg["source_state_provenance_dir"]
for field in ("U", "U_0", "p", "phi", "phi_0"):
    path = provenance / field
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != cfg["source_state_sha256"][field]:
        raise SystemExit(f"source restart provenance hash mismatch: {field}")
    staged = config_path.parent / f'{cfg["source_restart_time"]:g}' / field
    if field != "U" and hashlib.sha256(staged.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"staged restart field differs from source: {field}")
print(f'{cfg["start_time"]:g}')
print(f'{cfg["end_time"]:g}')
print(cfg["expected_solver_steps"])
PY
)
start_time="${bounds[0]}"
end_time="${bounds[1]}"
expected_steps="${bounds[2]}"
[[ -f "${target}/${start_time}/U" && -f "${target}/${start_time}/p" ]] \
    || { echo "Missing restart fields" >&2; exit 1; }
if find "${target}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk -v start="${start_time}" '$1 ~ /^[0-9.]+$/ && $1 > start {found=1} END {exit !found}'; then
    echo "Refusing case with existing solver output" >&2
    exit 1
fi

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 40 * 1024 * 1024 )) \
    || { echo "MemAvailable below 40 GiB guard" >&2; exit 1; }

echo "Starting ${name}: ${start_time} -> ${end_time}; manifest=${approved_sha256}"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"${log}" 2>&1; then
    solver_qc="${target}/solver_log_qc.json"
    [[ ! -e "${solver_qc}" ]] || { echo "Refusing to overwrite solver QC" >&2; exit 1; }
    python3 "${root}/check_matched_start_solver_log.py" \
        --log "${log}" --expected-steps "${expected_steps}" --expected-end "${end_time}" \
        >"${solver_qc}"
    transfer_dir="${repo}/artifacts/matched_start_acquisition/worker_transfer_manifests"
    mkdir -p "${transfer_dir}"
    transfer_manifest="${transfer_dir}/${name}.sha256"
    [[ ! -e "${transfer_manifest}" ]] \
        || { echo "Refusing to overwrite worker transfer manifest" >&2; exit 1; }
    python3 - "${target}" "${root}/cases" "${transfer_manifest}" \
        "${approved_sha256}" "${name}" "${expected_steps}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

case = Path(sys.argv[1])
cases = Path(sys.argv[2])
manifest = Path(sys.argv[3])
approved = sys.argv[4]
name = sys.argv[5]
steps = int(sys.argv[6])

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()

marker = case / "solver_complete.json"
if marker.exists():
    raise SystemExit("refusing to overwrite solver completion marker")
marker_tmp = marker.with_suffix(".json.tmp")
payload = {
    "status": "SOLVER_COMPLETED_PENDING_TRANSFER_QC",
    "case": name,
    "phase_manifest_sha256": approved,
    "solver_steps": steps,
    "solver_log_sha256": digest(case / "log.pimpleFoam.matched_start_acquisition"),
    "solver_log_qc_sha256": digest(case / "solver_log_qc.json"),
}
marker_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(marker_tmp, marker)

rows = []
for path in sorted(
    item
    for item in case.rglob("*")
    if item.is_file() and ".matched_start_solver_lock" not in item.parts
):
    rows.append(f"{digest(path)}  {path.relative_to(cases)}")
temporary = manifest.with_name(f".{manifest.name}.tmp")
temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
PY
    python3 "${root}/matched_start_case_lock.py" update \
        --lock "${lock}" --status COMPLETED --exit-code 0
    run_finalized=true
    echo "Completed ${name}"
else
    result=$?
    echo "Failed ${name} (exit ${result}); raw failure retained" >&2
    tail -n 50 "${log}" >&2
    exit "${result}"
fi
