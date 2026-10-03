#!/usr/bin/env bash
# Fail-stop serial execution of the six predeclared dynamic validation cases.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
runner="$root/cfd/tandem_cylinders/run_full40_dynamic_validation_case.sh"
mode="${1:---preflight-only}"
cases=(
  full40_dynamic_validation_b01_zero
  full40_dynamic_validation_b01_minus
  full40_dynamic_validation_b01_plus
  full40_dynamic_validation_b05_zero
  full40_dynamic_validation_b05_minus
  full40_dynamic_validation_b05_plus
)

[[ "$mode" == "--preflight-only" || "$mode" == "--execute" ]] \
  || { echo "mode must be --preflight-only or --execute" >&2; exit 2; }

lock="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/fluid-control-dynamic6-serial.lock"
exec 9>"$lock"
flock -n 9 || { echo "another dynamic6 serial runner holds the lock" >&2; exit 1; }

for case in "${cases[@]}"; do
  bash "$runner" "$case" --preflight-only
done
if [[ "$mode" == "--preflight-only" ]]; then
  echo "FULL40_DYNAMIC6_SERIAL_PREFLIGHT_OK cases=${#cases[@]}"
  exit 0
fi
[[ "${FULL40_DYNAMIC6_SERIAL_APPROVAL_TOKEN:-}" == \
  "EXECUTE_REVIEWED_FULL40_DYNAMIC6_SERIAL" ]] \
  || { echo "explicit serial execution token is required" >&2; exit 2; }

for case in "${cases[@]}"; do
  printf 'DYNAMIC6_CASE_START case=%s utc=%(%FT%TZ)T\n' "$case" -1
  FULL40_DYNAMIC_VALIDATION_APPROVAL_TOKEN=EXECUTE_REVIEWED_FULL40_DYNAMIC_VALIDATION \
    bash "$runner" "$case" --execute
  printf 'DYNAMIC6_CASE_COMPLETE case=%s utc=%(%FT%TZ)T\n' "$case" -1
done
echo "FULL40_DYNAMIC6_SERIAL_SOLVER_COMPLETE_PENDING_PANEL_QC"
