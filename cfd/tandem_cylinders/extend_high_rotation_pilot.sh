#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
name="${1:-}"
case "$name" in
    high_rotation_qm250_dt0025|high_rotation_qp250_dt0025|high_rotation_qm250_medium_dt0025|high_rotation_qp250_medium_dt0025) ;;
    *) echo 'Usage: extend_high_rotation_pilot.sh <q=+/-2.5 coarse/medium dt=0.0025 case>' >&2; exit 2 ;;
esac

target="${root}/cases/${name}"
control="${target}/system/controlDict"
log="${target}/log.pimpleFoam.extend_to_160"
if [[ ! -d "${target}/100" || ! -f "$control" ]]; then
    echo "Completed t=100 pilot is missing: ${target}" >&2
    exit 1
fi
if [[ -e "$log" ]]; then
    echo "Refusing to overwrite extension log: ${log}" >&2
    exit 1
fi

python3 - "$control" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for old, new in (("startFrom startTime;", "startFrom latestTime;"),
                 ("endTime 100;", "endTime 160;")):
    if text.count(old) != 1:
        raise SystemExit(f"expected one {old!r} in {path}, found {text.count(old)}")
    text = text.replace(old, new)
# Keep force/probe output every solver step; reduce only the first, field-output interval.
old = "writeInterval 1;"
position = text.find(old)
if position < 0:
    raise SystemExit(f"field write interval not found in {path}")
text = text[:position] + "writeInterval 5;" + text[position + len(old):]
path.write_text(text, encoding="utf-8")
PY

echo "Extending ${name} from t=100 to t=160; solver log: ${log}"
if bash "${root}/run_openfoam.sh" pimpleFoam -case "/case/cases/${name}" >"$log" 2>&1; then
    echo "Completed extension ${name}"
    tail -n 4 "$log"
else
    result=$?
    echo "Failed extension ${name} (exit ${result}); final log lines:" >&2
    tail -n 50 "$log" >&2
    exit "$result"
fi
