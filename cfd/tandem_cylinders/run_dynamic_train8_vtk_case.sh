#!/usr/bin/env bash
# Export one authorized train8 raw case to VTK with one CPU; refuse overwrite.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd "$root/../.." && pwd -P)"
name="${1:-}"
token="EXECUTE_REVIEWED_DYNAMIC_TRAIN8_VTK"
auth="$repo/artifacts/tandem_cylinders/dynamic_train8_curation_authorization_20261003.json"
auth_sha="acbd967f66ff6f938cac8457deabba3ce52ba2a235ab5596fc5e0b3cfd03bb0e"
image='opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319'

[[ "$name" =~ ^dynamic_train8_b(00|02|04|06)_(prbs|multisine)$ ]] \
  || { echo "case outside train8" >&2; exit 2; }
[[ "${DYNAMIC_TRAIN8_VTK_APPROVAL_TOKEN:-}" == "$token" ]] \
  || { echo "explicit VTK token required" >&2; exit 2; }
[[ "$(sha256sum "$auth" | awk '{print $1}')" == "$auth_sha" ]] \
  || { echo "curation authorization SHA differs" >&2; exit 2; }
python3 - "$auth" "$name" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
if p.get("status") != "DYNAMIC_TRAIN8_CURATION_AUTHORIZED_TRAIN_ONLY" or sys.argv[2] not in p.get("cases", {}):
    raise SystemExit("case absent from train-only curation authorization")
PY
case_dir="$root/cases/$name"
log="$case_dir/log.foamToVTK.dynamic_train8"
vtk="$case_dir/VTK_dynamic_train8"
[[ ! -e "$log" && ! -e "$vtk" ]] || { echo "refusing existing VTK/log" >&2; exit 1; }
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 40 * 1024 * 1024 )) || { echo "MemAvailable below 40 GiB" >&2; exit 1; }

docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m --cpus 1 --memory 8g \
  --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${root},dst=/case" --workdir /case \
  "$image" foamToVTK -case "/case/cases/$name" -fields '(U p)' \
  -no-boundary -name VTK_dynamic_train8 >"$log" 2>&1
