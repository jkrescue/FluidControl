#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd "$root/../.." && pwd -P)"
name="${1:-}"
auth="$repo/artifacts/tandem_cylinders/directppo_train16_curation_authorization_20261004.json"
auth_sha='321ae9d3ee4297212f91ee60ad774663c51ec16a16b663f5bccdc6e301d66e56'
image='opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319'
[[ "$name" =~ ^direct_cfd_directppo2048_v1_env(0_ep000[2-9]_b00|1_ep000[2-9]_b02)$ ]] || exit 2
[[ "${DIRECTPPO_TRAIN16_VTK_TOKEN:-}" == EXECUTE_REVIEWED_DIRECTPPO_TRAIN16_VTK ]] || exit 2
[[ "$(sha256sum "$auth" | awk '{print $1}')" == "$auth_sha" ]] || { echo authorization-sha-mismatch >&2; exit 2; }
python3 - "$auth" "$name" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="DIRECTPPO_TRAIN16_CURATION_AUTHORIZED_TRAIN_ONLY" and sys.argv[2] in p["cases"]
PY
case="$root/cases/$name"; log="$case/log.foamToVTK.directppo_train16"; vtk="$case/VTK_directppo_train16"
[[ ! -e "$log" && ! -e "$vtk" ]] || { echo refusing-existing-vtk >&2; exit 1; }
(( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) >= 40*1024*1024 )) || exit 1
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=512m \
  --cpus 1 --memory 8g --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --mount "type=bind,src=$root,dst=/case" --workdir /case \
  "$image" foamToVTK -case "/case/cases/$name" -fields '(U p)' -no-boundary \
  -name VTK_directppo_train16 >"$log" 2>&1
