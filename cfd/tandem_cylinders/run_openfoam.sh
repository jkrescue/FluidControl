#!/usr/bin/env bash
set -euo pipefail

# Run only this case directory in the pinned OpenFOAM image.
# Examples: bash run_openfoam.sh blockMesh -help
#           bash run_openfoam.sh pimpleFoam -help
case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
image_ref='opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319'

if (( $# == 0 )); then
    echo 'Usage: bash run_openfoam.sh <OpenFOAM command> [arguments...]' >&2
    exit 2
fi

exec docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=512m \
    --cpus 4 \
    --memory 8g \
    --pids-limit 128 \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=${case_dir},dst=/case" \
    --workdir /case \
    "$image_ref" "$@"
