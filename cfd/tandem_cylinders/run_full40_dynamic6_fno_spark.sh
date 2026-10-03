#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
mode="${1:---dry-run}"
data="$root/data/curated/tandem_cylinders_full40_dynamic_validation_v1"
norm="$root/data/curated/tandem_cylinders_matched_start_full40_v1"
ckpt="$root/artifacts/tandem_fno_full40_dev30_quickscreen_h20_qs1/best"
out="$root/artifacts/tandem_cylinders/full40_dynamic6_fno_e5_20261003"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
[[ "$(sha256sum "$data/manifest.json"|awk '{print $1}')" == bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae ]]
[[ "$(sha256sum "$ckpt/FNO.0.5.mdlus"|awk '{print $1}')" == a66779c18e4c6c0724f903dd4e767eee643d0867180ad8c6cab95587353537ae ]]
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
[[ ! -e "$out" ]] || { echo "refusing existing output" >&2; exit 2; }
if systemctl --user is-active --quiet fluid-control-dev30-qs1-validation10-20261003.service; then echo "authoritative validation10 still active" >&2; exit 1; fi
if ps -eo args= | grep -E '[e]valuate_tandem_fno.py' >/dev/null; then echo "another FNO evaluation is active" >&2; exit 1; fi
cmd=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 64g --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)" --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4 --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly" --mount "type=bind,src=$root/src,dst=/workspace/src,readonly" --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly" --mount "type=bind,src=$data,dst=/workspace/dynamic,readonly" --mount "type=bind,src=$norm,dst=/workspace/norm,readonly" --mount "type=bind,src=$ckpt,dst=/workspace/checkpoint,readonly" --mount "type=bind,src=$out,dst=/workspace/output" --workdir /workspace "$image" python -u /workspace/scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- python -u /workspace/scripts/evaluate_tandem_fno.py --data /workspace/dynamic --normalization-data /workspace/norm --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --split validation --horizons 1 10 50 100 --segment-stride 1 --evaluation-batch-size 4 --action-mode observed --visualizations-per-horizon 0 --output /workspace/output/evaluation.json --segment-metrics-output /workspace/output/segments.json)
if [[ "$mode" == --dry-run ]]; then printf 'DYNAMIC6_FNO_VALIDATION_ONLY_READY\n'; printf ' %q' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ "$mode" == --execute && "${DYNAMIC6_FNO_APPROVAL_TOKEN:-}" == EXECUTE_REVIEWED_DYNAMIC6_FNO_E5 ]] || { echo "explicit token required" >&2; exit 2; }
mkdir "$out"; "${cmd[@]}" >"$out/evaluate.log" 2>&1
python3 "$root/cfd/tandem_cylinders/audit_full40_dynamic6_fno.py" --data "$data" --checkpoint "$ckpt" --report "$out/evaluation.json" --segments "$out/segments.json" --physical-qc "$root/artifacts/tandem_cylinders/full40_dynamic_validation_real_cfd_qc_20261003.json" --output "$out/diagnostic.json"
