#!/usr/bin/env bash
# Sequential, resumable FC-P003C post-evaluation. Dry-run is the default.
set -euo pipefail

root="${FCP003C_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"; cd "$root"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--execute|--resume|--wait) ;; *) echo "invalid mode" >&2; exit 2;; esac
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
candidate="$root/artifacts/tandem_fno_true_state_paired_step_lambda10_20261005"
out="$candidate/posteval_fc_p003c"
dev30="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
full40="$root/data/curated/tandem_cylinders_matched_start_full40_v1"
dynamic="$root/data/curated/tandem_cylinders_full40_dynamic_validation_v1"
predecl="$root/artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
physical_qc="$root/artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json"
chain_root="${FCP003C_POSTEVAL_CHAIN_ROOT:-$root/scripts}"
auditor="$chain_root/audit_fcp003c_candidate.py"
validator="$chain_root/validate_fcp003c_posteval.py"
sha() { sha256sum "$1" | awk '{print $1}'; }

if [[ "$mode" != --dry-run && -z "${FCP003C_POSTEVAL_CHAIN_ROOT:-}" ]]; then
  reviewed="${FCP003C_REVIEWED_COMMIT:-}"
  [[ "$reviewed" =~ ^[0-9a-f]{40}$ && "$(git rev-parse "$reviewed^{commit}")" == "$reviewed" ]] || { echo "reviewed posteval commit required" >&2; exit 2; }
  snapshot="$root/artifacts/fcp003c_posteval_queue_${reviewed:0:12}_immutable"
  [[ ! -e "$snapshot" ]] || { echo "immutable queue snapshot already exists; invoke it directly" >&2; exit 2; }
  temporary="$(mktemp -d "$root/artifacts/.fcp003c-posteval-snapshot.XXXXXX")"
  for relative in scripts/run_fcp003c_posteval_spark.sh scripts/audit_fcp003c_candidate.py scripts/validate_fcp003c_posteval.py; do
    mkdir -p "$temporary/$(dirname "$relative")"
    git show "$reviewed:$relative" >"$temporary/$relative"
  done
  chmod +x "$temporary/scripts/"*.py "$temporary/scripts/"*.sh
  python3 - "$temporary/receipt.json" "$temporary" "$reviewed" "$(git rev-parse "$reviewed^{tree}")" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target,root=map(pathlib.Path,sys.argv[1:3]); commit,tree=sys.argv[3:]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob("*")) if path.is_file()}
payload={"status":"FC_P003C_IMMUTABLE_POSTEVAL_CHAIN_STAGED","git_commit":commit,"git_tree":tree,"sha256":files}
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
  mv "$temporary" "$snapshot"
  chmod -R a-w "$snapshot"
  exec env FCP003C_REPO_ROOT="$root" FCP003C_POSTEVAL_CHAIN_ROOT="$snapshot/scripts" \
    FCP003C_REVIEWED_COMMIT="$reviewed" "$snapshot/scripts/run_fcp003c_posteval_spark.sh" "$mode"
fi
if [[ -n "${FCP003C_POSTEVAL_CHAIN_ROOT:-}" ]]; then
  python3 "$auditor" --repo "$root" --chain-receipt "$chain_root/../receipt.json" \
    --chain-root "$chain_root" --reviewed-commit "${FCP003C_REVIEWED_COMMIT:-}" >/dev/null
fi

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "pinned image differs" >&2; exit 2; }
for path in "$dev30" "$full40" "$dynamic"; do [[ -d "$path" ]] || { echo "missing evaluation data: $path" >&2; exit 2; }; done
[[ "$(sha "$dev30/manifest.json")" == 5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2 ]]
[[ "$(sha "$full40/manifest.json")" == 1c9086b4d08da57e84fb4f6a22376f0935596727952a50c55acb1a130e77e90e ]]
[[ "$(sha "$dynamic/manifest.json")" == bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae ]]
[[ "$(sha "$predecl")" == d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b ]]
[[ "$(sha "$physical_qc")" == 9723203f922cbe6609f2c92b7b48d299ee9d948d694d3c6c2eab413472421d86 ]]

if [[ "$mode" == --dry-run ]]; then
  python3 "$auditor" --repo "$root" >/dev/null
  echo "FC_P003C_POSTEVAL_DRY_RUN_PASS_NO_GPU"
  echo "candidate=$candidate"
  echo "protocol=validation10_stride25_batch4,dynamic6_stride1_batch8,force_window6,unchanged_development_gate"
  echo "validation_mount=/workspace/devdata_from_dev30 endpoint_mount=/workspace/devdata_from_full40_in_separate_container"
  exit 0
fi
[[ "${FCP003C_POSTEVAL_TOKEN:-}" == EXECUTE_APPROVED_FC_P003C_POSTEVAL ]] || { echo "posteval token required" >&2; exit 2; }
if [[ "$mode" == --wait ]]; then
  while [[ ! -f "$candidate/completion_receipt.json" ]]; do sleep 30; done
  mode=--execute
fi
[[ -f "$candidate/completion_receipt.json" ]] || { echo "formal training is incomplete" >&2; exit 3; }
lineage_json="$(python3 "$auditor" --repo "$root" --candidate "$candidate")"
checkpoint_sha="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["checkpoint_sha256"])' <<<"$lineage_json")"
if [[ -f "$out/receipt.json" ]]; then
  python3 "$validator" --repo "$root" --candidate "$candidate" --output "$out" --checkpoint-sha256 "$checkpoint_sha" --step complete
  exit 0
fi
if [[ "$mode" == --execute && -e "$out" ]]; then echo "existing partial output requires --resume" >&2; exit 2; fi
mkdir -p "$out/validation10" "$out/dynamic6" "$out/force_window" "$out/step_receipts"
if [[ ! -f "$out/lineage.json" ]]; then
  python3 "$auditor" --repo "$root" --candidate "$candidate" --output "$out/lineage.json" >/dev/null
else
  python3 - "$auditor" "$root" "$candidate" "$out/lineage.json" <<'PY'
import importlib.util,json,pathlib,sys
path,repo,candidate,stored=map(pathlib.Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location("lineage",path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
if json.loads(stored.read_text()) != mod.validate_candidate(repo,candidate): raise SystemExit("stored lineage differs")
PY
fi
source="$candidate/source_snapshot"; checkpoint="$candidate/best"
common=(--rm --network none --cpus 8 --memory 64g --shm-size 2g --pids-limit 512
 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g
 --user "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1
 -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e PYTHONPATH=/workspace/src:/workspace/scripts
 -v "$source/scripts:/workspace/scripts:ro" -v "$source/src:/workspace/src:ro" -v "$source/conf:/workspace/conf:ro"
 -v "$source/cfd:/workspace/cfd:ro" -v "$checkpoint:/workspace/checkpoint:ro" -v "$out:/workspace/output:rw" -w /workspace)
validate_step() { python3 "$validator" --repo "$root" --candidate "$candidate" --output "$out" --checkpoint-sha256 "$checkpoint_sha" --step "$1"; }
step_receipt() {
  local step="$1"; shift
  python3 - "$out/step_receipts/$step.json" "$step" "$checkpoint_sha" "$@" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target=pathlib.Path(sys.argv[1]); step,checkpoint=sys.argv[2:4]; paths=list(map(pathlib.Path,sys.argv[4:]))
def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
if any(not path.is_file() for path in paths): raise SystemExit("step artifacts incomplete")
payload={"status":"FC_P003C_POSTEVAL_STEP_COMPLETE","step":step,"checkpoint_sha256":checkpoint,"sha256":{str(path):sha(path) for path in paths}}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
}

if [[ ! -f "$out/validation10/evaluation.json" || ! -f "$out/validation10/segments.json" ]]; then
  [[ ! -f "$out/validation10/evaluation.json" && ! -f "$out/validation10/segments.json" ]] || { echo "partial validation pair" >&2; exit 3; }
  docker run --gpus device=0 "${common[@]}" -v "$dev30:/workspace/devdata:ro" "$image" \
   python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
   python -u scripts/evaluate_tandem_fno.py --data /workspace/devdata --normalization-data /workspace/devdata \
   --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --split validation \
   --horizons 1 10 50 100 --segment-stride 25 --evaluation-batch-size 4 --action-mode observed \
   --visualizations-per-horizon 0 --output /workspace/output/validation10/evaluation.json \
   --segment-metrics-output /workspace/output/validation10/segments.json 2>&1 | tee "$out/validation10/evaluate.log"
fi
if [[ ! -f "$out/validation10/diagnostic.json" ]]; then
  python3 "$source/scripts/audit_dev30_validation_diagnostic.py" --report "$out/validation10/evaluation.json" \
   --segments "$out/validation10/segments.json" --data "$dev30" --checkpoint-dir "$checkpoint" \
   --candidate-kind dev30_free_ar_development --output "$out/validation10/diagnostic.json"
fi
if [[ ! -f "$out/validation10/endpoint_gate.json" ]]; then
  docker run "${common[@]}" -v "$full40/validation:/workspace/devdata/validation:ro" \
   -v "$full40/manifest.json:/workspace/devdata/manifest.json:ro" -v "$full40/normalization.json:/workspace/devdata/normalization.json:ro" \
   -v "$predecl:/workspace/predecl.json:ro" "$image" python -u scripts/audit_full40_validation_gate.py \
   --report /workspace/output/validation10/evaluation.json --segments /workspace/output/validation10/segments.json \
   --predeclaration /workspace/predecl.json --checkpoint-dir /workspace/checkpoint --data /workspace/devdata \
   --config /workspace/conf/tandem_fno_full40_h20.yaml --image-id "$image_id" --output /workspace/output/validation10/endpoint_gate.json
fi
validate_step validation10
[[ -f "$out/step_receipts/validation10.json" ]] || step_receipt validation10 "$out/validation10/evaluation.json" "$out/validation10/segments.json" "$out/validation10/diagnostic.json" "$out/validation10/endpoint_gate.json"

if [[ ! -f "$out/dynamic6/evaluation.json" || ! -f "$out/dynamic6/segments.json" ]]; then
  [[ ! -f "$out/dynamic6/evaluation.json" && ! -f "$out/dynamic6/segments.json" ]] || { echo "partial dynamic pair" >&2; exit 3; }
  docker run --gpus device=0 "${common[@]}" -v "$dynamic:/workspace/dynamic:ro" -v "$dev30:/workspace/devdata:ro" "$image" \
   python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
   python -u scripts/evaluate_tandem_fno.py --data /workspace/dynamic --normalization-data /workspace/devdata \
   --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --split validation \
   --horizons 1 10 50 100 --segment-stride 1 --evaluation-batch-size 8 --action-mode observed \
   --visualizations-per-horizon 0 --output /workspace/output/dynamic6/evaluation.json \
   --segment-metrics-output /workspace/output/dynamic6/segments.json 2>&1 | tee "$out/dynamic6/evaluate.log"
fi
if [[ ! -f "$out/dynamic6/diagnostic.json" ]]; then
  python3 "$source/cfd/tandem_cylinders/audit_full40_dynamic6_fno.py" --data "$dynamic" --checkpoint "$checkpoint" \
   --checkpoint-epoch "$(basename "$checkpoint"/FNO.0.*.mdlus | cut -d. -f3)" --expected-model-sha "$checkpoint_sha" \
   --report "$out/dynamic6/evaluation.json" --segments "$out/dynamic6/segments.json" --physical-qc "$physical_qc" \
   --output "$out/dynamic6/diagnostic.json"
fi
validate_step dynamic6
[[ -f "$out/step_receipts/dynamic6.json" ]] || step_receipt dynamic6 "$out/dynamic6/evaluation.json" "$out/dynamic6/segments.json" "$out/dynamic6/diagnostic.json"

if [[ ! -f "$out/force_window/result.json" ]]; then
  docker run --gpus device=0 "${common[@]}" -v "$dynamic:/workspace/dynamic:ro" -v "$dev30:/workspace/devdata:ro" "$image" \
   python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
   python -u scripts/diagnose_fno_force_window.py --data /workspace/dynamic --normalization-data /workspace/devdata \
   --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --expected-model-sha "$checkpoint_sha" \
   --output /workspace/output/force_window/result.json 2>&1 | tee "$out/force_window/diagnose.log"
fi
if [[ ! -f "$out/development_gate.json" ]]; then
  python3 "$source/scripts/audit_dynamic_fno_development_gates.py" --force-window "$out/force_window/result.json" \
   --checkpoint-sha256 "$checkpoint_sha" --output "$out/development_gate.json"
fi
validate_step force_window
[[ -f "$out/step_receipts/force_window.json" ]] || step_receipt force_window "$out/force_window/result.json" "$out/development_gate.json"

python3 - "$out" "$checkpoint_sha" "$image_id" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]); checkpoint,image=sys.argv[2:]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"receipt.json","outer.log"}}
payload={"status":"FC_P003C_POSTEVAL_COMPLETE","candidate_kind":"true_state_paired_step_lambda10","checkpoint_sha256":checkpoint,"official_image_id":image,
 "protocol":["validation10_H1_H10_H50_H100_stride25_batch4","dynamic6_H1_H10_H50_H100_stride1_batch8","force_window6","unchanged_development_gate"],
 "sha256":files,"frozen_test_accessed":False,"ppo_auto_launched":False}
target=root/"receipt.json"
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
validate_step complete
echo FC_P003C_POSTEVAL_COMPLETE
