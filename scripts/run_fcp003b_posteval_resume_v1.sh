#!/usr/bin/env bash
# Immutable recovery after completed validation10 inference and legacy path-audit failure.
set -euo pipefail

mode="${1:---dry-run}"
case "$mode" in --dry-run|--resume) ;; *) echo "mode must be --dry-run or --resume" >&2; exit 2;; esac
root="${FCP003B_ROOT:-/home/USER/workspace/fluid_control_fcp003b_dynamic_pairs_20261005}"
eval_root="${FCP003B_EVAL_ROOT:-/home/USER/workspace/fluid_control_paired_lambda10_posteval_20261004}"
candidate="$root/output"
source="$root/source_snapshot"
out="$candidate/posteval_fc_p003b"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
dev30="$root/data/dev30"
dynamic="$eval_root/data/curated/tandem_cylinders_full40_dynamic_validation_v1"
full40="$eval_root/data/curated/tandem_cylinders_matched_start_full40_v1"
predecl="$eval_root/artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
physical_qc="$eval_root/artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json"
lineage="$root/launch/audit_fcp003b_candidate_lineage.py"
lineage_sha="0730690dfce369d3c9b6a84725aa7ed1720d3692a1094ccd2521f3dd515ef99d"
validator_sha="11885f2a54e9e82cf75c82855137b13005b43adf74a554b26e2e699c7aa3f468"

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]]
for path in "$source" "$dev30" "$dynamic" "$full40/validation"; do [[ -e "$path" ]]; done
for path in "$predecl" "$physical_qc" "$lineage"; do [[ -f "$path" ]]; done
[[ "$(sha256sum "$lineage" | cut -d' ' -f1)" == "$lineage_sha" ]] || { echo "immutable lineage differs" >&2; exit 2; }
[[ "$(sha256sum "$root/launch/validate_fcp003b_worker_output.py" | cut -d' ' -f1)" == "$validator_sha" ]] || { echo "immutable validator differs" >&2; exit 2; }
[[ "$(sha256sum "$dev30/normalization.json" | cut -d' ' -f1)" == f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1 ]]
[[ "$(sha256sum "$dynamic/manifest.json" | cut -d' ' -f1)" == bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae ]]
sha() { sha256sum "$1" | cut -d' ' -f1; }
[[ "$(sha "$full40/manifest.json")" == 1c9086b4d08da57e84fb4f6a22376f0935596727952a50c55acb1a130e77e90e ]]
[[ "$(sha "$full40/normalization.json")" == f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1 ]]
[[ "$(sha "$predecl")" == d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b ]]
[[ "$(sha "$physical_qc")" == 9723203f922cbe6609f2c92b7b48d299ee9d948d694d3c6c2eab413472421d86 ]]
[[ "$(sha "$eval_root/cfd/tandem_cylinders/audit_full40_dynamic6_fno.py")" == 4e78d8473d1d0f93b25031a3bf9dcc0582f43604f7b1f67c65e6754f032af100 ]]
[[ "$(sha "$out/lineage.json")" == 9ef9165f8caba45ef310c19e6e298eb3494de249992d7498f53d4b6af78518ad ]]
[[ "$(sha "$out/validation10/evaluation.json")" == ba00fbdb85aa96b4d45e800203ae109aadb50e3740e8b9ea8a286594b66bb0d7 ]]
[[ "$(sha "$out/validation10/segments.json")" == 8040289089cb653692d58940e0bcb1eddc305408c681ca9ae76e33f8487233dd ]]
[[ "$(sha "$out/validation10/diagnostic.json")" == b970de206e706c3027ea367cf5bce713e3c290a5dbf5683c841219e50c0bd6b2 ]]
if [[ "$mode" == --dry-run ]]; then
  echo "FC_P003B_POSTEVAL_RESUME_READY_REUSE_VALIDATION10"
  echo "completed_gpu_stage=validation10 remaining=endpoint-audit,dynamic6,force-window,development-gate frozen=false ppo=false"
  exit 0
fi
[[ "${FCP003B_POSTEVAL_RESUME_TOKEN:-}" == EXECUTE_REVIEWED_FC_P003B_POSTEVAL_RESUME ]]
[[ -d "$out" && ! -e "$out/receipt.json" ]] || { echo "resume output state differs" >&2; exit 2; }

[[ -f "$candidate/completion_receipt.json" ]] || { echo "formal completion timeout" >&2; exit 3; }
recovery="$out/recovery_v2"
mkdir "$recovery"
mkdir -p "$out/dynamic6" "$out/force_window"
cp -a "$0" "$recovery/immutable_resume_runner.sh"
cmp -s "$0" "$recovery/immutable_resume_runner.sh" || { echo "resume runner copy differs" >&2; exit 3; }
python3 "$lineage" --candidate "$candidate" \
  --validator "$root/launch/validate_fcp003b_worker_output.py" --source "$source" \
  --source-receipt "$root/source_receipt.json" --source-required "$root/evidence/source_required_hashes.json" \
  --approval "$root/evidence/FC-P003B_APPROVAL.md" --dynamic-manifest "$root/evidence/dynamic_pair_manifest.json" \
  --real-sampling "$root/evidence/real_sampling_contract_v2.json" \
  --baseline-order "$root/evidence/fc_p003_dataloader_order_result.json" \
  --parent "$root/immutable_parent" --output "$recovery/lineage.recomputed.json"
cmp -s "$out/lineage.json" "$recovery/lineage.recomputed.json" || { echo "stored lineage differs from recomputation" >&2; exit 3; }
checkpoint_sha="$(jq -er .checkpoint_sha256 "$out/lineage.json")"

common=(--rm --network none --cpus 8 --memory 64g --shm-size 2g --pids-limit 512
  --cap-drop ALL --security-opt no-new-privileges --read-only
  --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)"
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1
  -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e PYTHONPATH=/workspace/src:/workspace/scripts
  -v "$source/scripts:/workspace/scripts:ro" -v "$source/src:/workspace/src:ro"
  -v "$source/conf:/workspace/conf:ro" -v "$candidate/best:/workspace/checkpoint:ro"
  -v "$out:/workspace/output:rw" -w /workspace)
write_step_receipt() {
  local step="$1"; shift
  python3 - "$recovery/${step}.json" "$step" "$checkpoint_sha" "$@" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target=pathlib.Path(sys.argv[1]); step,checkpoint=sys.argv[2:4]; paths=list(map(pathlib.Path,sys.argv[4:]))
if not paths or any(not p.is_file() for p in paths): raise SystemExit(f'{step} artifacts incomplete')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
value={'status':'FC_P003B_POSTEVAL_RECOVERY_STEP_COMPLETE','step':step,
 'checkpoint_sha256':checkpoint,'sha256':{str(p):sha(p) for p in paths},
 'inference_reused':step=='validation10','frozen_test_accessed':False}
with tempfile.NamedTemporaryFile('w',dir=target.parent,delete=False) as f:
 tmp=pathlib.Path(f.name); json.dump(value,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
}

python3 - "$out" "$checkpoint_sha" "$dev30" <<'PY'
import hashlib,json,math,pathlib,sys
out,checkpoint,data=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3])
paths=[out/'validation10/evaluation.json',out/'validation10/segments.json',out/'validation10/diagnostic.json']
if any(not p.is_file() for p in paths) or (out/'validation10/endpoint_gate.json').exists():
 raise SystemExit('completed validation10 reuse set differs')
def finite(value):
 if isinstance(value,dict): return all(finite(v) for v in value.values())
 if isinstance(value,list): return all(finite(v) for v in value)
 return not isinstance(value,float) or math.isfinite(value)
report=json.loads(paths[0].read_text()); segments=json.loads(paths[1].read_text())
if not finite(report) or not finite(segments): raise SystemExit('validation10 contains non-finite values')
if report.get('evaluation_data')!='/workspace/devdata' or report.get('normalization_data')!='/workspace/devdata':
 raise SystemExit('validation10 runtime data paths differ')
if report.get('checkpoint_dir')!='/workspace/checkpoint' or report.get('checkpoint_epoch')!=2:
 raise SystemExit('validation10 checkpoint runtime differs')
models=list((out.parent/'best').glob('FNO.0.2.mdlus'))
if len(models)!=1 or hashlib.sha256(models[0].read_bytes()).hexdigest()!=checkpoint:
 raise SystemExit('validation10 checkpoint differs')
manifest=json.loads((data/'manifest.json').read_text())
if manifest.get('declared_trajectory_counts',{}).get('validation')!=10 or manifest.get('materialized_trajectory_counts',{}).get('validation')!=10:
 raise SystemExit('validation count differs')
if 'frozen_test' in manifest.get('materialized_trajectory_counts',{}):
 raise SystemExit('frozen test unexpectedly materialized in dev30')
print(json.dumps({'status':'FC_P003B_VALIDATION10_GPU_REUSE_VERIFIED',
 'sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},sort_keys=True))
PY
docker run "${common[@]}" \
  -v "$full40/validation:/workspace/devdata/validation:ro" -v "$full40/manifest.json:/workspace/devdata/manifest.json:ro" \
  -v "$full40/normalization.json:/workspace/devdata/normalization.json:ro" -v "$predecl:/workspace/predecl.json:ro" "$image" \
  python -u scripts/audit_full40_validation_gate.py --report /workspace/output/validation10/evaluation.json \
  --segments /workspace/output/validation10/segments.json --predeclaration /workspace/predecl.json \
  --checkpoint-dir /workspace/checkpoint --data /workspace/devdata \
  --config /workspace/conf/tandem_fno_full40_h20.yaml --image-id "$image_id" \
  --output /workspace/output/validation10/endpoint_gate.json
write_step_receipt validation10 "$out/validation10/evaluation.json" "$out/validation10/segments.json" \
  "$out/validation10/diagnostic.json" "$out/validation10/endpoint_gate.json"

docker run --gpus device=0 "${common[@]}" -v "$dynamic:/workspace/dynamic:ro" -v "$dev30:/workspace/devdata:ro" "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
  python -u scripts/evaluate_tandem_fno.py --data /workspace/dynamic --normalization-data /workspace/devdata \
  --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --split validation \
  --horizons 1 10 50 100 --segment-stride 1 --evaluation-batch-size 8 --action-mode observed \
  --visualizations-per-horizon 0 --output /workspace/output/dynamic6/evaluation.json \
  --segment-metrics-output /workspace/output/dynamic6/segments.json \
  2>&1 | tee "$out/dynamic6/evaluate.log"
docker run "${common[@]}" -v "$eval_root/cfd:/workspace/cfd:ro" -v "$dynamic:/workspace/dynamic:ro" \
  -v "$physical_qc:/workspace/physical_qc.json:ro" "$image" \
  python -u cfd/tandem_cylinders/audit_full40_dynamic6_fno.py --data /workspace/dynamic \
  --checkpoint /workspace/checkpoint --checkpoint-epoch "$(jq -er .checkpoint_epoch "$out/lineage.json")" \
  --expected-model-sha "$checkpoint_sha" --report /workspace/output/dynamic6/evaluation.json \
  --segments /workspace/output/dynamic6/segments.json --physical-qc /workspace/physical_qc.json \
  --output /workspace/output/dynamic6/diagnostic.json
write_step_receipt dynamic6 "$out/dynamic6/evaluation.json" "$out/dynamic6/segments.json" \
  "$out/dynamic6/diagnostic.json"

docker run --gpus device=0 "${common[@]}" -v "$dynamic:/workspace/dynamic:ro" -v "$dev30:/workspace/devdata:ro" "$image" \
  python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
  python -u scripts/diagnose_fno_force_window.py --data /workspace/dynamic --normalization-data /workspace/devdata \
  --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint \
  --expected-model-sha "$checkpoint_sha" --output /workspace/output/force_window/result.json \
  2>&1 | tee "$out/force_window/diagnose.log"
docker run "${common[@]}" "$image" python -u scripts/audit_dynamic_fno_development_gates.py \
  --force-window /workspace/output/force_window/result.json --checkpoint-sha256 "$checkpoint_sha" \
  --output /workspace/output/development_gate.json
write_step_receipt force_window "$out/force_window/result.json" "$out/development_gate.json"

python3 - "$out" "$checkpoint_sha" "$image_id" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]); checkpoint=sys.argv[2]; image=sys.argv[3]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
 return h.hexdigest()
files={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob('*'))
       if path.is_file() and path.name != 'receipt.json' and not path.name.endswith('.outer.log')}
required={'lineage.json','validation10/evaluation.json','validation10/segments.json',
 'validation10/diagnostic.json','validation10/endpoint_gate.json','dynamic6/evaluation.json',
 'dynamic6/segments.json','dynamic6/diagnostic.json','force_window/result.json','development_gate.json'}
if not required.issubset(files): raise SystemExit('posteval artifacts incomplete')
lineage=json.loads((root/'lineage.json').read_text())
payload={'status':'FC_P003B_POSTEVAL_COMPLETE','candidate_kind':'dynamic_paired_interleaved_lambda10',
 'checkpoint_sha256':checkpoint,'checkpoint_epoch':lineage['checkpoint_epoch'],
 'official_image_id':image,'normalization_sha256':'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1',
 'protocol':['validation10_H1_H10_H50_H100_stride25_batch4','dynamic6_H1_H10_H50_H100_stride1_batch8','force_window6','unchanged_development_gate'],
 'case_counts':{'validation10':10,'dynamic6':6},'sha256':files,
 'recovery':{'validation10_inference_reused':True,
             'original_failed_unit':'fluid-control-fcp003b-posteval-wait-20261005.service',
             'resume_runner_sha256':sha(root/'recovery_v2/immutable_resume_runner.sh')},
 'frozen_test_accessed':False,'ppo_auto_launched':False}
target=root/'receipt.json'
with tempfile.NamedTemporaryFile('w',dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
echo FC_P003B_POSTEVAL_COMPLETE
