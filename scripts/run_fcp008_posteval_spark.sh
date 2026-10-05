#!/usr/bin/env bash
# Thin, resumable calibrated-candidate post-evaluation; dry-run is the default.
set -euo pipefail

profile="${FCP_POSTEVAL_PROFILE:-p008}"
root="${FCP008_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"; cd "$root"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--execute|--resume|--wait) ;; *) echo "invalid mode" >&2; exit 2;; esac
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
case "$profile" in
  p008)
    candidate="$root/artifacts/fcp008_force_readout_candidate_20261005"
    out="$candidate/posteval_fc_p008"
    auditor_relative="scripts/audit_fcp008_candidate.py"
    candidate_kind="full_train_force_row_recalibration"
    calibrated_kind="FC_P008_TRAIN_ONLY_FORCE_ROW_CANDIDATE"
    diagnostic_kind="fc_p008_force_row_calibrated_epoch0"
    lineage_status="FC_P008_CANDIDATE_LINEAGE_PASS"
    step_status="FC_P008_POSTEVAL_STEP_COMPLETE"
    complete_status="FC_P008_POSTEVAL_COMPLETE"
    chain_status="FC_P008_IMMUTABLE_POSTEVAL_CHAIN_STAGED"
    precision_status="FC_P008_FORMAL_EVALUATION_DEFAULT_TF32_HIGH"
    formal_status="FC_P008_FORMAL_EVALUATION_APPROVED"
    token_expected="EXECUTE_APPROVED_FC_P008_POSTEVAL"
    ;;
  p009)
    candidate="${FCP_POSTEVAL_CANDIDATE:-$root/artifacts/fcp009_joint_force_row_candidate_20261005}"
    out="$candidate/posteval_fc_p009"
    auditor_relative="scripts/audit_fcp009_candidate.py"
    candidate_kind="joint_h1_free_ar_force_row_recalibration"
    calibrated_kind="FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE"
    diagnostic_kind="fc_p009_joint_force_row_calibrated_epoch0"
    lineage_status="FC_P009_CANDIDATE_LINEAGE_PASS"
    step_status="FC_P009_POSTEVAL_STEP_COMPLETE"
    complete_status="FC_P009_POSTEVAL_COMPLETE"
    chain_status="FC_P009_IMMUTABLE_POSTEVAL_CHAIN_STAGED"
    precision_status="FC_P009_FORMAL_EVALUATION_DEFAULT_TF32_HIGH"
    formal_status="FC_P009_FORMAL_EVALUATION_APPROVED"
    token_expected="EXECUTE_APPROVED_FC_P009_POSTEVAL"
    ;;
  *) echo "invalid post-evaluation profile" >&2; exit 2;;
esac
dev30="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
full40="$root/data/curated/tandem_cylinders_matched_start_full40_v1"
dynamic="$root/data/curated/tandem_cylinders_full40_dynamic_validation_v1"
predecl="$root/artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
physical_qc="$root/artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json"
chain_root="${FCP008_POSTEVAL_CHAIN_ROOT:-$root}"
auditor="$chain_root/$auditor_relative"
validator="$chain_root/scripts/validate_fcp008_posteval.py"
sha() { sha256sum "$1" | awk '{print $1}'; }
host_python="${FCP008_HOST_PYTHON:-/home/USER/env_isaaclab/bin/python}"
[[ -x "$host_python" ]] || { echo "pinned host Python is unavailable" >&2; exit 2; }
kind_args=()
if [[ "$profile" == p009 ]]; then kind_args=(--expected-calibrated-kind "$calibrated_kind"); fi

if [[ "$mode" != --dry-run && -z "${FCP008_POSTEVAL_CHAIN_ROOT:-}" ]]; then
  reviewed="${FCP008_REVIEWED_COMMIT:-}"
  [[ "$reviewed" =~ ^[0-9a-f]{40}$ && "$(git rev-parse "$reviewed^{commit}")" == "$reviewed" ]] || { echo "reviewed FC-P008 posteval commit required" >&2; exit 2; }
  snapshot="$root/artifacts/${profile}_posteval_chain_${reviewed:0:12}_immutable"
  [[ ! -e "$snapshot" ]] || { echo "immutable queue snapshot already exists; invoke it directly" >&2; exit 2; }
  temporary="$(mktemp -d "$root/artifacts/.${profile}-posteval-snapshot.XXXXXX")"
  mkdir -p "$temporary/numerical_source"
  if [[ "$profile" == p009 ]]; then
    numerical_commit="ab7b9fe5e442c044319d3d686cb1e1d2fc0b4a82"
  else
    numerical_commit="7216214b545fbbd50b2fb5ed866f231039b06b18"
  fi
  # Hydra composes the evaluation child config through other files in conf/;
  # freeze the complete config tree instead of only the leaf YAML.
  git archive "$numerical_commit" -- src scripts conf cfd \
    | tar -x -C "$temporary/numerical_source"
  for relative in scripts/run_fcp008_posteval_spark.sh "$auditor_relative" scripts/validate_fcp008_posteval.py; do
    mkdir -p "$temporary/$(dirname "$relative")"
    git show "$reviewed:$relative" >"$temporary/$relative"
  done
  chmod +x "$temporary/scripts/"*.py "$temporary/scripts/"*.sh
  python3 - "$temporary/receipt.json" "$temporary" "$reviewed" "$(git rev-parse "$reviewed^{tree}")" "$numerical_commit" "$(git rev-parse "$numerical_commit^{tree}")" "$chain_status" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target,root=map(pathlib.Path,sys.argv[1:3]); commit,tree,numerical_commit,numerical_tree,status=sys.argv[3:]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob("*")) if path.is_file()}
payload={"status":status,"git_commit":commit,"git_tree":tree,"numerical_source_commit":numerical_commit,"numerical_source_tree":numerical_tree,"sha256":files}
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
  mv "$temporary" "$snapshot"; chmod -R a-w "$snapshot"
  exec env FCP_POSTEVAL_PROFILE="$profile" FCP_POSTEVAL_CANDIDATE="$candidate" FCP008_REPO_ROOT="$root" FCP008_POSTEVAL_CHAIN_ROOT="$snapshot" \
    FCP008_REVIEWED_COMMIT="$reviewed" FCP008_EXECUTION_APPROVAL_SHA256="${FCP008_EXECUTION_APPROVAL_SHA256:-}" \
    FCP008_FORMAL_APPROVAL="${FCP008_FORMAL_APPROVAL:-}" FCP008_FORMAL_APPROVAL_SHA256="${FCP008_FORMAL_APPROVAL_SHA256:-}" \
    FCP008_POSTEVAL_TOKEN="${FCP008_POSTEVAL_TOKEN:-}" "$snapshot/scripts/run_fcp008_posteval_spark.sh" "$mode"
fi
source="$chain_root/numerical_source"
chain_receipt="$chain_root/receipt.json"
host_source="$source"; [[ -d "$host_source/src" ]] || host_source="$root"
host_path="$host_source/src:$host_source/scripts"

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "pinned image differs" >&2; exit 2; }
[[ "$(sha "$dev30/manifest.json")" == 5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2 ]]
[[ "$(sha "$full40/manifest.json")" == 1c9086b4d08da57e84fb4f6a22376f0935596727952a50c55acb1a130e77e90e ]]
[[ "$(sha "$dynamic/manifest.json")" == bfa49031a1ab2b8e10a5cc5d95f1fbd8713e289c3ac8155838334e3c3d007dae ]]
[[ "$(sha "$predecl")" == d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b ]]
[[ "$(sha "$physical_qc")" == 9723203f922cbe6609f2c92b7b48d299ee9d948d694d3c6c2eab413472421d86 ]]

if [[ "$mode" == --dry-run ]]; then
  env PYTHONPATH="$host_path" "$host_python" "$auditor" --repo "$root" >/dev/null
  echo "${complete_status}_DRY_RUN_PASS_NO_GPU"
  echo "candidate=$candidate"
  echo "protocol=validation10_stride25_batch4,dynamic6_stride1_batch8,force_window6,unchanged_development_gate"
  echo "candidate_and_independent_formal_approval_required=true"
  exit 0
fi

[[ "${FCP008_POSTEVAL_TOKEN:-}" == "$token_expected" ]] || { echo "posteval token required" >&2; exit 2; }
candidate_approval_sha="${FCP008_EXECUTION_APPROVAL_SHA256:-}"
[[ "$candidate_approval_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "candidate execution approval SHA required" >&2; exit 2; }
formal_approval="${FCP008_FORMAL_APPROVAL:-}"
formal_approval_sha="${FCP008_FORMAL_APPROVAL_SHA256:-}"
[[ -f "$formal_approval" && "$formal_approval_sha" =~ ^[0-9a-f]{64}$ && "$(sha "$formal_approval")" == "$formal_approval_sha" ]] || { echo "formal evaluation approval required" >&2; exit 2; }
if [[ "$mode" == --wait ]]; then
  while [[ ! -f "$candidate/completion_receipt.json" ]]; do sleep 30; done
  mode=--execute
fi
[[ -f "$candidate/completion_receipt.json" ]] || { echo "FC-P008 candidate is incomplete" >&2; exit 3; }
lineage_json="$(env PYTHONPATH="$host_path" "$host_python" "$auditor" --repo "$root" --candidate "$candidate" --execution-approval-sha256 "$candidate_approval_sha")"
checkpoint_sha="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["checkpoint_sha256"])' <<<"$lineage_json")"
checkpoint_state_sha="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["checkpoint_state_sha256"])' <<<"$lineage_json")"
checkpoint_relative="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["checkpoint_relative_directory"])' <<<"$lineage_json")"
checkpoint="$candidate/$checkpoint_relative"
[[ "$checkpoint_relative" == candidate_build/candidate && -d "$checkpoint" ]] || { echo "audited checkpoint directory differs" >&2; exit 2; }
python3 - "$formal_approval" "$formal_approval_sha" "$checkpoint_sha" "$checkpoint_state_sha" "$formal_status" "$profile" "$candidate" <<'PY'
import hashlib,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); expected,model,state,status,profile=sys.argv[2:7]; candidate=pathlib.Path(sys.argv[7])
if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise SystemExit("formal approval SHA differs")
value=json.loads(path.read_text())
required={"status":status,"candidate_model_sha256":model,"candidate_state_sha256":state,"formal_evaluation_authorized":True,"frozen_test_accessed":False,"ppo_auto_launch":False}
if profile=="p009":
 sha=lambda item:hashlib.sha256(item.read_bytes()).hexdigest()
 required.update(candidate_kind="FC_P009_TRAIN_ONLY_JOINT_FORCE_ROW_CANDIDATE",candidate_result_sha256=sha(candidate/"candidate_build/result.json"),candidate_completion_receipt_sha256=sha(candidate/"completion_receipt.json"),protocol=["validation10_H1_H10_H50_H100_stride25_batch4","dynamic6_H1_H10_H50_H100_stride1_batch8","force_window6","unchanged_development_gate"])
if any(value.get(k)!=v for k,v in required.items()): raise SystemExit("formal approval contract differs")
PY
if [[ -f "$out/receipt.json" ]]; then
  env PYTHONPATH="$source/src:$source/scripts" "$host_python" "$validator" --profile "$profile" --repo "$root" --candidate "$candidate" --output "$out" --lineage "$out/lineage.json" --numerical-source "$source" --chain-receipt "$chain_receipt" --formal-approval-sha256 "$formal_approval_sha" --step complete
  exit 0
fi
if [[ "$mode" == --execute && -e "$out" ]]; then echo "existing partial output requires --resume" >&2; exit 2; fi
mkdir -p "$out/validation10" "$out/dynamic6" "$out/force_window" "$out/step_receipts" "$out/evidence"
formal_copy="$out/evidence/formal_evaluation_approval.json"
if [[ ! -f "$formal_copy" ]]; then cp "$formal_approval" "$formal_copy"; fi
[[ "$(sha "$formal_copy")" == "$formal_approval_sha" ]] || { echo "stored formal approval differs" >&2; exit 2; }
if [[ ! -f "$out/lineage.json" ]]; then
  env PYTHONPATH="$host_path" "$host_python" "$auditor" --repo "$root" --candidate "$candidate" --execution-approval-sha256 "$candidate_approval_sha" --output "$out/lineage.json" >/dev/null
else
  env PYTHONPATH="$host_path" "$host_python" - "$auditor" "$root" "$candidate" "$candidate_approval_sha" "$out/lineage.json" <<'PY'
import importlib.util,json,pathlib,sys
path,repo,candidate=map(pathlib.Path,sys.argv[1:4]); approval=sys.argv[4]; stored=pathlib.Path(sys.argv[5])
spec=importlib.util.spec_from_file_location("lineage",path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
if json.loads(stored.read_text()) != mod.validate_candidate(repo,candidate,approval): raise SystemExit("stored lineage differs")
PY
fi
lineage_sha="$(sha "$out/lineage.json")"
chain_receipt_sha="$(sha "$chain_receipt")"
common=(--rm --network none --cpus 8 --memory 64g --shm-size 2g --pids-limit 512 --cap-drop ALL
 --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)"
 -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1 -e "USER=$(id -un)" -e "LOGNAME=$(id -un)"
 -e PYTHONPATH=/workspace/src:/workspace/scripts -v "$source/scripts:/workspace/scripts:ro" -v "$source/src:/workspace/src:ro"
 -v "$source/conf:/workspace/conf:ro" -v "$source/cfd:/workspace/cfd:ro" -v "$checkpoint:/workspace/checkpoint:ro"
 -v "$out:/workspace/output:rw" -w /workspace)

if [[ ! -f "$out/precision.json" ]]; then
  tmp="$out/.precision.json.tmp"
  docker run --gpus device=0 "${common[@]}" --entrypoint python "$image" -c \
    "import json,os,torch; value={'status':'$precision_status','official_image_id':'$image_id','NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE'),'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),'float32_matmul_precision':torch.get_float32_matmul_precision()}; expected={'NVIDIA_TF32_OVERRIDE':None,'cuda_matmul_allow_tf32':True,'cudnn_allow_tf32':True,'float32_matmul_precision':'high'}; assert all(value[k]==v for k,v in expected.items()), value; print(json.dumps(value,sort_keys=True))" >"$tmp"
  mv "$tmp" "$out/precision.json"
fi

validate_step() { env PYTHONPATH="$source/src:$source/scripts" "$host_python" "$validator" --profile "$profile" --repo "$root" --candidate "$candidate" --output "$out" --lineage "$out/lineage.json" --numerical-source "$source" --chain-receipt "$chain_receipt" --formal-approval-sha256 "$formal_approval_sha" --step "$1"; }
step_receipt() {
  local step="$1"; shift
  python3 - "$out/step_receipts/$step.json" "$out" "$step" "$checkpoint_sha" "$checkpoint_state_sha" "$lineage_sha" "$chain_receipt_sha" "$formal_approval_sha" "$step_status" "$candidate_kind" "$@" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
target,out=map(pathlib.Path,sys.argv[1:3]); step,model,state,lineage,chain,formal,status,kind=sys.argv[3:11]; paths=list(map(pathlib.Path,sys.argv[11:]))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if any(not path.is_file() or out.resolve() not in path.resolve().parents for path in paths): raise SystemExit("step artifacts incomplete or escape output")
payload={"status":status,"candidate_kind":kind,"step":step,"checkpoint_epoch":0,"checkpoint_sha256":model,"checkpoint_state_sha256":state,"lineage_sha256":lineage,"posteval_chain_receipt_sha256":chain,"formal_evaluation_approval_sha256":formal,"precision_sha256":sha(out/"precision.json"),"sha256":{str(path.relative_to(out)):sha(path) for path in paths}}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
}

if [[ ! -f "$out/validation10/evaluation.json" || ! -f "$out/validation10/segments.json" ]]; then
  [[ ! -f "$out/validation10/evaluation.json" && ! -f "$out/validation10/segments.json" ]] || { echo "partial validation pair" >&2; exit 3; }
  docker run --gpus device=0 "${common[@]}" -v "$dev30:/workspace/devdata:ro" "$image" python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- python -u scripts/evaluate_tandem_fno.py --data /workspace/devdata --normalization-data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --allow-calibrated-epoch-zero --expected-calibrated-model-sha256 "$checkpoint_sha" --expected-calibrated-state-sha256 "$checkpoint_state_sha" "${kind_args[@]}" --split validation --horizons 1 10 50 100 --segment-stride 25 --evaluation-batch-size 4 --action-mode observed --visualizations-per-horizon 0 --output /workspace/output/validation10/evaluation.json --segment-metrics-output /workspace/output/validation10/segments.json 2>&1 | tee "$out/validation10/evaluate.log"
fi
[[ -f "$out/validation10/diagnostic.json" ]] || env PYTHONPATH="$source/src:$source/scripts" "$host_python" "$source/scripts/audit_dev30_validation_diagnostic.py" --report "$out/validation10/evaluation.json" --segments "$out/validation10/segments.json" --data "$dev30" --checkpoint-dir "$checkpoint" --candidate-kind "$diagnostic_kind" --allow-calibrated-epoch-zero --expected-calibrated-model-sha256 "$checkpoint_sha" --expected-calibrated-state-sha256 "$checkpoint_state_sha" "${kind_args[@]}" --output "$out/validation10/diagnostic.json"
if [[ ! -f "$out/validation10/endpoint_gate.json" ]]; then
  docker run "${common[@]}" -v "$full40/validation:/workspace/devdata/validation:ro" -v "$full40/manifest.json:/workspace/devdata/manifest.json:ro" -v "$full40/normalization.json:/workspace/devdata/normalization.json:ro" -v "$predecl:/workspace/predecl.json:ro" "$image" python -u scripts/audit_full40_validation_gate.py --report /workspace/output/validation10/evaluation.json --segments /workspace/output/validation10/segments.json --predeclaration /workspace/predecl.json --checkpoint-dir /workspace/checkpoint --data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml --image-id "$image_id" --allow-calibrated-epoch-zero --expected-calibrated-model-sha256 "$checkpoint_sha" --expected-calibrated-state-sha256 "$checkpoint_state_sha" "${kind_args[@]}" --output /workspace/output/validation10/endpoint_gate.json
fi
[[ -f "$out/step_receipts/validation10.json" ]] || step_receipt validation10 "$out/validation10/evaluation.json" "$out/validation10/segments.json" "$out/validation10/diagnostic.json" "$out/validation10/endpoint_gate.json"
validate_step validation10

if [[ ! -f "$out/dynamic6/evaluation.json" || ! -f "$out/dynamic6/segments.json" ]]; then
  [[ ! -f "$out/dynamic6/evaluation.json" && ! -f "$out/dynamic6/segments.json" ]] || { echo "partial dynamic pair" >&2; exit 3; }
  docker run --gpus device=0 "${common[@]}" -v "$dynamic:/workspace/dynamic:ro" -v "$dev30:/workspace/devdata:ro" "$image" python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- python -u scripts/evaluate_tandem_fno.py --data /workspace/dynamic --normalization-data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --allow-calibrated-epoch-zero --expected-calibrated-model-sha256 "$checkpoint_sha" --expected-calibrated-state-sha256 "$checkpoint_state_sha" "${kind_args[@]}" --split validation --horizons 1 10 50 100 --segment-stride 1 --evaluation-batch-size 8 --action-mode observed --visualizations-per-horizon 0 --output /workspace/output/dynamic6/evaluation.json --segment-metrics-output /workspace/output/dynamic6/segments.json 2>&1 | tee "$out/dynamic6/evaluate.log"
fi
[[ -f "$out/dynamic6/diagnostic.json" ]] || env PYTHONPATH="$source/src:$source/scripts" "$host_python" "$source/cfd/tandem_cylinders/audit_full40_dynamic6_fno.py" --data "$dynamic" --checkpoint "$checkpoint" --checkpoint-epoch 0 --expected-model-sha "$checkpoint_sha" --report "$out/dynamic6/evaluation.json" --segments "$out/dynamic6/segments.json" --physical-qc "$physical_qc" --output "$out/dynamic6/diagnostic.json"
[[ -f "$out/step_receipts/dynamic6.json" ]] || step_receipt dynamic6 "$out/dynamic6/evaluation.json" "$out/dynamic6/segments.json" "$out/dynamic6/diagnostic.json"
validate_step dynamic6

if [[ ! -f "$out/force_window/result.json" ]]; then
  docker run --gpus device=0 "${common[@]}" -v "$dynamic:/workspace/dynamic:ro" -v "$dev30:/workspace/devdata:ro" "$image" python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- python -u scripts/diagnose_fno_force_window.py --data /workspace/dynamic --normalization-data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml --checkpoint-dir /workspace/checkpoint --expected-model-sha "$checkpoint_sha" --allow-calibrated-epoch-zero --expected-calibrated-state-sha256 "$checkpoint_state_sha" "${kind_args[@]}" --output /workspace/output/force_window/result.json 2>&1 | tee "$out/force_window/diagnose.log"
fi
[[ -f "$out/development_gate.json" ]] || env PYTHONPATH="$source/src:$source/scripts" "$host_python" "$source/scripts/audit_dynamic_fno_development_gates.py" --force-window "$out/force_window/result.json" --checkpoint-sha256 "$checkpoint_sha" --output "$out/development_gate.json"
[[ -f "$out/step_receipts/force_window.json" ]] || step_receipt force_window "$out/force_window/result.json" "$out/development_gate.json"
validate_step force_window

python3 - "$out" "$checkpoint_sha" "$checkpoint_state_sha" "$lineage_sha" "$chain_receipt_sha" "$formal_approval_sha" "$image_id" "$complete_status" "$candidate_kind" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]); model,state,lineage,chain,formal,image,status,kind=sys.argv[2:]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"receipt.json","outer.log"}}
payload={"status":status,"candidate_kind":kind,"checkpoint_epoch":0,"checkpoint_sha256":model,"checkpoint_state_sha256":state,"lineage_sha256":lineage,"posteval_chain_receipt_sha256":chain,"formal_evaluation_approval_sha256":formal,"precision_sha256":sha(root/"precision.json"),"official_image_id":image,"protocol":["validation10_H1_H10_H50_H100_stride25_batch4","dynamic6_H1_H10_H50_H100_stride1_batch8","force_window6","unchanged_development_gate"],"sha256":files,"frozen_test_accessed":False,"ppo_auto_launched":False}
target=root/"receipt.json"
with tempfile.NamedTemporaryFile("w",dir=root,delete=False) as stream:
 tmp=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,target)
finally: tmp.unlink(missing_ok=True)
PY
validate_step complete
echo "$complete_status"
