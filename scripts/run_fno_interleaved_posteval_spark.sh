#!/usr/bin/env bash
# Sequential, fail-closed post-evaluation for the FC-P003 interleaved candidate.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
case "$mode" in --dry-run|--execute|--resume) ;; *) echo "mode must be --dry-run, --execute, or --resume" >&2; exit 2 ;; esac

image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
main="$root/artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005"
dev30="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
full40="$root/data/curated/tandem_cylinders_matched_start_full40_v1"
dynamic="$root/data/curated/tandem_cylinders_full40_dynamic_validation_v1"
predecl="$root/artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"

[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || {
  echo "pinned PhysicsNeMo image identity differs" >&2; exit 2;
}
for path in "$dev30" "$full40" "$dynamic"; do
  [[ -d "$path" ]] || { echo "required evaluation data absent: $path" >&2; exit 2; }
done
[[ -f "$predecl" ]] || { echo "full40 predeclaration absent" >&2; exit 2; }

if [[ "$mode" == "--dry-run" ]]; then
  printf 'FC_P003_POSTEVAL_READY_NO_GPU\n'
  printf 'suites=validation10,dynamic6,force-window,development-gate ppo_auto_launch=false\n'
  printf 'candidate=%s\n' "$main"
  exit 0
fi
[[ "${FC_P003_POSTEVAL_TOKEN:-}" == "EXECUTE_APPROVED_FC_P003" ]] || {
  echo "reviewed post-evaluation token required" >&2; exit 2;
}

wait_for_main() {
  while [[ ! -f "$main/completion_receipt.json" ]]; do
    if ! systemctl --user is-active --quiet \
      fluid-control-fcp003-interleaved-lambda10-r2-20261005.service; then
      echo "FC-P003 training stopped without a completion receipt" >&2
      exit 3
    fi
    sleep 30
  done
  python3 - "$main/completion_receipt.json" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("status") != "FC_P003_INTERLEAVED_TRAINING_COMPLETE" or value.get("epochs") != 2:
 raise SystemExit("FC-P003 training completion receipt differs")
PY
}

evaluate_candidate() {
  local label="$1" candidate="$2" expected_kind="$3"
  local out="$candidate/posteval_fc_p003"
  if [[ "$mode" == "--execute" && -e "$out" ]]; then
    echo "$label post-evaluation output already exists; reviewed --resume is required" >&2
    exit 2
  fi
  [[ -d "$candidate/source_snapshot" && -d "$candidate/best" ]] || {
    echo "$label immutable source/checkpoint absent" >&2; exit 3;
  }

  local identity
  identity="$(python3 - "$root" "$candidate" "$expected_kind" <<'PY'
import importlib.util, json, pathlib, sys
repo,candidate=map(pathlib.Path,sys.argv[1:3]); expected=sys.argv[3]
spec=importlib.util.spec_from_file_location("lineage",repo/"scripts/audit_fc_p003_candidate_lineage.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
receipt=module.build(repo,candidate)
if receipt.get("candidate_kind") != expected: raise SystemExit("candidate kind differs")
history=json.loads((candidate/"training_history.json").read_text())
if [row.get("epoch") for row in history] != [1,2]: raise SystemExit("training is not complete at two epochs")
models=list((candidate/"best").glob("FNO.0.*.mdlus")); states=list((candidate/"best").glob("checkpoint.0.*.pt"))
if len(models)!=1 or len(states)!=1: raise SystemExit("best is not one model/state pair")
if models[0].name.split(".")[2] != states[0].name.split(".")[2]: raise SystemExit("best epochs differ")
print(receipt["checkpoint_sha256"])
PY
)"
  local checkpoint_sha="$identity"
  if [[ -f "$out/receipt.json" ]]; then
    python3 scripts/validate_fc_p003_posteval_step.py --repo "$root" \
      --candidate "$candidate" --output "$out" --checkpoint-sha256 "$checkpoint_sha" \
      --step complete
    return
  fi
  mkdir -p "$out/validation10" "$out/dynamic6" "$out/force_window"
  if [[ ! -f "$out/lineage.json" ]]; then
    python3 scripts/audit_fc_p003_candidate_lineage.py \
      --repo "$root" --candidate-root "$candidate" --output "$out/lineage.json"
  else
    python3 - "$root" "$candidate" "$out/lineage.json" <<'PY'
import importlib.util,json,pathlib,sys
repo,candidate,stored=map(pathlib.Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location("lineage",repo/"scripts/audit_fc_p003_candidate_lineage.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
if json.loads(stored.read_text()) != module.build(repo,candidate):
 raise SystemExit("stored lineage differs from strict recomputation")
PY
  fi

  local source="$candidate/source_snapshot"
  local checkpoint="$candidate/best"
  validate_step() {
    python3 scripts/validate_fc_p003_posteval_step.py --repo "$root" \
      --candidate "$candidate" --output "$out" --checkpoint-sha256 "$checkpoint_sha" \
      --step "$1"
  }
  mkdir -p "$out/step_receipts"
  step_receipt() {
    local step="$1"; shift
    python3 - "$out/step_receipts/$step.json" "$step" "$checkpoint_sha" "$@" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
receipt=pathlib.Path(sys.argv[1]); step=sys.argv[2]; checkpoint=sys.argv[3]
paths=[pathlib.Path(value) for value in sys.argv[4:]]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
if not paths or any(not path.is_file() for path in paths):
 raise SystemExit(f"{step} step artifacts are incomplete")
payload={"status":"CONTROL_TRAIN16_POSTEVAL_STEP_COMPLETE","step":step,
         "checkpoint_sha256":checkpoint,"sha256":{str(path):sha(path) for path in paths}}
if receipt.exists():
 if json.loads(receipt.read_text()) != payload: raise SystemExit(f"{step} step receipt differs")
else:
 with tempfile.NamedTemporaryFile("w",dir=receipt.parent,delete=False) as stream:
  temporary=pathlib.Path(stream.name); json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n")
  stream.flush(); os.fsync(stream.fileno())
 try: os.link(temporary,receipt)
 finally: temporary.unlink(missing_ok=True)
PY
  }
  local common=(--rm --network none --cpus 8 --memory 64g --shm-size 2g
    --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges --read-only
    --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)"
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1
    --env "USER=$(id -un)" --env "LOGNAME=$(id -un)"
    --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4
    --mount "type=bind,src=$source/scripts,dst=/workspace/scripts,readonly"
    --mount "type=bind,src=$source/src,dst=/workspace/src,readonly"
    --mount "type=bind,src=$source/conf,dst=/workspace/conf,readonly"
    --mount "type=bind,src=$checkpoint,dst=/workspace/checkpoint,readonly"
    --mount "type=bind,src=$out,dst=/workspace/output"
    --workdir /workspace)

  if [[ ! -f "$out/validation10/evaluation.json" || ! -f "$out/validation10/segments.json" ]]; then
    [[ ! -f "$out/validation10/evaluation.json" && ! -f "$out/validation10/segments.json" ]] || {
      echo "validation10 has only one of evaluation/segments; refusing ambiguous resume" >&2; exit 3;
    }
    docker run --gpus device=0 "${common[@]}" \
    --mount "type=bind,src=$dev30,dst=/workspace/devdata,readonly" "$image" \
    python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
    python -u scripts/evaluate_tandem_fno.py --data /workspace/devdata \
    --normalization-data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml \
    --checkpoint-dir /workspace/checkpoint --split validation --horizons 1 10 50 100 \
    --segment-stride 25 --evaluation-batch-size 4 --action-mode observed \
    --visualizations-per-horizon 0 --output /workspace/output/validation10/evaluation.json \
    --segment-metrics-output /workspace/output/validation10/segments.json \
      2>&1 | tee "$out/validation10/evaluate.resume.log"
  fi
  if [[ ! -f "$out/validation10/diagnostic.json" ]]; then
    python3 "$source/scripts/audit_dev30_validation_diagnostic.py" \
    --report "$out/validation10/evaluation.json" --segments "$out/validation10/segments.json" \
    --data "$dev30" --checkpoint-dir "$checkpoint" --candidate-kind dev30_free_ar_development \
      --output "$out/validation10/diagnostic.json"
  fi

  # The launch snapshot predates the scoped CLI-key repair in this audit only.
  # Use the current reviewed audit for post-processing and bind its SHA below;
  # inference remains entirely launch-snapshot code.
  if [[ ! -f "$out/validation10/endpoint_gate.json" ]]; then
    docker run "${common[@]}" \
    --mount "type=bind,src=$root/scripts/audit_full40_validation_gate.py,dst=/workspace/recovery/audit_full40_validation_gate.py,readonly" \
    --mount "type=bind,src=$full40/validation,dst=/workspace/devdata/validation,readonly" \
    --mount "type=bind,src=$full40/manifest.json,dst=/workspace/devdata/manifest.json,readonly" \
    --mount "type=bind,src=$full40/normalization.json,dst=/workspace/devdata/normalization.json,readonly" \
    --mount "type=bind,src=$predecl,dst=/workspace/predeclaration.json,readonly" "$image" \
    python -u /workspace/recovery/audit_full40_validation_gate.py \
    --report /workspace/output/validation10/evaluation.json \
    --segments /workspace/output/validation10/segments.json \
    --predeclaration /workspace/predeclaration.json --checkpoint-dir /workspace/checkpoint \
    --data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml \
      --image-id "$image_id" --output /workspace/output/validation10/endpoint_gate.json
  fi
  validate_step validation10
  step_receipt validation10 "$out/validation10/evaluation.json" \
    "$out/validation10/segments.json" "$out/validation10/diagnostic.json" \
    "$out/validation10/endpoint_gate.json"

  if [[ ! -f "$out/dynamic6/evaluation.json" || ! -f "$out/dynamic6/segments.json" ]]; then
    [[ ! -f "$out/dynamic6/evaluation.json" && ! -f "$out/dynamic6/segments.json" ]] || {
      echo "dynamic6 has only one of evaluation/segments; refusing ambiguous resume" >&2; exit 3;
    }
    docker run --gpus device=0 "${common[@]}" \
    --mount "type=bind,src=$dynamic,dst=/workspace/dynamic,readonly" \
    --mount "type=bind,src=$dev30,dst=/workspace/devdata,readonly" "$image" \
    python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
    python -u scripts/evaluate_tandem_fno.py --data /workspace/dynamic \
    --normalization-data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml \
    --checkpoint-dir /workspace/checkpoint --split validation --horizons 1 10 50 100 \
    --segment-stride 1 --evaluation-batch-size 8 --action-mode observed \
    --visualizations-per-horizon 0 --output /workspace/output/dynamic6/evaluation.json \
    --segment-metrics-output /workspace/output/dynamic6/segments.json \
      2>&1 | tee "$out/dynamic6/evaluate.resume.log"
  fi
  if [[ ! -f "$out/dynamic6/diagnostic.json" ]]; then
    python3 cfd/tandem_cylinders/audit_full40_dynamic6_fno.py \
    --data "$dynamic" --checkpoint "$checkpoint" \
    --checkpoint-epoch "$(basename "$checkpoint"/FNO.0.*.mdlus | cut -d. -f3)" \
    --expected-model-sha "$checkpoint_sha" --report "$out/dynamic6/evaluation.json" \
    --segments "$out/dynamic6/segments.json" \
    --physical-qc artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json \
      --output "$out/dynamic6/diagnostic.json"
  fi
  validate_step dynamic6
  step_receipt dynamic6 "$out/dynamic6/evaluation.json" \
    "$out/dynamic6/segments.json" "$out/dynamic6/diagnostic.json"

  if [[ ! -f "$out/force_window/result.json" ]]; then
    docker run --gpus device=0 "${common[@]}" \
    --mount "type=bind,src=$dynamic,dst=/workspace/dynamic,readonly" \
    --mount "type=bind,src=$dev30,dst=/workspace/devdata,readonly" "$image" \
    python -u scripts/spark_gpu_guard.py --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 -- \
    python -u scripts/diagnose_fno_force_window.py --data /workspace/dynamic \
    --normalization-data /workspace/devdata --config /workspace/conf/tandem_fno_full40_h20.yaml \
    --checkpoint-dir /workspace/checkpoint --expected-model-sha "$checkpoint_sha" \
    --output /workspace/output/force_window/result.json \
      2>&1 | tee "$out/force_window/diagnose.resume.log"
  fi
  if [[ ! -f "$out/development_gate.json" ]]; then
    python3 "$source/scripts/audit_dynamic_fno_development_gates.py" \
    --force-window "$out/force_window/result.json" --checkpoint-sha256 "$checkpoint_sha" \
      --output "$out/development_gate.json"
  fi
  validate_step force_window
  step_receipt force_window "$out/force_window/result.json" \
    "$out/development_gate.json"

  python3 - "$root" "$candidate" "$out" "$label" "$expected_kind" "$checkpoint_sha" <<'PY'
import hashlib,json,pathlib,sys
repo,candidate,out=map(pathlib.Path,sys.argv[1:4]); label,kind,checkpoint=sys.argv[4:]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
lineage=json.loads((out/"lineage.json").read_text())
development=json.loads((out/"development_gate.json").read_text())
endpoint=json.loads((out/"validation10/endpoint_gate.json").read_text())
files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"receipt.json","outer.log"}}
runtime={
 "status":"FC_P003_POSTEVAL_COMPLETE",
 "branch":label,"candidate_kind":kind,"checkpoint_sha256":checkpoint,
 "validation_endpoint_status":endpoint.get("status"),
 "development_gate_status":development.get("status"),
 "ppo_auto_launched":False,"frozen_test_accessed":False,
 "lineage_sha256":sha(out/"lineage.json"),
 "launch_receipt_sha256":lineage["launch_receipt_sha256"],
 "completion_receipt_sha256":lineage["completion_receipt_sha256"],
 "endpoint_audit_recovery_sha256":sha(repo/"scripts/audit_full40_validation_gate.py"),
 "dynamic6_audit_sha256":sha(repo/"cfd/tandem_cylinders/audit_full40_dynamic6_fno.py"),
 "reuse_validator_sha256":sha(repo/"scripts/validate_fc_p003_posteval_step.py"),
 "sha256":files,
}
(out/"receipt.json").write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")
PY
}

wait_for_main
evaluate_candidate interleaved_lambda10 "$main" paired_stats_interleaved_lambda10
