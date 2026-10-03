#!/usr/bin/env bash
# Candidate-specific validation10 endpoint Gate for train20+train8 FNOs.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
candidate_relative="${DYNAMIC_FNO_CANDIDATE:-}"
lineage_relative="${DYNAMIC_FNO_LINEAGE:-}"
output_relative="${VALIDATION_OUTPUT:-}"
development_gate_relative="${DYNAMIC_FNO_DEVELOPMENT_GATE:-}"

[[ "$mode" == "--dry-run" || "$mode" == "--execute" ]] || {
  echo "mode must be --dry-run or --execute" >&2; exit 2;
}
[[ "$candidate_relative" == artifacts/tandem_fno_dynamic_train8_h* && "$candidate_relative" != *..* ]] || {
  echo "DYNAMIC_FNO_CANDIDATE must be a reviewed project-relative dynamic artifact" >&2; exit 2;
}
[[ "$lineage_relative" == artifacts/tandem_cylinders/dynamic_fno_candidate_lineage_* && "$lineage_relative" != *..* ]] || {
  echo "DYNAMIC_FNO_LINEAGE must remain under the dedicated artifact root" >&2; exit 2;
}
[[ "$output_relative" == artifacts/tandem_cylinders/dynamic_fno_formal_validation_* && "$output_relative" != *..* ]] || {
  echo "VALIDATION_OUTPUT must be a new dedicated formal-validation artifact" >&2; exit 2;
}
candidate="$root/$candidate_relative"
lineage="$root/$lineage_relative"
output="$root/$output_relative"
[[ -d "$candidate" && -f "$lineage" ]] || { echo "candidate/lineage is absent" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing existing validation output" >&2; exit 2; }

readarray -t identity < <(python3 - "$root" "$candidate" "$lineage" <<'PY'
import importlib.util
import json
import pathlib
import sys

repo, candidate, receipt = map(pathlib.Path, sys.argv[1:])
path = repo / "scripts/audit_dynamic_fno_candidate_lineage.py"
spec = importlib.util.spec_from_file_location("candidate_lineage_preflight", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
actual = module.build(repo, candidate)
stored = json.loads(receipt.read_text(encoding="utf-8"))
if stored != actual or stored.get("status") != "DYNAMIC_FNO_CANDIDATE_LINEAGE_PASS":
    raise SystemExit("candidate lineage receipt differs from recomputation")
print(stored["candidate_kind"].removeprefix("dynamic_train8_h"))
print(stored["checkpoint_sha256"])
PY
)
horizon="${identity[0]}"
checkpoint_sha="${identity[1]}"
[[ "$horizon" == 50 || "$horizon" == 100 ]] || { echo "unsupported horizon" >&2; exit 2; }
config="$root/conf/tandem_fno_dynamic_train8_h${horizon}.yaml"
checkpoint="$candidate/best"
data="$root/data/curated/tandem_cylinders_matched_start_full40_v1"
predeclaration="$root/artifacts/tandem_cylinders/matched_start_full40_predeclared_20261003.json"
[[ "$(find "$checkpoint" -maxdepth 1 -name 'FNO.*.mdlus' | wc -l)" -eq 1 ]] || {
  echo "candidate best view must contain exactly one model" >&2; exit 2;
}
[[ "$(sha256sum "$checkpoint"/FNO.*.mdlus | awk '{print $1}')" == "$checkpoint_sha" ]] || {
  echo "candidate checkpoint changed after lineage receipt" >&2; exit 2;
}
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || {
  echo "PhysicsNeMo image ID mismatch" >&2; exit 2;
}

common=(--rm --network none --cpus 8 --memory 64g --pids-limit 512
  --cap-drop ALL --security-opt no-new-privileges --read-only
  --tmpfs /tmp:rw,nosuid,nodev,size=4g --user "$(id -u):$(id -g)"
  --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPATH=/workspace/src:/workspace/scripts --env OMP_NUM_THREADS=4
  --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly"
  --mount "type=bind,src=$root/src,dst=/workspace/src,readonly"
  --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly"
  --mount "type=bind,src=$data/validation,dst=/workspace/data/validation,readonly"
  --mount "type=bind,src=$data/manifest.json,dst=/workspace/data/manifest.json,readonly"
  --mount "type=bind,src=$data/normalization.json,dst=/workspace/data/normalization.json,readonly"
  --mount "type=bind,src=$checkpoint,dst=/workspace/checkpoint,readonly"
  --mount "type=bind,src=$predeclaration,dst=/workspace/predeclaration.json,readonly"
  --mount "type=bind,src=$output,dst=/workspace/output"
  --workdir /workspace "$image")

evaluate=(docker run --gpus device=0 "${common[@]}"
  python -u /workspace/scripts/spark_gpu_guard.py --min-free-gib 20
  --allocator-fraction 0.45 --margin-gib 4 --
  python -u /workspace/scripts/evaluate_tandem_fno.py
  --data /workspace/data --normalization-data /workspace/data
  --config "/workspace/conf/tandem_fno_dynamic_train8_h${horizon}.yaml"
  --checkpoint-dir /workspace/checkpoint --split validation
  --horizons 1 10 50 100 --segment-stride 25 --evaluation-batch-size 4
  --action-mode observed --visualizations-per-horizon 0
  --output /workspace/output/evaluation.json
  --segment-metrics-output /workspace/output/segments.json)

endpoint_audit=(docker run "${common[@]}"
  python -u /workspace/scripts/audit_full40_validation_gate.py
  --report /workspace/output/evaluation.json --segments /workspace/output/segments.json
  --predeclaration /workspace/predeclaration.json
  --checkpoint-dir /workspace/checkpoint --data /workspace/data
  --config "/workspace/conf/tandem_fno_dynamic_train8_h${horizon}.yaml"
  --image-id "$image_id" --output /workspace/output/endpoint_gate.json)

handoff=(python3 scripts/audit_dynamic_fno_formal_handoff.py
  --repo "$root" --candidate-root "$candidate" --lineage "$lineage"
  --endpoint-gate "$output/endpoint_gate.json" --output "$output/formal_handoff.json")
if [[ -n "$development_gate_relative" ]]; then
  [[ "$development_gate_relative" == artifacts/tandem_cylinders/dynamic_fno_development_gate_* && "$development_gate_relative" != *..* ]] || {
    echo "DYNAMIC_FNO_DEVELOPMENT_GATE must remain under its dedicated artifact root" >&2; exit 2;
  }
  handoff+=(--development-gate "$root/$development_gate_relative")
fi

if [[ "$mode" == "--dry-run" ]]; then
  printf 'DYNAMIC_FNO_FORMAL_VALIDATION_PREFLIGHT_ONLY_NO_FROZEN_NO_PPO\n'
  printf 'candidate_kind=dynamic_train8_h%s checkpoint_sha256=%s episode_steps_limit=100\n' "$horizon" "$checkpoint_sha"
  printf 'evaluate:'; printf ' %q' "${evaluate[@]}"; printf '\n'
  printf 'audit:'; printf ' %q' "${endpoint_audit[@]}"; printf '\n'
  printf 'handoff:'; printf ' %q' "${handoff[@]}"; printf '\n'
  printf 'development_admission=blocked_until_candidate_stepwise_force_window_evidence_is_recomputed\n'
  exit 0
fi
[[ "${DYNAMIC_FNO_FORMAL_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_DYNAMIC_FNO_FORMAL_VALIDATION" ]] || {
  echo "reviewed dynamic formal-validation token is required" >&2; exit 2;
}
mkdir "$output"
"${evaluate[@]}"
"${endpoint_audit[@]}"
"${handoff[@]}"
