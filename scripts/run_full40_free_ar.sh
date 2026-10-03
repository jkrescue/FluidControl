#!/usr/bin/env bash
# Pure-autoregressive official-FNO continuation and independent validation.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
variant="${1:?h20 or h50}"
mode="${2:---dry-run}"
run_id="${3:?unique run id}"
case "$variant" in h20) fraction=.25 ;; h50) fraction=.45 ;; *) exit 2 ;; esac
case "$mode" in --dry-run|--probe|--execute|--resume) ;; *) exit 2 ;; esac
[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || exit 2
image=fluid-control-physicsnemo:2.2.2
image_sha=sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e
data="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
parent="$root/artifacts/tandem_fno_full40_dev30_quickscreen_h20_qs1/best"
out="$root/artifacts/tandem_fno_full40_free_ar_${variant}_${run_id}"
config="tandem_fno_full40_free_ar_${variant}"
if [[ "$mode" == --resume ]]; then
 [[ -f "$out/training_history.json" && -d "$out/checkpoints" ]] || exit 2
 [[ ! -e "$out/resume_predeclaration.json" ]] || exit 2
else
 [[ ! -e "$out" ]] || { echo "Output already exists: $out" >&2; exit 2; }
fi
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_sha" ]] || exit 2
python3 - "$data" "$parent" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,parent=map(Path,sys.argv[1:])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
assert sha(data/'manifest.json')=='5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2'
assert sha(data/'normalization.json')=='f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1'
assert not (data/'frozen_test').exists()
assert len(list((data/'train').glob('*.h5')))==20
assert len(list((data/'validation').glob('*.h5')))==10
assert sha(parent/'FNO.0.5.mdlus')=='a66779c18e4c6c0724f903dd4e767eee643d0867180ad8c6cab95587353537ae'
print('Preflight: official image, fixed parent, train20/validation10 identity verified')
PY
base=(docker run --rm --network none --gpus device=0 --cpus 8 --memory 90g
 --shm-size 2g --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges
 --read-only --tmpfs /tmp:rw,nosuid,nodev,size=4g
 --user "$(id -u):$(id -g)" --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache
 --env "USER=$(id -un)" --env "LOGNAME=$(id -un)"
 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/workspace/src:/workspace/scripts
 --env OMP_NUM_THREADS=4
 --mount "type=bind,src=$root/scripts,dst=/workspace/scripts,readonly"
 --mount "type=bind,src=$root/src,dst=/workspace/src,readonly"
 --mount "type=bind,src=$root/conf,dst=/workspace/conf,readonly"
 --mount "type=bind,src=$data,dst=/workspace/devdata,readonly"
 --mount "type=bind,src=$parent,dst=/workspace/parent,readonly"
 --mount "type=bind,src=$out,dst=/workspace/output"
 --workdir /workspace)
train=("${base[@]}" "$image" python -u /workspace/scripts/spark_gpu_guard.py
 --min-free-gib 20 --allocator-fraction "$fraction" --margin-gib 4 --
 python -u /workspace/scripts/train_tandem_fno_rollout.py --config-name "$config"
 data.root=/workspace/devdata output_dir=/workspace/output
 training.initial_checkpoint=/workspace/parent data.prefetch_factor=0
 hydra.run.dir=/tmp/hydra hydra.output_subdir=null)
if [[ "$mode" == --probe ]]; then
 train+=(training.epochs=1 training.max_train_batches=2 training.max_validation_batches=1)
fi
if [[ "$mode" == --dry-run ]]; then
 printf '%q ' "${train[@]}"; printf '\n'; exit 0
fi
record="$out/predeclaration.json"
log="$out/train.log"
if [[ "$mode" == --resume ]]; then
 record="$out/resume_predeclaration.json"
 log="$out/train.resume.log"
else
 mkdir "$out"
fi
python3 - "$record" "$variant" "$mode" "$run_id" "$(git rev-parse HEAD)" <<'PY'
import json,sys,datetime
from pathlib import Path
p,variant,mode,run_id,commit=sys.argv[1:]
record=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),variant=variant,mode=mode,
 run_id=run_id,git_commit=commit,model='official physicsnemo.models.fno.FNO',teacher_forcing=0,
 train_horizon=20 if variant=='h20' else 50,validation_horizon=100,
 epochs=1 if mode=='--probe' else 8,seed=20261003,
 parent_sha256='a66779c18e4c6c0724f903dd4e767eee643d0867180ad8c6cab95587353537ae',
 frozen_test_accessed=False,ppo_authorized=False,purpose='free autoregression training ablation',
 dataloader_prefetch_factor=0,
 resume_from_own_checkpoint=(mode=='--resume'),
 bitwise_equivalent_uninterrupted_resume=False,
 execution_note='Official synchronous DataLoader avoids installed TensorDict global device-recorder thread race')
if mode=='--resume':
 history=json.loads((Path(p).parent/'training_history.json').read_text())
 record['completed_history_entries']=len(history)
 record['last_completed_history']=history[-1]
Path(p).write_text(json.dumps(record,indent=2)+'\n')
PY
"${train[@]}" 2>&1 | tee "$log"
[[ "$mode" == --execute || "$mode" == --resume ]] || exit 0
# Train process has exited before independent evaluation gets the GPU.
mkdir "$out/validation10"
eval_cmd=("${base[@]}"
 --mount "type=bind,src=$out/best,dst=/workspace/checkpoint,readonly"
 "$image" python -u /workspace/scripts/spark_gpu_guard.py
 --min-free-gib 20 --allocator-fraction .15 --margin-gib 4 --
 python -u /workspace/scripts/evaluate_tandem_fno.py
 --data /workspace/devdata --normalization-data /workspace/devdata
 --config /workspace/conf/tandem_fno_full40_h20.yaml
 --checkpoint-dir /workspace/checkpoint --split validation
 --horizons 1 10 50 100 --segment-stride 25 --evaluation-batch-size 4
 --action-mode observed --visualizations-per-horizon 2
 --output /workspace/output/validation10/evaluation.json
 --segment-metrics-output /workspace/output/validation10/segments.json)
"${eval_cmd[@]}" 2>&1 | tee "$out/validation10/evaluate.log"
python3 scripts/audit_dev30_validation_diagnostic.py \
 --report "$out/validation10/evaluation.json" --segments "$out/validation10/segments.json" \
 --data "$data" --checkpoint-dir "$out/best" --candidate-kind dev30_free_ar_development \
 --output "$out/validation10/diagnostic.json"
