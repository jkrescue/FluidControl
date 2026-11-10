#!/bin/bash
# eval_checkpoint.sh <root> <run_id> <update> <case> [steps=300] [split=fsi_viv] [role=DEV]
# Waits for checkpoint <update> of a training run, evaluates it on one case (launch_eval.py), renders figures vs CFD.
# Output: <root>/experiments/runs/eval_<run_id>_u<update>/{results/metrics.json, figures/}
set -e
P=$1; RUN=$2; U=$3; CASE=$4; STEPS=${5:-300}; SPLIT=${6:-fsi_viv}; ROLE=${7:-DEV}
CK=$P/experiments/runs/$RUN/results/checkpoint_u$(printf '%05d' $U).pt
until [ -f $CK ]; do sleep 30; done; sleep 20
TAG=$(python3 -c "import json; print(json.load(open('$P/experiments/runs/$RUN/contracts/config.json'))['code_tag'])")
ID=eval_${RUN}_u$U; R=$P/experiments/runs/$ID
mkdir -p $P/experiments/recipes
python3 - $P/experiments/recipes/$ID.json <<PY
import json, sys
json.dump({'eval_id': '$ID', 'code_tag': '$TAG', 'split': '$SPLIT', 'roles': ['$ROLE'], 'cases': ['$CASE'],
           'checkpoint': '$RUN/results/checkpoint_u$(printf '%05d' $U).pt', 'offsets': [0], 'steps': $STEPS,
           'save_frames': ['$CASE']}, open(sys.argv[1], 'w'), indent=1)
PY
until [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -ge 20 ]; do sleep 60; done
python3 $P/tools/launch_eval.py $P $P/experiments/recipes/$ID.json
while docker ps --format '{{.Names}}' | grep -qx $ID; do sleep 20; done
docker logs $ID > $R/results/container.log 2>&1 || true; docker rm $ID >/dev/null 2>&1 || true
python3 $P/tools/render.py $R $P/data/views/${SPLIT}__eval/$CASE $R/figures
rm -f $R/results/*/*/frames.npz; chmod -R u+w $R/source; rm -rf $R/source $R/inputs
echo "done $R"
