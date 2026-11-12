#!/bin/bash
# multi_eval.sh <root> <tag> <label=checkpoint_rel> ...: evaluate each checkpoint on all DEV+VAL cases (300 steps, offset 0), sequentially
P=$1; TAG=$2; shift 2
for a in "$@"; do
  L=${a%%=*}; CK=${a#*=}; ID=mev_$L; R=$P/experiments/runs/$ID
  [ -f $R/results/metrics.json ] && continue
  python3 - $P/experiments/recipes/$ID.json <<PY
import json, sys
json.dump({'eval_id': '$ID', 'code_tag': '$TAG', 'split': 'fsi_viv', 'roles': ['DEV', 'VAL'], 'checkpoint': '$CK', 'offsets': [0], 'steps': 300}, open(sys.argv[1], 'w'), indent=1)
PY
  until [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -ge 20 ]; do sleep 60; done
  python3 $P/tools/launch_eval.py $P $P/experiments/recipes/$ID.json
  while docker ps --format '{{.Names}}' | grep -qx $ID; do sleep 20; done
  docker logs $ID > $R/results/container.log 2>&1 || true; docker rm $ID >/dev/null 2>&1 || true
  chmod -R u+w $R/source; rm -rf $R/source $R/inputs; echo "done $ID"
done
echo "all done"
