#!/bin/bash
# eval_milestones.sh <root> <run_id> <case> [updates="50 100 150 200 300"]: evaluate milestone checkpoints as they appear.
P=$1; RUN=$2; CASE=$3
for U in ${4:-50 100 150 200 300}; do
  bash $P/tools/eval_checkpoint.sh $P $RUN $U $CASE >> $P/experiments/runs/$RUN/results/eval.log 2>&1
done
