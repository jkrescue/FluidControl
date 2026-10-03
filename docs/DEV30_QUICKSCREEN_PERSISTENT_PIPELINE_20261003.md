# Persistent dev30 quick-screen pipeline

`scripts/run_full40_dev30_quickscreen_pipeline_spark.sh` is a thin,
fail-closed orchestration layer over the already reviewed one-step, H20, and
validation-only wrappers. It changes no model, data split, seed, epoch count,
metric, or threshold.

The service runs 10 one-step epochs, hashes the unique best official
PhysicsNeMo FNO checkpoint, passes that exact SHA to the 5-epoch H20 wrapper,
then hashes the unique H20 candidate and runs H1/H10/H50/H100 on validation10.
Shell `errexit`, an exclusive pipeline lock, and post-stage contract checks
prevent a failed stage from starting its successor. Every stage has an
independent run-ID output, refuses overwrite, and records manifest,
train-only-normalization, runner, checkpoint, image, and commit lineage.
Neither the pipeline nor its child containers mount a frozen path.

The final state remains `STAGE_DIAGNOSTIC_ONLY`: it does not satisfy the
formal full40 promotion/Gate or authorize PPO, even if its development metrics
look favorable.

After the immutable dev30 planner reports `FULL40_DEV30_FNO_RETRAIN_READY`, a
reviewed persistent launch is:

```bash
systemd-run --user --unit=fluid-control-dev30-quickscreen-pipeline-qs1 \
  --collect \
  --working-directory=/workspace/fluid_control \
  --property=Type=exec --property=Restart=no \
  --property=CPUQuota=800% --property=MemoryHigh=56G --property=MemoryMax=64G \
  --property=TasksMax=512 --property=OOMPolicy=stop --property=Nice=5 \
  --setenv=FULL40_DEV30_QUICKSCREEN_PIPELINE_TOKEN=EXECUTE_REVIEWED_DEV30_QUICKSCREEN_PIPELINE \
  bash scripts/run_full40_dev30_quickscreen_pipeline_spark.sh --execute qs1
```

Monitor with `systemctl --user status` and `journalctl --user -fu` for that
exact unit. A retry must use a new run ID; partial outputs are retained for
audit and are never resumed or overwritten.
