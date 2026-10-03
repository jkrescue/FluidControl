# Full40 four-hour critical path audit (2026-10-03)

This is a read-only scheduling audit. It does not authorize CFD, Curator,
training, validation, or frozen-test access. The operational deadline used
below is 11:35 UTC on 2026-10-03.

## Evidence at 07:59 UTC

Measured state:

- Extension RAW CFD: all 11 new train cases complete; 8/10 validation cases
  complete and two active; 0/10 frozen cases generated. CFD was completing a
  four-case wave in about eight minutes, so RAW generation was not the main
  development-data bottleneck.
- HDF5: 6/9 commissioning cases complete, with the last three active. One
  extension case was in Curator parity and no extension HDF5 was complete.
- Spark has 20 CPU cores. Four simultaneous Curator processes were each using
  about 310--350% CPU, so the host was already close to CPU saturation.
- Two completed three-case Curator waves each took about 1,382 seconds per
  case. This supports a measured Spark throughput of three cases per roughly
  23 minutes, not eight cases per 23 minutes.
- Available unified memory was about 110 GiB. Memory capacity was healthy, but
  CPU and memory-bandwidth contention still matter on GB10 unified memory.

Historical training measurements:

- Comparable one-step 30-epoch training took about 59 minutes.
- H20 training on 28 train trajectories took 65.4 minutes for five epochs.
  Linear scaling to 20 trajectories and ten epochs gives about 93 minutes.
- A previous four-case H1/H10/H50/H100 validation took about 2.3 minutes per
  model. Ten-case full40 validation is budgeted at approximately six minutes.

The 93-minute H20 value and six-minute validation value are linear estimates,
not measurements of the not-yet-existing full40 release. Together with the
measured one-step time, a finalized development dataset requires at least
about 158 minutes before a strict validation result, excluding finalization
and scheduling overhead.

## Earliest honest schedules

The development release needs 21 newly curated HDF5 files: 11 train and 10
validation. If the currently active parity case finishes around 08:13 UTC,
about 20 development cases remain.

- Unvalidated optimistic lower bound: simultaneous Spark-3 plus Worker-8
  throughput gives 11 slots, hence two 23-minute waves. HDF5 would finish no
  earlier than about 08:59. A 10--20 minute immutable-release finalization
  gives 09:09--09:19, and training plus validation ends about 11:47--11:57.
  This is a lower bound, not a forecast: Worker eight-way parity and sustained
  throughput had not been demonstrated.
- More credible eight-slot bound: 20 cases require three waves, or about 69
  minutes. HDF5 ends about 09:22, release finalization about 09:32--09:42, and
  training plus validation about 12:10--12:20.
- Spark-only measured throughput: seven waves are needed for the remaining
  development cases. HDF5 ends around 10:54 before finalization; the strict
  result is correspondingly much later.

To finish by 11:35, an immutable train20+validation10 release would need to be
fully finalized by about 08:57. No evidence-backed schedule meets that time.
The four-hour target therefore must not be reported as achievable.

## Development-only immutable release

A train20+validation10 immutable view is scientifically useful because it can
decouple model development from deterministic frozen curation. It does not
create Curator throughput and is not an emergency deadline workaround.

It is safe only if all of these conditions hold:

1. Exact train and validation case lists and HDF5 SHA-256 values are fixed.
2. Normalization and state-support bounds use train20 only.
3. The training container can physically mount only the development view;
   mounting the whole repository while merely filtering split names is not the
   strongest leakage guard.
4. Frozen cases remain in a separate Worker path, and no frozen metrics,
   summaries, ranks, or file enumeration can enter checkpoint selection.
5. The later frozen release is one-time and cannot mutate the development
   hashes.

The current finalizer requires all 31 remainder HDF5 files and the current
training container bind-mounts the repository. Therefore training concurrently
with unfinished frozen curation is **not safe through the current path**. A
reviewed immutable development release is worth implementing as durable
infrastructure, but only by reusing and testing the existing finalizer
contracts rather than bypassing them under deadline pressure.

## Safe acceleration order

1. Finish and verify Worker Curator parity.
2. Curate train cases first, then validation, using case-exclusive atomic
   outputs and SHA-verified transfer. Increase concurrency only after a real
   wave confirms throughput and memory behavior.
3. Keep heavy Curator work off Spark while GPU training runs. Spark is a
   unified-memory machine; spare capacity alone does not eliminate CPU and
   memory-bandwidth interference.
4. Do not shorten epochs or weaken the fixed 10% H100, 0.023 delta-Cd, action
   ordering, lift, or real-CFD gates.
5. A training retry must use a new run ID. Any pre-existing output directory,
   including a partial failed run, is immutable and must be rejected. When an
   H20 run follows a retried one-step run, its reviewed parent run ID must be
   supplied explicitly rather than silently falling back to the failed path.

## Time-varying action seed decision

The fixed-action full40 set does not adequately teach action transitions. A
minimum useful future seed is eight train-only cases:

- phases b00, b02, b04, and b06 only;
- two sign-mirrored low-amplitude schedules per phase;
- identical predeclared source-restart hashes within each phase;
- action support within +/-0.75 and delta-omega at most 0.1 per 0.1 D/U;
- +/-0.375 primary plateaus, causal linear slew, and alternating/zero returns
  approximately once per shedding period;
- the existing 80 D/U run and final 60 D/U force statistics, with all four
  force channels, action history, solver health, and provenance retained.

Four cases, one schedule per phase, are only a commissioning diagnostic. Eight
cases are still a seed for dynamic action primitives, not a complete closed-
loop dataset. Validation and frozen phases must not be used to choose these
schedules.

At four-way OpenFOAM throughput the eight RAW cases need roughly two waves, or
about 16 minutes. At measured Spark Curator throughput they need at least three
waves, about 69 minutes, plus VTK export and verified transfer. Starting them
inside the current four-hour window would consume the bottleneck needed by
full40 and delay the primary result. Decision: **do not start them now**. Run
them after development curation completes, or on an isolated Worker after GPU
training starts, without changing the frozen policy.
