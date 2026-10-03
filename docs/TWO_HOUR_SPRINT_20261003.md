# Two-hour research sprint: 2026-10-03 03:23–05:23 UTC

The physical objective is unchanged: rear-cylinder rotation of fixed tandem
cylinders at Re=100 and L/D=5 should reduce *system-total* mean drag in real
OpenFOAM CFD by at least 2%, with rear lift-fluctuation RMS no more than 5%
above the phase-matched uncontrolled case and absolute mean rear lift below
10% of the uncontrolled fluctuating RMS. Action magnitude/rate and energy
must be reported. One favorable drag number alone is not success.

## Parallel work and measured outcomes

1. **PhysicsNeMo model:** assess the completed H20/rear-drag FNO on the
   predeclared validation CFD action panel, then train/evaluate the v4
   train-only signed-action acquisition on the compute-only Spark. The v3
   completed model has validation/test/independent-phase 100-step total-drag
   NRMSE of 13.414%/15.540%/10.812%, so it has *not* passed the <=10%
   model gate. Keep official FNO APIs and pinned isolated container.
2. **Real-CFD control:** design one small bounded, physically interpretable
   feedback experiment from the same OpenFOAM restart as its zero-action
   control. Predeclare the observation, action rule, action rate, solver
   interval, analysis interval and success metrics before looking at the
   result. A short replay can establish feasibility but cannot replace the
   required long-window, multiple-phase verification.
3. **Decision and records:** compare true CFD and FNO control ranking,
   inspect model failure modes, check memory/disk and both nodes' resource
   use, update the live dashboard and commit only reproducible milestones to
   GitLab. The primary Spark remains the sole durable data host.

At 05:23 UTC report which of these have completed, with exact physical and
model metrics and failed checks. If no control meets all three physical
checks, state that plainly and continue with the best verified trade-off;
do not weaken criteria or call surrogate-only PPO a physical result. The
two-hour time target is a work window, not evidence that the scientific
objective has been achieved.

Every training job keeps at least 20 GiB unified-memory availability on its
host. New environments remain isolated; no host Python changes.
