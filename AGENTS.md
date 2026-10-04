# Research project operating instructions

## Scope and mandatory context

Work in `/workspace/fluid_control` on DGX Spark.
The Mac is an interface, not a scientific compute/storage node. Worker scratch is temporary; return results with hashes to Spark.
Before choosing work, read `PROJECT_STATE.md`, `DECISIONS.md`, `EXPERIMENTS.md`, `experiments/results.csv`, and `docs/RESEARCH_ROADMAP.md`. Inspect live units and artifacts; chat history is not current process evidence.
This project studies two fixed tandem cylinders, Re=100, L/D=5, rear-cylinder rotation only. Do not silently expand to multiple Reynolds numbers, moving cylinders, or other geometries.

## Team and approval

The Lead owns scientific scope, hypotheses, experiment approval, integration and persistent state; it is not primarily a coder. Four concurrent slots are available: Lead, Physics/Data, Surrogate, and Control/Evaluation. Control and Evaluation are separate responsibilities scheduled within one slot; independent evaluation must not be replaced by the training agent's claim.
Only the Lead approves a new expensive experiment after recording hypothesis, parent, one intended change, data/split, immutable configuration, evaluation protocol, resource budget, owner, decision rule, and next action. A technical probe is not a scientific experiment result.
Parallelize independent work. Never edit the same owned file concurrently or launch duplicate jobs. Do not change a running shell script; execute an immutable launch copy.

## Scientific invariants

- Use real CFD and official pinned PhysicsNeMo FNO/Curator/DataPipe components. Custom adapters, losses, OpenFOAM transport and training orchestration must be labelled project code. HydroGym supplies environment interfaces; SB3 supplies PPO.
- Preserve data manifests, train-only normalization, fixed train/validation/frozen splits and model/policy hashes. Repeated validation-based development is not independent testing.
- Evaluate fields, forces, action-minus-zero response and rollout separately. Do not infer control benefit from field L2, a terminal Cd metric, training loss, or surrogate reward.
- A changed surrogate requires a newly trained compatible policy and real-CFD feedback validation. Preserve the successful CFD-only PPO baseline; do not relabel it FNO-assisted.
- No measured improvement claim without the same dataset, starts, horizons, normalization, aggregation and evaluation code/protocol. Unknown metrics remain unknown, not zero.
- Do not lower acceptance thresholds to admit a candidate. Original physical criteria and later development-admission criteria are distinct; see `PROJECT_STATE.md` and the authoritative audit code.
- Uncertainty sampling, MPC and multi-Re coverage are proposed stages, not implemented capabilities or automatic new priorities.

## Execution and recovery

Keep unified physical MemAvailable >=20 GiB, accounting for CPU/GPU sharing; isolate environments. GPU utilization is a resource signal, not evidence of scientific progress.
An incomplete goal requires a next action. Distinguish operational failure, scientific rejection, resource wait, active analysis, stage completion and project completion. Recover approved idempotent steps after verified checks; diagnose unknown faults rather than blindly restart. Preserve failures and completed results. A timer cannot autonomously solve arbitrary new research/code problems; do not claim it can.
After every milestone, update state and ledger from artifacts, record decisions, attach reports/hashes, then push only reviewed owned files. Do not fabricate historical preregistration dates or metadata. Historical reconstructions are explicitly retrospective.
The overall goal is complete only when the accepted surrogate, its controller and the real-CFD constrained closed-loop result are all verified. Finished training or a PASS on one audit is not project completion.
