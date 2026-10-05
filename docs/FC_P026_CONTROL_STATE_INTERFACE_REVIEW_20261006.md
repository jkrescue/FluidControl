# FC-P026 control-state interface review (2026-10-06)

## Status and scope

This is a read-only architecture-gap review. It does not authorize a model,
data-generation, MPC, PPO, formal-evaluation, or admission run. The original
P026 K1/K4 full matched evaluation and all physical thresholds remain
unchanged. At the time of this review K1 formal evaluation is running, not a
terminal result; therefore the conditional follow-up below is not yet eligible.

The fixed problem remains tandem cylinders at Re=100 and L/D=5 with rear-cylinder
rotation. This review inspected repository source, manifests, and existing
reports only. It did not read model files or HDF payloads.

## Corrected repository facts

### Existing historical policy data are already in the training set

`data/curated/tandem_cylinders_directppo_train16_v1/manifest.json` declares 16
train trajectories, 129 frames each, and no validation or frozen-test
trajectories. They are eight b00 episodes and eight b02 episodes from historical
changing exploratory PPO policies. The manifest explicitly says they are not
final-policy on-policy samples. P026 already assigns this family 240 of the
1,368 regular H100 windows, alongside 720 base and 408 train8 windows
(`docs/FC_P026_HISTORY_COMPARISON_PLAN_20261005.md`). Thus “collect historical
PPO trajectories” is not a new intervention, and the existing train16 evidence
must not be described as four-phase or final-policy coverage.

### The real-CFD policy observation is not the FNO state

`TandemSurrogateFlow.get_observations()` in
`src/fluid_control/tandem_hydrogym.py` constructs 69 values: 32 probes of `(u,v)`
(64 values), four forces, and rear-cylinder omega. It does not contain pressure
or the full spatial field. In the same class, the FNO state is the complete
normalized masked `(u,v,p)` tensor, and `TandemFNOStepper._history_step()` builds
the six-channel flow input from that field, the mask, current omega, and next
omega. There is no implemented inverse map from the 69D observation to that
full normalized field.

The real-feedback loop in
`scripts/run_full40_canonical_ppo_openfoam_feedback.py` follows the same sparse
interface: `total_drag_observation_at()` creates the initial 69D observation;
the policy selects an action; OpenFOAM advances one 0.1 D/U control interval;
then a new 69D observation is extracted. The loop does not curate the new CFD
field into an FNO reset tensor. Therefore the current real-CFD feedback path is
a valid policy-observation loop, but is not an implemented recurrent
CFD-to-FNO state-correction bridge.

This distinction does **not** create a new blocker for the project's main PPO
route. A compatible policy may be trained with the surrogate and then deployed
directly on fresh 69D real-CFD observations; the deployed policy does not need
to run or reset the FNO between CFD control intervals. The missing full-state
bridge matters only when the FNO itself remains online as a receding MPC/world
model or when real-CFD observations are intended to correct its rollout state.

### P026 has two distinct histories

`scripts/p026_history_inference.py` represents K1/K4 aerodynamic history
explicitly as full normalized states `[B,K,3,H,W]`, stored applied actions, and a
padding mask. A fresh reset repeats trajectory frame 0 only for unavailable
past slots; subsequent steps shift in the predicted next full field and the
applied action. No future observation is admitted.

Separately, `TandemSurrogateFlow.stage_c_ledger()` uses the causal 62-sample
force history for the one-shedding-period reward. The full-state K4 buffer and
the 62-point force/reward buffer are not interchangeable. A correct receding
reset must define and preserve both; supplying only the 69D policy observation
does neither.

### The current CEM program is an offline legacy screen

`src/fluid_control/cem_mpc.py` is a generic CPU CEM optimizer whose caller must
provide the dynamics-dependent objective. The existing caller,
`scripts/screen_tandem_cem_mpc.py`, loads one legacy seven-output FNO after a
Gate-B receipt, reads one full initial field from HDF, rolls the surrogate open
loop, and evaluates `stage_c_sequence_costs()`. Its own output is labelled
`surrogate_only_cem_screen` and `not_openfoam_control_evidence`.

That caller does not load the P026 dual model, propagate explicit K4 state, use
the current 62-point canonical prehistory at a receding reset, assimilate the
69D real-CFD feedback, or test alternative actions against real CFD. Merely
shortening its horizon would not close these interface gaps.

## Gaps before FNO-in-loop MPC or state-corrected world-model control

1. **State reconstruction/reset:** after each real-CFD segment, the controller
   has a fresh 69D observation but no reviewed transformation to the full
   normalized `(u,v,p)` grid required by the flow FNO.
2. **History reconstruction:** a K4 aerodynamic call needs three causal past
   full fields plus stored applied actions. These cannot be recovered from one
   69D observation. Frame-0 padding is defined only for a true episode reset,
   not an arbitrary mid-trajectory correction.
3. **Reward continuity:** the canonical cost requires the causal 62-point force
   history. A receding planner must carry or reconstruct it without duplicating
   the reset endpoint.
4. **Planner integration:** the existing CEM caller has neither the P026 dual
   loader/history profile nor a real-CFD receding update. Those are project
   integration tasks, not features supplied automatically by PhysicsNeMo.
5. **Action identifiability:** each recorded state in current data has one
   realized action. It can test prediction along that recorded action, but it
   cannot establish counterfactual action ranking needed to validate an MPC
   optimizer.

These gaps do not invalidate the current H100 gates. H100 remains the agreed
test of a long autonomous surrogate rollout and is especially relevant to a
world-model PPO path. A future short-horizon receding MPC may need less than
H100 between corrections, but it first needs a real state-correction interface;
shorter horizon alone is not evidence of closed-loop adequacy.

## One conditional, bounded audit using existing data

Only if **both** P026 K1 and K4 complete their unchanged formal suites and both
are rejected, a separate approval may consider this train/development audit:

- Use the terminal K1 and K4 candidates and existing recorded trajectories; do
  not generate new CFD or refit either model.
- At eligible origins with a stored full CFD field, initialize the exact full
  normalized state. For K4, use the actual preceding three stored fields and
  actions where available, retaining the documented frame-0 padding cases.
- Follow the **recorded** applied action for exactly H10. Do not search actions.
- Form the canonical 62-point window from 52 causal true force endpoints ending
  at the rollout origin `t` (so `t` appears exactly once), followed by the 10
  predictions for `t+1` through `t+10`. Report the unchanged physical
  quantities: total-drag error, rear-lift centered RMS and mean bias, plus
  per-step four-force errors. Retain warm/padded labels and group train16
  results by b00/b02 and episode.
- Compare persistence, K1, and K4 under identical origins and actions. An
  existing dynamic-development set may be reported with the same fixed
  protocol, but must not select horizon, model, loss, or thresholds.

This would test a narrow claim: whether recorded-action H10 predictions are
accurate enough to justify engineering a state bridge and a later action-ranking
test. It cannot establish MPC benefit, PPO readiness, counterfactual ranking, or
scientific admission. Passing it would only motivate a separately reviewed and
approved bridge experiment; it would confer no automatic authorization.
Failing it would close the current P026-to-H10-MPC route
under this representation and budget; it would not prove that all short-history
or receding-horizon approaches are impossible.

## Source anchors

- `AGENTS.md`: fixed geometry/Reynolds number, unchanged gates, and MPC as a
  proposed rather than implemented capability.
- `data/curated/tandem_cylinders_directppo_train16_v1/manifest.json`: exact
  train16 provenance, counts, phases, and non-final-policy scope.
- `docs/FC_P026_HISTORY_COMPARISON_PLAN_20261005.md`: 720/408/240 window
  exposure, K1/K4 causal history, frame-0 padding, and unchanged formal suite.
- `src/fluid_control/tandem_hydrogym.py`: full normalized FNO state, 69D policy
  observation, explicit P026 history advance, and 62-point reward ledger.
- `scripts/p026_history_inference.py`: K1/K4 buffer shapes, causal packing,
  padding, cloning, and predicted-state shift.
- `scripts/run_full40_canonical_ppo_openfoam_feedback.py`: actual 0.1 D/U
  policy/OpenFOAM/69D-observation feedback cadence.
- `src/fluid_control/cem_mpc.py` and `scripts/screen_tandem_cem_mpc.py`: generic
  optimizer and the current legacy surrogate-only screen.

Primary research gives context but not a project-specific guarantee: reduced
models for flow control must be judged by control-relevant prediction, and
robust learning MPC explicitly treats model error as a closed-loop concern
([Deda, Wolf & Dawson, *Physical Review Fluids* 9, 063904 (2024)](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.9.063904),
[Buerger et al., L4DC 2024](https://proceedings.mlr.press/v242/buerger24a.html)).
Neither result supplies the missing 69D-to-field bridge or relaxes this
project's force/window criteria.
