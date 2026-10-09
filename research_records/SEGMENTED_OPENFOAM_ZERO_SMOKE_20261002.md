# Segmented OpenFOAM zero-action observation smoke (2026-10-02)

## Scope and provenance

This is a real numerical CFD run, not a public experimental dataset, an RL result, or evidence of improved control. It tests whether a solver can stop at each prospective control interval, restart without changing the zero-action solution, and expose the 67-channel wake/force observation used by the HydroGym-facing interface.

The case is two tandem cylinders (diameter 1, center spacing L/D=5) at Reynolds number 100, with rotation imposed on the rear cylinder. The source is the existing `tandem_backward_dt005` OpenFOAM baseline at t=80. The mesh has 19,290 cells. OpenFOAM v2512 `pimpleFoam` runs in the digest-pinned `run_openfoam.sh` container, with `deltaT=0.005`, a 0.1-time-unit control interval, and 20 solver steps per interval. The first interval is a zero-action warm-up to obtain a fresh raw observation. No policy is present in this smoke.

## Reproduce on DGX Spark

Run in the project root with the existing OpenFOAM container available. These commands create a **new** case and output directory and refuse to overwrite either; change the suffix if rerunning.

```bash
python3 cfd/tandem_cylinders/make_probe_feedback_case.py probe_feedback_zero_smoke_20261002 --steps 3
python3 scripts/run_tandem_probe_feedback_zero.py probe_feedback_zero_smoke_20261002 --output artifacts/tandem_cylinders/probe_feedback_zero_smoke_20261002
python3 scripts/validate_tandem_segmented_zero_reference.py \
  --cases-root cfd/tandem_cylinders/cases \
  --case probe_feedback_zero_smoke_20261002 \
  --run-result artifacts/tandem_cylinders/probe_feedback_zero_smoke_20261002/result.json \
  --output docs/results/tandem_segmented_zero_reference_20261002.json
```

The initializer clones only the real t=80 restart and relevant constant/system configuration into its isolated case. The runner invokes the pinned solver three times, reads 32 two-component wake probes plus rear-cylinder Cd and Cl plus the applied angular velocity (67 values), and checks solver completion, Courant number, continuity, finite values, and restart time. Raw solver time directories and logs remain only in `cfd/tandem_cylinders/cases/probe_feedback_zero_smoke_20261002`; per-step output is in the ignored `artifacts/tandem_cylinders/probe_feedback_zero_smoke_20261002` directory. The compact independently regenerated audit is committed at `docs/results/tandem_segmented_zero_reference_20261002.json`.

## Observed result and limits

The three segments reached t=80.1, 80.2, and 80.3. All had 20 CFD steps, clean solver exit, maximum Courant number below 0.243, and maximum absolute global continuity per step below 3.6e-13. The segmented observations matched the corresponding monolithic baseline: the largest probe absolute difference was 3.73e-9 and rear Cd/Cl differences were zero at recorded precision. This validates short zero-action restart/observation plumbing only; it does not validate nonzero boundary updates, long-horizon error, grid convergence, RL performance, or drag reduction.

Next: after the 20-epoch PhysicsNeMo FNO independent-test and ablation gate finishes, freeze a candidate policy and test nonzero bounded actions first on a short isolated CFD case. Compare its trajectory against a matched zero-action reference from the same t=80 restart; do not describe surrogate reward as real-CFD benefit. The medium-mesh constant-rotation grid-pair check is running separately and must be reported before a physical-control claim.
