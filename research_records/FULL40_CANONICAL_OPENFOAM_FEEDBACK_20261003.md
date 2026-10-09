# Canonical PPO real-OpenFOAM feedback adapter

## Status

The implementation is reviewable and dry-run only at this checkpoint. No new
OpenFOAM case has been staged or solved, and no closed-loop result is claimed.
The historical 67D/high-action PPO feedback runner remains unchanged.

## Fixed physical contract

- OpenFOAM v2512 `pimpleFoam`, tandem circular cylinders at Re=100 and L/D=5.
- Rear-cylinder rotation only, `|omega| <= 0.75` and one control decision every
  `0.1 D/U`; the applied increment is at most `0.1`, so
  `|d omega/dt| <= 1`.
- Exactly 69 policy inputs: 32 wake probes × `(u,v)`, front `(Cd,Cl)`, rear
  `(Cd,Cl)`, and the previously applied omega.
- One matched-start PPO/zero pair, 800 control intervals (`80 D/U`). The
  analysis window is predeclared as the final `60 D/U`, exactly 12,001 force
  samples at solver `dt=0.005`.
- The canonical terminal gate requires total-drag reduction of at least 2%,
  rear `Cl'` RMS no more than `1.05×` the paired zero branch, and absolute mean
  rear `Cl` no more than `0.10×` the paired-zero `Cl'` RMS. A failure remains a
  valid diagnostic result and is never hidden or relabelled.

## Evidence chain

Each feedback interval records:

1. SHA-256 of the exact 69D sensor vector presented to PPO;
2. the deterministic Stable-Baselines3 action marker and policy SHA;
3. requested, bounded, and rate-limited applied omega;
4. the exact rear-cylinder boundary table for the next `0.1 D/U` segment;
5. solver termination, Courant, continuity, and time-end checks;
6. four force coefficients plus probe/force source paths returned by the real
   OpenFOAM segment;
7. SHA-256 of the output observation, which is the next action input.

The zero branch advances from the byte-identical restart in parallel and uses
the same segmented solver schedule.

## Adapter review and runtime lifecycle

The 2026-10-03 strict review found that the first implementation launched a
new policy container once per decision and a new solver container once per
branch and decision.  That would have caused 800 policy image starts and
1,600 OpenFOAM image starts, even though the numerical work still consisted
of only 20 solver steps per control interval.  The reviewed implementation
now loads the exact policy once in one read-only, CPU-only JSON-lines process
and keeps two pinned OpenFOAM containers alive for the paired branches.  Each
decision uses one policy request and two `docker exec` solver segments.  The
container totals are therefore fixed at one policy plus two OpenFOAM starts;
the evidence file records these counts.  Cleanup occurs on normal completion
and exceptions.

The online channel order was checked against
`Full40CanonicalSurrogateFlow.get_observations`: both are exactly
`32*(u,v), front Cd, front Cl, rear Cd, rear Cl, previous applied omega`.
Forces are coefficient-file columns `Cd` and `Cl` at the same exact endpoint
as the probes.  The policy and zero cases are copied from the same five-field
matched restart, advanced over identical `0.1 D/U` intervals, and the zero
action stays identically zero.  The final comparison begins only after the
first `20 D/U` and requires the inclusive `dt=0.005` grid from `start+20` to
`start+80`; no shorter or post-selected window is accepted.

## Fail-closed gates

`scripts/run_full40_canonical_ppo_openfoam_feedback.py --dry-run` verifies:

- the selected policy is the final checkpoint in the formal canonical PPO
  audit and its byte SHA matches;
- canonical PPO readiness, full40 FNO gate, dev30/full40 promotion lineage,
  full40 manifest, normalization, and FNO checkpoint SHA all agree;
- a separately reviewed predeclaration file matches an explicitly supplied SHA
  and binds the case pair, source-state five-file hashes, policy/data lineage,
  mesh, transport, solver and force-function configuration hashes, 800-step
  schedule, and final-60D/U window;
- Spark has at least 40 GiB available memory and 200 GiB disk, Worker has at
  least 40 GiB memory and 100 GiB disk, both nodes are reachable/idle with
  respect to OpenFOAM/Curator work, and pinned OpenFOAM/HydroGym images and
  HydroGym commit are present.

If existing CFD or Curator work is detected, readiness is blocked. The dry-run
does not create cases, mutate OpenFOAM files, invoke PPO, request a GPU, or run
the solver. Formal execution additionally requires an explicit environment
token and a non-blocking project lock.

Frozen-test HDF data are neither opened nor enumerated. Even a canonical joint
PASS from this single matched start remains one coarse-mesh real-CFD pair, not
proof of cross-phase generalization or deployable closed-loop control.
