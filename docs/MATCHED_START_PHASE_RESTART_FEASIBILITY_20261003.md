# Matched-start restart phase feasibility audit

Date: 2026-10-03
Decision: **`GO_9_CASE_COMMISSIONING`**
Scope: uncontrolled baseline restart feasibility only; no controlled CFD was launched and no frozen controlled label was read.

## Immutable evidence

The machine-readable predeclaration is:

```text
artifacts/tandem_cylinders/matched_start_phase_restart_predeclared_v3_20261003.json
SHA256 6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603
```

This hash was recorded before any matched-start branch was generated. Future case generators must require this exact manifest SHA and refuse to proceed on a mismatch. Neither restart times nor splits may be changed after observing a controlled force history.

Provenance at audit time:

- Git HEAD: `9d530ac85203d94d35fed0adb54e999edd5f67eb` (new audit files remain uncommitted);
- phase-selection script SHA256: `ba4192da0f9825b18fa40ef540e5071320785852f6795a94bf23278c356f03cc`;
- test SHA256: `17b0f9bc7bef0c8eea00166bfc80f36ee9c0f47cc5a172a90a63244b13a84b95`;
- source rear-force file SHA256: `654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b`.

The v3 manifest records that last value directly as
`signal_qc.force_sha256`; it is the hash of the exact raw rear-`Cl`
algorithm input, not merely a documentation checksum. The earlier unversioned
manifest is superseded because it recorded the input path but omitted this hash.
The intermediate v2 draft used a non-canonical key name and is also superseded.
Neither earlier artifact was used to generate a CFD branch.

The implementation is `scripts/audit_matched_start_phase_restarts.py`; focused tests are in `tests/test_matched_start_phase_restarts.py`. The writer refuses to overwrite an existing audit artifact.

## Fixed phase algorithm

The algorithm was fixed without using any controlled outcome:

1. Read raw rear-cylinder `Cl` from the uncontrolled baseline `tandem_backward_dt005/postProcessing/forceRear/0/coefficient.dat`.
2. Use the fixed `t=80..160 D/U` interval and subtract its arithmetic mean.
3. Define phase zero at each positive-going mean crossing. Linear interpolation between the two adjacent `Cl` samples is used only to locate this scalar diagnostic crossing.
4. Define phase between adjacent crossings by their local elapsed-time fraction. A candidate must be bracketed by two observed crossings; phase is never extrapolated.
5. Consider only real on-disk restart directories at `t>=100`, on the existing `2 D/U` cadence, containing both `U` and `p`.
6. For each target `phi_k=k*pi/4`, select the real restart with the smallest circular phase error. Require absolute error `<=pi/16`.
7. Hash every selected `U` and `p`. CFD states are never interpolated or synthesized.

The baseline has 41 real restart directories over `t=80..160`. Applying the `t>=100` rule and the adjacent-crossing bracket leaves 28 eligible real states, `t=100..154`; `t=156,158,160` are deliberately rejected because the baseline ends before a following positive crossing can bracket them.

## Period and near-repeat audit

Twelve positive crossings were observed in the fixed window. Their first interval is `6.22339 D/U` and the sequence settles near `6.20675 D/U`; the fixed median is:

```text
T = 6.207139389 D/U
period standard deviation = 0.004803903 D/U
```

This differs slightly from the older approximate `6.154 D/U`; the present selection uses the raw fixed-window measurement above, not the historical rounded value.

The limit cycle is highly repetitive. At the nearest raw sample lag to one period (`6.205 D/U`), rear-`Cl` correlation is `0.9999592` and cycle-shift NRMSE is `0.009113`. At two and three periods the correlations remain `0.9999379` and `0.9999050`. Unique file hashes establish that the eight selected states are different files, but they do not make them independent physical conditions.

Consequently, the panel tests phase coverage on one baseline limit cycle. It does not establish robustness to Reynolds number, geometry, inlet disturbances, noise, or long-time non-stationarity, and its frame count must not be reported as an independent-sample count.

## Frozen selection and splits

| bin | target phase | real restart | signed error (rad) | abs error (deg) | split | commissioning |
|---:|---:|---:|---:|---:|---|---|
| 0 | `0` | `148` | `+0.071103` | `4.07` | train | yes |
| 1 | `pi/4` | `130` | `-0.086202` | `4.94` | validation | no |
| 2 | `pi/2` | `106` | `-0.032196` | `1.84` | train | yes |
| 3 | `3pi/4` | `144` | `-0.051153` | `2.93` | frozen test | no |
| 4 | `pi` | `120` | `+0.001238` | `0.07` | train | yes |
| 5 | `5pi/4` | `102` | `-0.153081` | `8.77` | validation | no |
| 6 | `3pi/2` | `134` | `+0.035959` | `2.06` | train | no |
| 7 | `7pi/4` | `110` | `+0.089256` | `5.11` | frozen test | no |

The worst phase error is `0.153081 rad = 8.77 degrees`, below the frozen `pi/16 = 11.25 degrees` ceiling. All restart times and combined `U+p` hashes are unique. Actual selected phase gaps range from `0.62809` to `0.97444 rad`; all eight bins are covered.

The split unit is the restart/phase group, never a frame or sliding segment. All five future actions from one restart remain in its one declared split. The audit only hashes uncontrolled source states for the future frozen groups; it does not read or create frozen controlled trajectories.

## Commissioning authorization boundary

This audit supports exactly the following first-stage panel after review:

```text
train phase bins: 0, 2, 4
restart times:    148, 106, 120
actions:          -0.75, 0, +0.75
case count:       3 x 3 = 9
```

The case generator must verify the manifest SHA, copy the selected real `U/p` files, recheck their individual hashes, and apply the already declared deterministic rate-bounded action protocol. The nine cases are commissioning evidence for solver/action/force/Curator integrity and strict onset pairing. They are not enough for model training, a control claim, or opening the frozen split.

This audit does **not** authorize the full 40-case matrix. That remains a subsequent reviewed step. No restart may be replaced because its controlled result looks unfavorable.

## Baseline rerun decision

A higher-frequency uncontrolled baseline restart run is **not required** for the nine-case commissioning: every bin is covered by a real existing state within the frozen phase tolerance.

If a future hash/config check invalidates any chosen state, the fail-closed fallback is a zero-control OpenFOAM continuation long enough to cover more than one measured period, with `dt=0.005`, force output every step, and real `U/p` writes at no coarser than `0.5 D/U`. For example, an `8 D/U` continuation produces at least 17 real snapshots and resolves eight phase bins without synthesizing states. This fallback must receive a new manifest and hashes before use; it must not overwrite or silently amend the artifact above.

## Reproduction

```bash
python3 -m unittest tests.test_matched_start_phase_restarts

python3 scripts/audit_matched_start_phase_restarts.py \
  --output artifacts/tandem_cylinders/matched_start_phase_restart_predeclared_v3_20261003.json
```

The second command is intentionally non-repeatable at the same output path: an existing predeclaration causes a hard failure rather than silent replacement.
