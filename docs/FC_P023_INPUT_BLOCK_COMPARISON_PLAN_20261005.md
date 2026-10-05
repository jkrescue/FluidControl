# P023 isolated causal-input learning-rate comparison

Status: scientific protocol recorded before execution; CPU trainer preparation
authorized. GPU execution requires reviewed final hashes and immutable launcher.
Owner: Lead; implementation: Surrogate agent; independent evaluation: Evaluation
agent. Builds on P022, does not replace the established project or thresholds.

## Hypothesis and comparison

P022 updated old and new weights together. It did not establish useful reliance
on current-force inputs, and old-weight changes may obscure that learning.
Freeze every original P018 force-FNO parameter and learn only the 96 new input
coefficients using the verified project adapter around official PhysicsNeMo FNO.
Compare LOW lr=1.5625e-7 against HIGH lr=1e-5, both causal. This tests learning
scale within the isolated-block intervention; it cannot separately prove that
freezing old weights alone caused improvement relative to historical P022.

Parent, data, normalization, source pins and six windows remain those in P022.
Exact current forces are from the independently audited six-window raw sidecar;
no future force input, no HDF rewriting, no heldout or full-data expansion.
Official image b40d5888 and adapter367a532b remain pinned. CPU prerequisite and
actual official-model execution are recorded in FC_P023_CPU_REVIEW_20261005.md
and FC_P023_ROOT_CPU_EXECUTION_20261005.json (commit0f3ee83).

## Fixed protocol

- Six train windows160/816/923/975/1077/1233; all six averaged before clipping1.
- Original J0: half teacher-forced H1 and half full100-step autoregression;
  original balanced force loss. Checkpoint blocks10, no detached recurrence.
- Sixteen updates per arm. Fresh AdamW(.9,.999),eps1e-8,weight_decay1e-4;
  sole optimizer parameter is the 96-element leaf; no old-weight moments/decay.
- Identical zero-block initial state and fixed frozen field histories. Construct
  adapter after base.to(device); no later module move/deepcopy of version state.
- Initial/final panels repeated twice each:48 window evaluations. Each terminal
  also receives two zero-force-input ablations:24 additional window evaluations,
  explicitly diagnostic, not an independent control training arm or validation.
- Record actual96 coefficient values, update/cumulative norms, optimizer scope,
  original parameter/buffer hashes and initial-force dependence. Check ablation
  does not alter weights, gradients, optimizer state or random-generator state.

## Evaluation and decision

Keep the same P022 local H1/AR objective, mean error, RMS-amplitude error and
centered-waveform conditions. HIGH is the experimental arm and LOW the control;
compare both with the shared initial state and report every individual window.
Repeat agreement is an observation, not a rigorous numerical error bound.
No physical meanlift criterion or surrogate-admission threshold changes here.

If local conditions hold, independently review results and input reliance before
separately approving exact-time full-train provenance and one candidate training.
If they fail, preserve measured failure and decide next hypothesis from evidence;
do not automatically extend updates, scan learning rates, save/deploy a candidate
or start PPO. This is not final surrogate admission or real-CFD control success.

## Resource and execution limits

One official isolated GPU container, no concurrent project GPU job. Reuse P022
allocator0.06, container12GiB, startup MemFree>=30GiB/MemAvailable>=50GiB,
continuous MemFree and MemAvailable>=20GiB. Inner1800seconds, external1900 and
outer1940. Record actual peaks; estimated capacity is not observed capacity.
Retain scoped readonly-HDF cache advice only after same-file hash/identity checks;
no global cache clearing.192 full-window backwards,32 total updates,72 endpoint
window evaluations. Stop on resource/nonfinite/identity/mutation violation, retain
failure and diagnose before a separately approved recovery. No blind retries.

All final source/config/approval hashes and actual unit/container identity must
be recorded before reporting execution. Review and sync only owned tested files.
