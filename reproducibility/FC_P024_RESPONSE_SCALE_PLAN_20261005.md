# P024 response-scale mechanism diagnostic — CPU preparation approval

P023 completed with local_support=false. It isolates a real but small causal
input effect; increasing coefficient norm64-fold did not give proportional output
change. H1bias regression is concentrated in window816. Neither fundamental
tradeoff nor sufficient training is established by those sixteen updates.

Lead approves CPU implementation/tests only for the following fixed diagnostic.
GPU execution requires final hashes, review and immutable resource-guarded launch.

## Question and scope

Along the measured HIGH96-coefficient direction, does greater input effect change
the sign of H1 mean-bias response while preserving AR waveform behavior, or does
the observed tradeoff grow? This is a local response/curvature study, not a search
for an optimum, new learning-rate selection, candidate training or admission.

Use unchanged official expanded P018 FNO plus reviewed adapter367a532b, frozen
flow histories, exact same six causal windows, original J0/metrics/fullH100 and
precision as P023. Parent/dependency/data identities unchanged. Reconstruct the
96float32 HIGH values from pinned resultadfdd9a8; verify actual block and effective
weight identity equal the recorded terminal identity. No optimizer/backward.

## Preregistered finite protocol

Fixed execution order scales0,1,-1,8,64 of this single recorded HIGH vector.
Scale0 is original zero-block parent; scale1 is the actual observed terminal.
First repeat each0/1 panel twice and require raw normalized predictions, metrics
and aggregate values exactly reproduce the corresponding P023 initial/HIGH rows.
If reproduction fails, stop scientific interpretation and diagnose, do not silently
introduce a tolerance after seeing results. No extra scale execution before this.

Then scales-1,8,64, each with two repeated panels and the same six windows.
All inputs remain causal; H1 true-current and AR self-predicted recurrence remain
unchanged. No terminal zero-input ablation needed: P023 has already established it.
Total60window evaluations,6000paired model calls plus600frozenflow cache calls.
Record actual scaled96 values/identities, all normalized predictions, per-window
physical mean/RMS/centered errors and original objectives. Check old-base hashes,
zero gradients, RNG/mode and no optimizer creation; final restorezero verified.

## Interpretation and limits

Report every scale; negative scale provides a signed-direction comparison.
8/64 are substantial extrapolations, not demonstrated safe actions or a validated
model. No full-data or real-CFD claim. Compare scale1 with-1 to characterize local
direction, and positive scales with0/1 to describe curvature and per-window error.
Do not pick the numerically best scale as a candidate or call this optimization.
Any next training hypothesis must be separately justified and approved.

No physical or surrogate acceptance threshold changes. Preserve all failures.
Nonfinite prediction, source/restoration/reproduction mismatch or resource breach
ends this run; investigate rather than skip that scale or retry automatically.
One official isolated GPU container, .06allocator,12GiBcontainer; startupfree30/
available50GiB; bothMemFree/MemAvailable>=20GiB continuously. Inner900seconds,
external1000seconds,outer1040seconds. Actual budget approved after CPU review.
No candidate saving, full-data expansion, heldout, PPO or CFD launch is authorized.
