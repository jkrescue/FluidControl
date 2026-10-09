# FC-P018 — controlled full-data lower-learning-rate training

Preparation and CPU testing approved; GPU execution requires separate approval
after implementation and independent review. This continues existing work; it
does not replace the official model, CFD data, controller objective or baseline.

## Hypothesis and evidence

P017 result `ebfb80fc9bbc4b701c6450062895a99e1383f0899153a1cf91fb202f966d495d`
shows negative first-step directional derivative and a lower objective at the
fixed positive 1/64 displacement, while the full step raises the same objective
20.48-fold. Initial/restored evaluations agree; actual full-step tensors replay
exactly. This supports local overshoot, not convergence or a universally optimal
learning rate. P016 terminal simultaneous-repair hypothesis was unsupported.

Test whether one fixed smaller learning rate improves the complete unchanged
control-prediction evaluation after the same full-data budget as P015.

## Matched baseline and single intervention

Baseline P015: original 44 train HDF files, same P009 initial model/state,
same 1368 H100 windows and original order, eight-window raw-gradient mean,
clip norm1 once, fresh AdamW171 updates, lr1e-5 and weight_decay1e-4.
Immutable trainer SHA `2f5de1a946b040c42c21f8164da41a5afc642fd6174e7bb35372bb7ba98eb996`.
Use pinned accumulation_step and diagnostic_panel helpers with explicit SHA.
Same P013 numerical objective/P014 diagnostics and official PhysicsNeMo image.

P018 changes only lr to `1.5625e-7` (1e-5/64). AdamW betas(.9,.999), eps1e-8,
weight_decay1e-4, seed20261003, TF32/high, H1/AR weights and four-force/rear-Cl
objective remain unchanged. Flow FNO stays frozen; same28 aerodynamic parameter
tensors train, same2 lifting biases frozen. No architecture or data expansion.

Lower lr also lowers AdamW's effective decoupled shrinkage per update; it does
not preserve later optimizer-state trajectories or scale P015 displacements
uniformly. Report as learning-rate intervention, not isolated displacement cause.

## Effective configuration and identity

Retain original YAML as explicitly labelled base model/data configuration.
Add an immutable training-protocol JSON with exactly this fixed override; bind
its SHA in launcher, trainer, manifest, saved metadata and result. The optimizer
must read the validated protocol; assert actual param-group lr matches it.
Do not imply the base YAML is the full effective configuration.

Use distinct FC-P018 status/kind/experiment throughout. Preserve P015 artifacts.
Dual loader, audit, CPU dual reload and formal profile need explicit compatible
P018 support; no fallback to P015 identity and no bypass on missing manifest.

## Budget, measurements and decision

Exactly171 updates /1368 original-order windows. Panels at consumed0/456/912/1368
are diagnostics only. Save terminal only; no early selection, resume, extra steps,
lr scan or validation-based tuning. Same train-only normalization and splits.
Require gradient/optimizer finite checks and frozen-flow/two-bias invariants.
Record actual lr/protocol identity alongside each update or verified invariant.
Official terminal flow/aerodynamic checkpoint fresh reload must reproduce tensors.

After complete independent training audit and actual dual runtime reload, use
the SAME original validation10/dynamic6/force-window full formal protocol and
numeric thresholds. Compare P015/P009 with matching aggregations; never exchange
macro/pooled, endpoint/window or train/validation quantities. A fixed-flow result
cannot claim new flow-field accuracy. Repeated development validation is not test.

Failure: retain negative result, analyze actual optimization/error evidence,
no automatic extra training and no PPO. Pass: separately approve compatible new
PPO through HydroGym/SB3 and paired real-OpenFOAM feedback under original physical
criteria (drag reduction>=2%, Cl-prime RMS ratio<=1.05, mean-Cl constraint and
unchanged action/rate limits). Training completion alone is not admission.

## Resources and responsibilities

Main Spark GPU0 only. Pinned official image, isolated train-only mounts and no
concurrent GPU jobs. Keep both MemAvailable and MemFree>=20GiB, startup free>=30GiB;
allocator fraction0.15 pending resource review. Upper training runtime4h with
bounded exact-owned-container cleanup; never blindly restart failed runs.
Root owns plan/protocol/launcher, Surrogate owns new trainer/tests,
independent Evaluation owns separate semantic review and audits. Reuse existing
evaluation numerical source; adapter compatibility is engineering, not acceptance.
