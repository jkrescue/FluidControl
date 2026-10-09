# FC-P023 frozen-parent input-block study: CPU preparation approval

Hypothesis: isolating the new current-force input coefficients avoids old-weight
drift, and a fixed larger learning scale may provide useful causal conditioning.
P022 did not establish this: its original loss decreased in both arms, but local
joint support failed; nonzero gradients are not proof of useful input reliance.

Root approves staged CPU implementation/tests only under the constraints below.
No GPU experiment, full-data expansion or threshold change is approved here.

## Parameter isolation and official APIs

Use the same pinned official PhysicsNeMo expanded FNO, all its original parameters
frozen with no gradients. A separate zero-initialized nn.Parameter of shape
24x4x1x1 (96 scalars) is the sole AdamW parameter. Each forward assembles the first
lifting tensor as [frozen physical6, new block4, frozen coordinate2] and uses
torch.func.functional_call to invoke the unmodified official forward. This is
project-owned training glue, not a new PhysicsNeMo API or invented neural model.
No full-tensor masked AdamW that decays old columns; no persistent overwrites.

Root inspected actual pinned image b40d5888 on CPU only: PyTorch
2.13.0a0+8145d630e8.nv26.06 provides functional_call(module, parameter_and_buffer_dicts,
args=None, kwargs=None, tie_weights=True, strict=False). Installed source
/usr/local/lib/python3.12/dist-packages/torch/_functorch/functional_call.py SHA
dcee4f31fd66bb7641fc1a99dd6eb6ccf1fa1e7e7f3df9a343f24a36d9db2407.
Reference: https://docs.pytorch.org/docs/2.14/generated/torch.func.functional_call.html
The web reference is not the installed-version proof. In-place parameter/buffer
operations must not be allowed to invalidate the frozen-state assumption.

## Required CPU evidence before any GPU approval

- Zero block reproduces parent output; nonzero block matches a materialized
  equivalent official FNO, including gradients to the block and recurrent inputs.
- Full100-step recurrence, checkpoint10 and uncheckpointed gradients agree using
  nonzero block; assemble the effective tensor inside every recomputed forward.
- Original model parameter/buffer bytes and gradients stay unchanged through
  optimizer updates and no-grad endpoint ablations. Only one96-element optimizer
  state exists; old weights have no Adam moments or weight decay.
- Test separate low/high-arm restoration, current-force causality, finite guards,
  and exact mapping physical6/new4/coordinate2. Reuse reviewed P021 recurrence.
- Prepare a tiny official-model CPU verification script for separate Root review
  and run approval. Do not run GPU or revise official source for compatibility.

## Intended bounded scientific comparison, not yet authorized

Same six train windows, original J0, fullH100, average6 thenclip1, sixteen updates
per arm; low1.5625e-7 versus high1e-5, otherwise freshAdamW(.9,.999),eps1e-8,wd1e-4.
Both arms use causal inputs; compare high versus low and identical initial state.
Retain P022 local H1/AR objective/mean/RMS/centered waveform rules, endpoint
repeats, per-window regressions and no automatic extra iterations after failure.
Record actual96 values/update norms and terminal zero-input ablation, clearly
labelled diagnostic rather than independent validation. No candidate deployment.
Resource and execution budget need final approval after implementation review;
both20GiB floors remain mandatory. Full closed-loop goal is unchanged.
