# P028 flow objective: CPU engineering verification

2026-10-06. This is implementation evidence, not model accuracy or execution approval.

Canonical `scripts/p028_flow_h10_objective.py` SHA256
`28a9a462f72a7ed5965fc97d22de47ef7fd12740edd66c3f505e822eb1ffd41a`;
`tests/test_p028_flow_h10_objective.py` SHA256
`9b9eaff2803c2525efe7685636679c7485babf45206e213f0eb8608ebdd12907`.

Implementation, independent evaluation and Lead canonical rerun each pass10
synthetic CPU tests (Lead0.96s). No real HDF/model/GPU access. The project helper
uses the existing official-model prediction interface; it is not an NVIDIA API.

Covered: exact first10 transitions from the original H100/101 action batch;
t/t+1 action and target alignment; fluid mask in every recurrent update and
loss; later losses backpropagating through earlier steps without detach;
unused four force-output rows receiving zero gradient; exactly eight raw window
gradients averaged before any optimizer step; seven/nine-window rejection;
strict shape and boolean contract; complex gradient magnitude not discarding
imaginary components. No activation checkpointing is used here.

The trainer must separately protect unused output rows/moments against AdamW
weight decay, freeze the complete aerodynamic model, check original sampler
identity, enforce resource limits and verify official save/reload. Its review
is not complete. Actual-size GPU memory feasibility and scientific improvement
remain unproven. Original K1/K4 formal rejection is unchanged.

The existing dashboard was separately updated to show44/44 completed diagnostic
cases rather than optimizer updates, with bound result/review identities and
no admission. Nineteen dashboard regression tests passed; actual API reports
verified completion with running=false and admission=false. Chrome's existing
local dashboard tab was refreshed. No public hosting or data publication.
