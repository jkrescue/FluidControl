# P064 temporal-force residual parameterization — preparation only

Date: 2026-10-07. A preparation-only numerical core and no-save worker now exist in an isolated staging directory and have CPU tests; no GPU probe, candidate training, inference, data mutation, or scientific execution has occurred or is authorized by this document.

## Existing evidence and novelty check

The current aerodynamic FNO consumes six physical planes: normalized `(u,v,p)`, mask, `omega_now`, and `omega_next`; its four force outputs are absolute normalized endpoint forces. The ROI contains both cylinder surfaces. The HDF has the four current forces, although the legacy rollout sample exposes only endpoint targets; HydroGym also maintains the causal current force.

This is not another current-force-input proposal. FC-P022 already expanded the official FNO from 6 to 10 inputs and compared current-force planes with zero planes under full H100 recurrence; its preregistered local support failed. FC-P023 trained only the 96 new input coefficients, and FC-P025 changed their statistical objective; both also failed their local rule. K4 added field/action history, not force history, and improved AR only marginally. Repository search found action-minus-zero force-difference objectives and diagnostics, but no experiment that trained the model to predict the **temporal** force increment `f[t+1]-f[t]` with a causal current-force skip.

## One intervention

Keep the official FNO architecture, six network input channels, output width, flow branch, data/splits, normalization, physical loss, precision, optimizer family, and evaluation unchanged. Change only the aerodynamic force-output parameterization:

`z_hat[t+1] = z_current[t] + delta_z_theta(q[t], mask, omega[t], omega[t+1])`,

where `z=(f-mean)/std`, so the supervised increment is exactly `(f[t+1]-f[t])/std`. H1 uses recorded causal `f[t]`; free AR uses only its previous predicted `f_hat[t]` after the recorded initial force. Field outputs retain their existing meaning. The training objective is still computed on the reconstructed absolute `z_hat[t+1]` with the unchanged original H1/AR force loss; no new coefficient or auxiliary loss is introduced.

The fixed skip adds a causal force state and therefore is a substantive model parameterization, not merely algebraic relabelling. Defining `delta=a(x)-z_current` and adding `z_current` back would reproduce the old function and gradient and is explicitly not this experiment. No force planes are added to the network input.

## Initialization and matched control

Both arms use the same official parent, train data/order, updates, optimizer settings, precision and trainable scope. The absolute arm retains the parent absolute-force output. The residual arm copies all compatible tensors but zeros only the four terminal force-output rows/biases, making its endpoint-0 force prediction exactly causal persistence; hidden and field-output tensors remain matched. This unavoidable initialization difference is part of the parameterization and must be reported, not hidden.

Record identical fixed train panels at update 0 and fixed intermediate/final updates without selecting a checkpoint. Separate:

1. initial-prior effect: residual update-0 persistence versus absolute update-0 parent;
2. learning effect: each arm's terminal minus its own update-0 metrics;
3. terminal representation effect: residual versus absolute at the same fixed terminal budget.

The residual arm may be called learned dynamics only if its final model improves over its own exact zero-increment persistence initialization and the matched absolute terminal on the preregistered original H1 and AR objectives. All individual windows and four force channels remain reported. An initially better persistence prior alone is not learning evidence.

## Budget and decision boundary

The existing 32-update/256-window P064 budget is directly comparable, but current evidence does **not** establish that it is sufficient for the zeroed force head: P022/P023 used 16 updates on six windows and produced only small movements, while P064's 32 updates started from an already trained absolute head. Therefore no execution budget is asserted here.

Before any training approval, a separately reviewed no-save train-only probe must measure initial absolute/residual losses, all 28 aerodynamic gradients, one fixed optimizer-step update norm, clipping, and runtime using the proposed exact protocol. Lead may then freeze one finite budget in advance; it must not be extended after seeing dev results. A positive result must satisfy the learning comparison above. A negative result under a resource-proven but still finite budget rejects only that fixed test, not temporal residuals universally.

Preparation status: the staged probe fixes the first controlled-b00 training window at start 0 and compares two isolated parent copies. It performs one real AdamW step per arm and saves no model. The residual path uses an exact replay/VJP calculation: a no-grad 100-step prediction pass builds the small force-state recurrence, obtains all future AR cotangents, then deterministically replays model chunks. CPU oracle tests match the full unrolled graph for multiple batch sizes and chunk sizes. Actual official-model CPU construction also confirms the terminal force rows are exactly rows 3–6 of a `(7,128)` weight and `(7,)` bias. These are preparation checks, not a result from the proposed GPU probe.

Any trained candidate would undergo the existing fixed development protocol with no changed threshold and no PPO/CFD unless separately approved. This plan does not alter the retained B delivery policy or the already demonstrated real-CFD closed loop.
