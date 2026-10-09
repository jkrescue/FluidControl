# Fixed same-q0 force-response auxiliary candidate — preparation only

No training, inference on real models, PPO or CFD is authorized by this preparation.

Historical preparation statement above: subsequently Lead authorized one E training after independent package review. See [final approval](P064_RESPONSE_AUX_E_TRAINING_APPROVAL_20261007.json) and [actual launch](P064_RESPONSE_AUX_E_LAUNCH_20261007.md). This does not authorize any subsequent evaluation or control run. Canonical source promotion occurred after launch; executed immutable bytes are unchanged. The isolated consumer is archived as `src/fluid_control/dual_fno_response_aux.py`, not a replacement of the original loader.

## Hypothesis and sole intended intervention

Direct local action-minus-zero supervision may improve aerodynamic response prediction. This is not proof that long-AR objective gradients caused the error. Reuse the actual B training's K1 starting model, not the trained B checkpoint: parent manifest SHA `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`, aero tensor `b0ec7405826f785d33407d5b8d222948ebd643c797ab38437315dcd5e31280eb`, flow tensor `89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb`. Existing measured B is the control; do not repeat its training.

Same official FNO, original B high/TF32 runtime and loader, seed20261003, exact 256-window B schedule (192 original +64 controlled b00), fresh AdamW,32 updates×8 windows,lr1.5625e−7, original H1/AR loss, clipping and parameter ownership. Train the same28 aerodynamic tensors; two original aerodynamic biases and separate flow FNO remain frozen. No new architecture. This added loss is project code, not an official PhysicsNeMo built-in.

## Fixed auxiliary rule

Use audited train-only b00/b02/b04/b06 q0, currentω0=0; one negative and one positive next-action endpoint per phase. Receipt `6269562b1c4cc92f32853b9c7260e21f702f7ff8249125071fee27365a2be4ce` binds20 HDF identities; select m075/zero/p075 mechanically. m0375/p0375 first pairs are exact duplicates and excluded. Four initial states, eight contrasts sharing four zero controls: not eight statistically independent observations.

For each update, run one batch12 aerodynamic prediction on four phases×three unique actions. q1 field never enters model input; q1 force is the target. Use original train normalization: mean cancels from deltas. `L_aux=mean_8contrasts,4channels[(predicted_normalized_delta−true_normalized_delta)^2]`, fixed λ=1, chosen before any candidate/development result. Shared zero participates in both gradients. No weight scan or scaling from development.

After the eighth unchanged base-window backward, add `backward(8*λ*L_aux)` so the existing accumulator's single division by8 yields `mean(base gradients)+λ*aux gradient`; then the original one clip and one AdamW step. Keep original loss accounting separate from auxiliary records. Exactly32 extra aerodynamic batch calls/384 sample evaluations, no extra flow calls. This is equal updates/base windows, **not equal compute**. Auxiliary batch12 also differs from original mixed H1/AR batch20; TF32/batching is part of the intervention implementation and prevents attributing all differences to a single physical cause.

## Evaluation and decision

Final checkpoint only. Existing opened b01/b03 fixed16-origin H1–H5 development, all phase/horizon/force/field/persistence metrics; existing fixed-six train H1/AR retention. Same prospective finite engineering choice: both pooled H1 rear-Cl MAE and total-Cd MAE strictly improve B, and both fixed-six H1/AR objectives do not worsen B. Report everything; no post-result rule changes. Passing is not formal science admission or physical success. Four-q0 overfitting risk is severe; no universal amplitude/dynamic-response claim. Stored first-step actions have float32 timestamp quantization, not exact nominal ±.1.

## Implementation and remaining launch preparation

`train_p064_response_aux.py` is a minimal clone of actual frozen B producer8066f4a1; the original producer/immutable closure is untouched. `p064_response_aux.py` reuses the supplied official train Dataset/HDF5Reader, validates its first-frame hashes against the CPU receipt, and builds six-channel K1 inputs. The new candidate kind/status cannot masquerade as old B or be loaded by its unchanged consumer.

CPU tests exercise the real original accumulation function, shared-zero derivative,8×compensation, single Adam step, normalized-force algebra, and real scope function's28/two-frozen contract on synthetic tensors. These are engineering fixtures, not an official-model training/reload proof.

The actual pinned py312 official-reader synthetic integration passed (12×6-channel inputs and first-pair normalization), and the staged runner's full no-execute CLI preflight returned `FC_P064_PREPARATION_ONLY_NO_GPU` with the exact original B source/data/runtime references. The isolated new-kind consumer has positive protocol and negative lambda/old-kind tests; no real candidate exists yet. Before execution approval: finish immutable overlay/import closure and final serialized spec, independent review of the new consumer, and reuse original B resource protections. No execution command is approved yet. No existing scientific output is overwritten.
