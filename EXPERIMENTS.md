# Experiment ledger

This is the human-readable index for `experiments/results.csv`. Stable completed-history IDs use `FC-E###`; proposed work is intentionally excluded and uses the separate `FC-P###` namespace.

The entries below are retrospective reconstructions from immutable artifacts and receipts; they are not presented as historical preregistrations.

## Recording rules

- One CSV row is one `experiment × protocol × metric` observation. Protocol names are deliberately distinct: epoch-internal validation, validation10 H100, dynamic6 H100, force-window6, and paired 80D CFD are not interchangeable.
- Blank numeric values mean unknown. `NOT_EVALUATED` means the protocol has not produced verified evidence; it never means zero.
- `checkpoint_sha256` is the model/policy identity. `artifact_sha256` identifies the cited receipt when one exists; `evaluation_manifest_sha256` binds a validation or dynamic panel where one manifest applies. A blank commit or hash is preserved as unknown rather than inferred from the current tree.
- Epoch-internal metrics are training diagnostics only. Admission uses independently produced post-evaluation evidence and unchanged gates.
- FNO development admission and the final physical CFD gate are separate. Passing an endpoint metric cannot override a failed force window, and surrogate metrics cannot substitute for paired OpenFOAM verification.
- The 16 matched training pairs contain only four distinct initial states. Frozen10 is materialized and sealed in full40, but excluded from the development dev30 view and was not opened for this ledger.

## Historical experiments

| ID | Experiment | Hypothesis | Evidence and outcome | Interpretation | Next action |
|---|---|---|---|---|---|
| FC-E001 | B5 immutable parent | Static-train FNO may transfer to controlled trajectories. | validation10 recorded; dynamic6 Cd NRMSE 56.78% and action-delta Cd MAE 0.11461 both fail. | Useful immutable parent and negative control, not a controlled surrogate. | Retain unchanged. |
| FC-E002 | Dynamic train8 H50 | Train-only action trajectories at H50 improve controlled response. | dynamic Cd gate passes; action-delta 0.03009 exceeds 0.023. | Partial endpoint improvement does not admit PPO. | Preserve as H50 development evidence. |
| FC-E003 | Dynamic train8 H100 | An H100 training regimen may improve long-rollout behavior. | validation action-delta 0.023258 and dynamic action-delta 0.030907 both fail 0.023. | The near-threshold static result is still a failure. H50 used four epochs and H100 two, so this is not a pure horizon comparison. | Used only as an immutable parent. |
| FC-E004 | Control train16 Main | Genuine direct-PPO train-only trajectories improve action response. | dynamic6 passes, but static action-delta and 6.15D/U force-window development gate fail; rear Cl-prime errors dominate. | Good endpoint agreement is insufficient for trustworthy control optimization. | Evaluate a predeclared matched-pair statistic intervention. |
| FC-E005 | Control train16 lift-balanced | Increasing rear-lift channel weight improves force-window fidelity. | dynamic6 passes, but static action-delta and force-window development gate fail. | Channel reweighting did not solve the window mechanics. No claim of superiority over Main is made across different diagnostics. | Retain as controlled negative comparison. |
| FC-E006 | Paired-stat lambda0 | Loss-off with the same parent, sampler, and batch is the within-experiment control for lambda10. | Training completed; only epoch-internal diagnostics exist. Formal validation10/dynamic6/force-window is `NOT_EVALUATED`. | Scientific outcome is unknown; it is not a pure sampling comparison against older differently configured models. | Run the unchanged formal posteval suite. |
| FC-E007 | Paired-stat lambda10 | A train-only matched-pair statistic loss improves force-window mechanics. | Training completed; only epoch-internal diagnostics exist. Formal validation10/dynamic6/force-window is `NOT_EVALUATED`. | Lambda0 versus lambda10 cannot be judged before identical posteval. | Run the unchanged formal posteval suite. |
| FC-E008 | Direct real-CFD PPO training | SB3 PPO through official HydroGym interfaces plus the project OpenFOAM adapter can learn in a genuine online CFD loop. | 2048 transitions, two environments, eight rollout/update rounds completed; each PPO round contains multiple minibatch optimizer steps. | This establishes CFD-only RL closure, not PhysicsNeMo-surrogate success. | Freeze the final policy and use paired CFD for physical claims. |
| FC-E009 | Direct CFD PPO b00 | The frozen final policy improves the same-start physical response. | Final 60D/U: drag reduction 4.2212%, rear Cl-prime ratio 0.93565, mean-bias ratio 0.01993; joint pass. | A valid training-phase CFD-only baseline. | Keep policy and VecNormalize immutable. |
| FC-E010 | Direct CFD PPO b01 | The frozen policy transfers to an unused start time. | Final 60D/U: drag reduction 4.2502%, rear Cl-prime ratio 0.93597, mean-bias ratio 0.03867; joint pass. | b01 was not used for training, but its start is only 18D/U (about three shedding periods) from b00; statistical independence and broad generalization are not established. | Add genuinely separated starts before broader claims. |

## Dataset identities

| Dataset/evidence | SHA-256 | Scope |
|---|---|---|
| dev30 manifest | `5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2` | train20 + validation10 |
| dynamic train8 manifest | `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35` | train-only |
| direct-PPO train16 manifest | `7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b` | train-only; reused already-generated CFD interactions |
| matched-pair manifest | `15bfa7a47e3195afad59884f96fd4e305c8ba0001dd6eb043be2412b2b9ce2b7` | 16 action/zero pairs but four unique initial states |
| fixed normalization | `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1` | train-only normalization shared by listed FNO candidates |
| direct-PPO baseline summary | `b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7` | b00/b02 training baselines |

Frozen10 is materialized and sealed in full40, but is excluded from dev30 development and was not opened here. No frozen metric appears in the CSV.

## Current boundary

At the 2026-10-04 13:40 UTC audit, Spark had no active paired-postevaluation service or GPU compute process. Both paired training directories and completion receipts exist, but neither lambda0 nor lambda10 has verified formal post-evaluation evidence. This ledger therefore records both as training-complete and posteval-pending; it does not convert epoch-internal metrics into admission results.

The strongest verified physical result remains the frozen CFD-only PPO baseline at two start times. It does not show that an FNO surrogate is accurate enough for PPO, that PhysicsNeMo contributed to the physical benefit, or that the policy generalizes statistically.
