# Short-history force prediction: conditional next-stage review

Status: read-only scientific/official-interface review while P025 is running.
Not an implementation, execution approval, proven root cause or new result.

## Evidence and hypothesis

P018 complete development assessment remains failed; P023/P024 show limited
benefit and direction-dependent tradeoffs from current-force input coefficients.
P025 tests one statistical-loss intervention and has not yet ended. If unsupported,
the declared isolated-block/statistical-loss branch should end, not expand to
another weight/scale sweep. A separate representation question is whether the
cropped current flow state plus current/next action sufficiently identifies force
dynamics. Short historical states may help; that is a hypothesis, not a diagnosis.

## Official and research grounding (checked 2026-10-05)

- [PhysicsNeMo FNO API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/fnos.html)
  exposes in_channels and regular-grid2D inputs. Stacking causal history in input
  channels can retain the same official FNO family. This does not imply that
  PhysicsNeMo provides our specific history/action adapter or a tandem benchmark.
- [Official transient Navier–Stokes RNN example](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/navier_stokes_rnn/README.html)
  explicitly supports initial-state or multiple-input-time-step forecasting. Its
  ConvGRU/ResNet example and periodic vorticity dataset are not our tandem case;
  it is a temporal-modeling precedent, not an instruction to replace our model.
- [ICLR2025: On the Benefits of Memory for Modeling Time-Dependent PDEs](https://proceedings.iclr.cc/paper_files/paper/2025/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html)
  studies explicit memory and reports benefits under low-resolution/noisy PDE
  observations. The paper motivates testing temporal information but does not
  establish benefit in this Re100 case. We are not proposing to invent or claim
  an official MemNO implementation.

## Conditional candidate design, not approved

Preserve the current frozen flow predictor, real44train trajectories, geometry,
normalization and complete physical/evaluation criteria. Compare matched one-frame
and fixed four-frame force-FNO inputs using the same official architecture size,
budget and initial inherited current-state weights. One possible18-channel layout
is four3-channel flow states + one mask + five historical/current/next action
values. This is a proposed layout, subject to actual dataset/action timing review.
Existing-current columns and coordinate columns must be mapped explicitly; extra
history columns start zero. It is not a larger-mode or different-network sweep.

Before deciding: verify true pre-window field/action availability, timestamp
causality, and cold-start treatment. Padding a missing history is initialization,
not a new CFD observation; it must be declared and evaluated separately from
warm history. Do not use future truth during autoregression or claim causal
force sidecars already cover all44trajectories. Reuse existing data first;
new CFD should follow demonstrated coverage needs, not accompany the model change.

A separately preregistered matched training experiment would still require the
original complete development assessment, newly compatible controller training
and real-CFD closed-loop validation. Neither research citations nor improved
training-panel errors establish the full project goal.
