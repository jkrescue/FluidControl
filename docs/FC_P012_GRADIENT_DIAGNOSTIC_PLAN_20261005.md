# FC-P012: train-only gradient decomposition

Lead approval: implementation and CPU tests only. GPU execution requires a separate
approval binding the reviewed source hashes. This is not a new training candidate.

## Evidence and question

FC-P011 A and B both failed the unchanged complete development evaluation. B
improved some field diagnostics but did not repair rotating lift-window accuracy;
it also failed validation action-response Cd error. Every B optimizer update was
clipped, versus none in A. These facts do not establish a cause: scalar loss size
does not determine gradient size, and clipping scales alone do not establish an
AdamW optimization failure.

Question: within the existing shared decoder, do the field and force objectives
have strongly unequal or conflicting gradients on the same causal rollout?

## Fixed protocol

- Models: exact P009 parent and exact FC-P011 B terminal, evaluated separately.
- P009 model/state SHA: dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31 /
  4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e.
- B model/state SHA: 5102e83e00276994ad85b88c59721d52915400a037fc29f321d593fa69b800d8 /
  c3c8c92ea792ed7332da75af269143b10ea0d0541b84273b7424e2922d9ae6b0.
- Reuse the immutable FC-P011 trainer's exact six diagnostic windows, data
  identities, normalization, official dataset/model construction and H100 rollout.
  Parent trainer SHA: 9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5.
- Batch one; six windows times two models; no sampling search. Teacher forcing
  zero. Same default TF32/high arithmetic. No altered architecture or API.
- Parameter scope: existing decoder hidden layer plus final rear-Cl row/bias.
  Each gradient component must mask the other final rows exactly as FC-P011 did.
- Compute gradients of field loss, 0.2 times balanced force loss, and their sum.
  No optimizer, clipping operation, parameter update, checkpoint selection or save.
- Report component losses, norms and cosine separately for hidden parameters,
  rear-Cl row/bias, and the complete allowed parameter vector. Zero-norm cosine
  is undefined/null, not zero. Use float64 only to summarize the existing gradients.
- Report the discrepancy between direct total gradient and component sum as a
  numerical observation; do not silently change TF32 or require byte equality.
- Check model tensors before/after exactly; publish all twelve rows, finite-state
  checks, identities and hashes. Never read validation/frozen/PPO data.

## Interpretation decided before execution

Use the five nonzero-action-history train windows separately from the zero case.
Field/weighted-force norm ratio above 10 in at least four of these five windows
is a diagnostic signal of scale imbalance, not proof of a remedy. Cosine below
-0.2 in at least three is a signal of conflicting local directions, not proof of
global task incompatibility. Report both models independently and every window;
do not hide mixed outcomes behind averages. These are diagnostic decision rules,
not relaxed scientific admission gates.

If supported, propose one controlled loss-balancing intervention, subject to a
new approval and the unchanged complete evaluation. If unsupported, do not blame
clipping or keep scanning weights; revisit state representation/data observability.
No automatic training or PPO follows this diagnostic.

## Resources and ownership

Compute agent: immutable implementation and launch preparation. Lead: review and
execution approval. Evaluation agent: independent interpretation and ledger.
Main Spark only, official pinned PhysicsNeMo image; narrow train-only mounts,
allocator fraction at most 0.45, unified MemAvailable at least 20 GiB, proposed
wall cap 15 minutes for twelve windows. Preserve all failures and outputs.
