# True-state paired-force backward technical probe

This implementation prepares one bounded technical probe. It is not a new
candidate, training run, validation result, or control-success claim. GPU
execution remains disabled until the independent code review and Lead GO.

The fixed input is the first train-only dynamic8 pair, `b00:multisine`, at
start 0 and H100, paired with its same-restart zero branch. The launcher pins
the action/zero HDF, pair manifest, normalization, Main-e2 model/state,
resolved inherited configuration, official PhysicsNeMo image, adapter,
helper, and execution script hashes. Only train leaves are mounted; no
validation or frozen-test directory is visible inside the container.

For each endpoint k, the project helper constructs the model input from the
recorded CFD state at k, mask, and recorded omega(k), omega(k+1). The official
PhysicsNeMo FNO returns normalized four-force predictions. A 10-step chunk
loss is multiplied by `chunk_length / 100` before `backward()`, so ten chunks
accumulate the exact H100 temporal mean. The fixed force order is front Cd,
front Cl, rear Cd, rear Cl and weights are `[1,1,4,1]/7`.

Before H100, the same real pair runs a fixed T=20 check: one unchunked
backward is compared with two chunk-10 backwards. Predeclared tolerances are
loss rtol `2e-5`, atol `1e-7`, and gradient rtol `3e-4`, atol `3e-6`. These
values must not be changed after looking at a GPU result. The probe then
clears gradients and runs H100.

No optimizer is constructed, no optimizer step occurs, and no checkpoint or
candidate weight is written. Parameters and buffers are hashed before and
after. The output records finite normalized loss, per-channel normalized MSE,
finite/nonzero gradient norms, runtime, peak allocated/reserved CUDA memory,
and continuously sampled physical MemAvailable. The outer Spark guard kills
the container if physical MemAvailable falls below 20 GiB. The container is
network-disabled and read-only except for its dedicated results mount.

Passing proves only that this isolated paired term has a finite gradient and
fits under this chunked one-pair budget. It does not establish memory safety
for the mixed regular+paired trainer, model accuracy, action-response
fidelity, or closed-loop benefit.
