# Live research dashboard

The dashboard runs on the **primary DGX Spark** from this repository. It reads
the formal tandem-cylinder run and Gate-B audit without changing either. The
Mac is only a browser and SSH forwarding endpoint; the worker Spark is sampled
over the primary node's existing SSH connection.

## Start and view

On the primary node, from the repository root:

```bash
python3 -u scripts/serve_live_research_dashboard.py --port 8766
```

The server binds only to `127.0.0.1`. From the Mac, create an SSH local forward
to the primary node (`127.0.0.1:8766` on both ends), then open
`http://127.0.0.1:8766/` in Chrome. The page polls every five seconds; the
server samples each node every ten seconds. Raw samples are appended to
`artifacts/monitor/live_resource_samples.jsonl` on the primary node.

## Interpretation

- The training chart reads the formal PhysicsNeMo FNO multistep history.
- Gate-B values and decisions come from the immutable evaluation and audit JSON.
- Flow images show the current multistep model if its evaluation figures exist.
  Until then the page explicitly labels the older single-step model images.
- GB10 has unified CPU/GPU memory. `nvidia-smi` reports GPU activity but not
  separate dedicated VRAM usage, so the resource cards show `MemAvailable` and
  the 20 GiB required headroom. The guarded training log additionally records
  minimum CUDA-free and system-available memory during execution.
- Stage-C CEM and current-objective PPO are shown as withheld until Gate B
  passes. Historical PPO experiments for other objectives are not counted as
  current total-drag results.
- A zero GPU reading means no GPU work at that sample time. It does not erase
  completed training, CFD, evaluation or CPU-only analysis in the run history.

This page is operational monitoring, not evidence of CFD closed-loop drag
reduction. Final physical claims require the phase-matched OpenFOAM replay and
lift constraints in `docs/RESEARCH_OBJECTIVE.md`.
