# Spark tandem surrogate-to-control gate (2026-10-02)

The new `scripts/assess_tandem_control_readiness.py` is an **offline candidate screen**, not a claim of physical control. It reads the independent four-case OpenFOAM held-out evaluations with observed and zeroed action inputs. Both reports must match checkpoint epoch, case order, segment stride, action scale and segment counts; each must pass the existing numerical evaluation validator.

The conservative screen requires field and rear-force MAE to beat state/force persistence at 1, 10 and 50 steps; observed-action force MAE must beat zero-action-input MAE at each horizon; and observed-action field MAE must beat zero-action-input field MAE at 50 steps. These are directional tests, not confidence intervals or a proof of causal action benefit. Failing one condition prevents promotion to longer policy search without diagnostic review. Passing merely permits a **short-horizon, multi-seed candidate-policy experiment**; independent OpenFOAM state-feedback control remains mandatory.

Run after held-out observed and zero-action reports exist:

```bash
python3 scripts/assess_tandem_control_readiness.py \
  artifacts/tandem_fno_expanded_spark_20epoch/heldout_evaluation.json \
  artifacts/tandem_fno_expanded_spark_20epoch/heldout_evaluation_zero.json \
  --output artifacts/tandem_fno_expanded_spark_20epoch/control_readiness.json
```

The existing epoch-5 checkpoint fails exactly two checks: one-step rear-force MAE 0.12813 exceeds force persistence 0.09986, and 50-step field MAE 0.12890 exceeds zeroed-action-input 0.12831. The machine-readable result is `docs/results/expanded_fno_5epoch_control_readiness.json`. Do not use the 32-step PPO software smoke as evidence of control benefit.

The legacy `scripts/run_tandem_cfd_feedback.sh` is intentionally disabled on aarch64. Its host virtual environment, full-field VTK conversion and temporary-file deletion require a separately reviewed Spark implementation. A future frozen policy can consume the already validated 32 OpenFOAM wake probes plus rear Cd/Cl and applied omega, without needing per-step VTK fields. It must compare against a matched zero-request-action CFD run from the **same restart**, enforce `|omega|<=5` and `|delta omega|<=0.5` per 0.1 time unit, log requested versus applied action, and report Cd, Cl RMS, actuation and numerical health. Probe-only observations would need their own real-CFD parity test against the surrogate observation definition.
