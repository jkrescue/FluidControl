# Canonical causal-history H2 objective — CPU source preparation

This is an explicit alternative selector mode. It does not change the accepted
instantaneous-H2 implementation or its completed result, and it authorizes no
model/CFD execution.

## One intended change

Everything remains fixed: original failed-long-AR K1 dual FNO, newly observed
current CFD field, H2, the five held-action candidates, bounds, same t=148
restart, paired zero branch and ten 0.1-D/U cycles. Only candidate scoring
changes from the two-point instantaneous expression to the existing canonical
force ledger and canonical cost components.

At t=148, `actual_causal_prehistory` must load exactly 62 raw OpenFOAM force
samples at t=141.9,…,148.0 without interpolation and bind every source file SHA.
For each candidate independently:

1. clone the current real 62-sample history;
2. append predicted force at t+0.1, drop the oldest sample, evaluate the existing
   canonical ledger/cost with `delta_omega=candidate-current`;
3. append predicted force at t+0.2, drop the oldest sample, evaluate the same
   cost with the held action and `delta_omega=0`;
4. average every unmodified canonical component over H2, then sum components.

After real CFD executes the selected first action, discard all predicted local
histories. Advance the persistent history with the actual endpoint force only.
No future truth, predicted-force persistence, weight/threshold search, validation
selection or model change is permitted.

## Interpretation

The hypothesis is that force phase context prevents the instantaneous mean-Cl²
term from repeatedly preferring positive action when that trades away drag.
The matched ten-cycle result must report action sequence, all canonical ledgers
and components, actual total Cd, rear-Cl mean/RMS and zero-paired effects.
Two predicted points may be too diluted within 62 samples, yielding a hold
policy; that is a valid negative result, not permission to tune weights. This
remains exploratory one-D/U evidence and cannot alter original gates or admit a
surrogate/controller/PPO.

Retrospective, non-control evidence recorded before any new execution: applying
this exact score to the ten saved candidate-prediction panels and rolling the
history with the *old trial's* realized CFD endpoints selected the hold
candidate at all ten old states (first 0 instead of 0.05; last 0.45 instead of
0.50). This is off-policy rescoring of the old trajectory, not a new closed-loop
result and not proof that a newly generated trajectory will keep holding. It
supports the predeclared dilution/rate-cost risk; it does not authorize changing
the horizon or weights.
