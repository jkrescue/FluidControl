# Exploratory canonical-history H5 preparation

This prepares one horizon-only comparison; it authorizes no model, CFD, GPU,
PPO, or policy execution.  The completed H2 trial remains a reviewed negative
result: ten HOLD actions and zero paired benefit.

The sole proposed change is planning horizon 2 to 5.  Keep the original failed
long-AR K1 dual FNO, current real-CFD field reset every cycle, the same five
held-action candidates, action/rate bounds, exact 62-sample actual causal force
history, canonical ledger and component weights, t=148 restart, paired zero
branch, ten 0.1-D/U cycles, and execute-only-first-action semantics.  Predicted
histories stay candidate-local; persistent history advances only with measured
CFD endpoints.

The canonical ledger accepts this change without a new metric: each stage still
sees exactly 62 force rows after append/drop.  At H5, the last local window has
57 measured rows and five predicted rows.  The same initial rate cost occurs at
stage one and is averaged over five stages rather than two; this is inherent in
the predeclared horizon comparison, not a weight change.  A nonzero choice does
not prove that dilution was the unique H2 cause, and HOLD or nonbeneficial CFD
response is a valid negative result.

The H2 code path delegates directly to the reviewed H2 functions.  Regression
tests require identical H2 actions, score dictionaries, and rollout dispatch.
The explicit H5 path performs 25 flow and 25 aerodynamic forwards per decision,
versus ten plus ten at H2.  The completed **causal-history H2** unit took 182
seconds, with ten per-decision inference timings totaling about 40 seconds.  A
linear estimate is roughly 240 seconds
for H5; allow 300--450 seconds observationally under the unchanged 900-second
outer deadline.  Peak model/state memory should remain similar because rolls
stream one candidate and retain only 5x5x4 forces and five state maxima.  This
is a resource estimate, not a runtime guarantee or real-time claim.

Before any execution, a separately reviewed runnable profile must bind the new
mode/status/source hashes and preserve all previous resource, solver, restart,
force-grid, provenance, cleanup, and non-admission checks.  No H10 or horizon
sweep is prepared.
