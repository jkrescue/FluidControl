# FC-P003 fallback: existing dynamic8 matched-pair candidate

This CPU-only audit prepares, but does not authorize, the smallest data-side
intervention if the approved FC-P003 interleaving experiment fails. It pairs
the eight existing train-only PRBS/multisine trajectories at phases
`b00/b02/b04/b06` with the corresponding train20 zero-control trajectory.

All eight pairs share the exact five-file OpenFOAM restart SHA map (`U`, `U_0`,
`p`, `phi`, `phi_0`) within each phase. Their curated HDF time grid, mask and
initial force agree. The two independent Curator paths produce a maximum
state0 difference of `2.3841858e-7`; the audit therefore predeclares
`rtol=0, atol=3e-7`, records every measured maximum, and rejects larger errors.

The candidate adds dynamic histories (PRBS and multisine, sign changes and
ramps) to paired action-versus-zero supervision using only existing train
data. It does not add phases, Reynolds numbers, geometries, validation data or
frozen-test data. It must not be used unless FC-P003 is evaluated and a new
Lead approval binds the candidate manifest. Passing this QC is not evidence
of surrogate or control improvement.
