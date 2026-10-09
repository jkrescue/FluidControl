# FC-P017 first-step observation

Result SHA `ebfb80fc9bbc4b701c6450062895a99e1383f0899153a1cf91fb202f966d495d`.
Source approval `5f5f2c0`. Actual invocation
`cfa568f1f4d5412a93628273d93a95a1` now exited, PID0; internal GPU guard exit0.
Full independent terminal audit remains pending at this observation.

| Same six-window mean objective | Total | H1 | Autoregressive |
|---|---:|---:|---:|
| Initial | 0.006246136754 | 0.003651720137 | 0.008840553239 |
| Full actual Adam step | 0.127936633925 | 0.126110440741 | 0.129762819658 |
| Positive 1/64 displacement | 0.006146119762 | 0.003535265729 | 0.008756974377 |
| Negative 1/64 displacement | 0.006402113611 | 0.003823863144 | 0.008980363821 |

Raw-mean-gradient directional dot with actual full displacement is
`-0.009201020181272402`; central directional difference is
`-0.008191803159813077`. Initial repeats and restored evaluation agree exactly;
full displacement reconstructs actual post-step tensors exactly. This supports
finite first-step overshoot under this numerical protocol, not gradient equality,
convergence, generalization or deployment readiness. Zero repeat spread is not a
rigorous TF32 accuracy bound. Formula difference norm `9.85625645141189e-06` is
observational versus actual displacement norm `0.02960694734665126`.

Both H1/AR decrease at the fixed small positive displacement. It is a diagnostic
point, not a selected saved model. No candidate, validation, frozen test or PPO.
Guard minimum observed CUDA free is 24.1615GiB; host available 105.9237GiB.

Next proposal under review: a separately preregistered full-data experiment
matched to P015, changing only AdamW learning rate to 1/64, keeping original
data/order/171 updates/architecture/objective and terminal-only evaluation.
First-step evidence does not establish this rate is optimal or stable for all
updates. No execution approval is implied by this observation.
