# FNO spatial-error diagnostic: purpose and limits

This is an observational diagnostic plan, not a model change, a new acceptance
threshold, or a claim that an instability mechanism has been established.
The physical problem remains two fixed tandem cylinders, Re=100, L/D=5,
rear-cylinder rotation, total-drag reduction with the existing lift constraints.

## Evidence motivating the check

The fixed b01-plus H100 true/predicted/error image shows small-scale spatial
variations in predicted u, v and p that are absent or much weaker in CFD.
This is a qualitative observation; the image alone does not prove aliasing,
insufficient Fourier modes, an incorrect CFD solution, or the need for a
different architecture. No smoothing or image editing is used to hide it.

H50's observational pressure-reference diagnostic gives only about 0.4% average
improvement in pressure relative L2 after removing the predicted spatial mean.
For the action-driven cases the improvement is about 0.08–0.12%. Constant
pressure-reference drift is therefore not a leading explanation of the observed
error. No pressure projection has been added to the recurrence.

## Fixed diagnostic definition

- Use the already declared b01/b05 minus/zero/plus trajectories and H1, H10,
  H50, H100 snapshots. Do not select a nicer frame after viewing results.
- Restrict spectral analysis to x/D in [17,24], y/D in [5,10], downstream of
  both cylinders. Require a uniform grid and entirely fluid cells. This avoids
  treating the solid-mask jump as generated flow structure.
- Subtract each channel's regional spatial mean and apply the same separable
  Hann window to prediction and CFD. Use orthonormal FFT2 and report energy in
  fixed radial bands [0,.5), [.5,1), [1,2), [2,4), [4,infinity) cycles/D.
- Report true, predicted and error energy, including Parseval consistency.
  Combine u/v only for a velocity statistic; do not add pressure energy to
  velocity energy. This finite-window spatial-frequency diagnostic is not a
  homogeneous-turbulence energy spectrum or a conservation budget.
- Separately compare gradients of the unfiltered fields using actual grid
  spacing and excluding the rectangle's outermost cells. A zero reference
  norm produces an undefined/null relative metric, not an artificial zero.
- These operations act on copied diagnostic arrays only. They never modify
  model inputs, recurrence, force predictions, published field errors or PPO
  rewards. There is no spectral PASS/FAIL threshold in this experiment.

The currently running data-augmentation experiment and the planned lift-weight
comparison retain the official PhysicsNeMo FNO architecture. Spectral findings
will inform a later controlled experiment only if the data support it.

## Primary research context, checked 2026-10-04

[Lanthaler, Stuart and Trautner, *Discretization Error of Fourier Neural
Operators*](https://arxiv.org/abs/2405.02221), submitted 2024 and revised 2025,
separate grid-discretization error from other model errors. This supports
measuring spatial error rather than assuming a larger network resolves it.
It does not identify the cause of this tandem-cylinder model's error.

[Cao et al., *Spectral-Refiner*, ICLR
2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/972eea97af7fe9a7588f5af136edef0c-Paper-Conference.pdf)
study spatiotemporal operators and numerical-method-informed fine-tuning.
Their specialized spectral layer and benchmark setting are not a drop-in
replacement for our official PhysicsNeMo FNO or nonperiodic cylinder problem.

[Li et al., *SGNO*, 2026 preprint](https://arxiv.org/abs/2602.18801v2)
study amplitude, phase and mode-interaction errors in long rollouts, primarily
for periodic evolution PDEs with Fourier-multiplier linear dynamics. This
motivates diagnostics, not importing an unofficial model or claiming that
their performance transfers to our case.
