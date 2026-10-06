# Reflection-projected 32768 PPO b01 replication protocol

Status: **source and CPU-test preparation only; no b01 CFD execution is approved**.

## Purpose and fixed contrast

This is a conditional second-initial-phase replication of FC-E058. It is eligible
for separate execution review only if the actual FC-E058 b00 projected run
completes and its predeclared primary window meets all unchanged physical
criteria. It does not select b01 because of a favorable result: b01/restart130
was predeclared as phase bin1 and validation split before any controlled outcome.

The controller is the same frozen final32768 PPO and identity VecNormalize. The
only wrapper remains `0.5*(pi(o)-pi(R(o)))`, followed by exactly one existing
amplitude/rate filter. Policy, reflection, observation, solver, action limits,
zero branch, numerical metrics, and thresholds are unchanged.

## Exact b01 source and clock

- Real source: `cfd/tandem_cylinders/cases/tandem_backward_dt005`, restart130.
- Predeclared phase manifest SHA:
  `6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603`.
- Source U/p SHA:
  `ac412e9e3de151253dd3006a70ab195ef8f49f9c81edeeef28086b808fe9d230` /
  `1ddc27110bacd814e31b3425e3550927ba7fee3f57112e62a4cb2a18288aa52c`.
- Source phase `0.6991961542646722 rad`; b00/restart148 phase
  `0.07110264316075172 rad`. The difference is a limit-cycle phase contrast,
  not a statistically independent physical sample.
- Initial rear-cylinder omega is zero. Run 800 decisions at 0.1 D/U from130
  through210 with solver `dt=0.005`.

All mesh files, transport/turbulence properties, and solver schemes/solution
files have byte-identical SHA to b00. Existing matched evidence is
`cfd/tandem_cylinders/cases/matched_start_acquisition_validation_b01_zero`;
its front/rear/probe streams span130.005 through210. These historical streams
prove source availability and are not substituted for a new paired execution.

## Predeclared relative windows

Report all six, with the same durations and open/inclusive convention as b00:

- early12.4 `(130,142.4]`;
- early first6.2 `(130,136.2]`;
- early trailing6.2 `(136.2,142.4]`;
- primary final60 `(150,210]`, 12,000 force samples;
- historical companion `[150,210]`, 12,001 force samples;
- full80 `(130,210]`.

The original 2% drag, 1.05 rear-Cl fluctuation RMS ratio, and 0.10 mean-bias
ratio remain unchanged; 0.20 remains sensitivity only. Cross-phase deltas are
descriptive. b01 has already been used for historical validation and is neither
a fresh final-test split nor evidence of broad phase generalization.

## Identity, resources, and authorization boundary

A future approval must bind the actual successful FC-E058 result SHA, exact
restart130/constant/system tree, final policy/VecNormalize/training result,
source closure, image, exclusive output, and the same resources: 8 GiB CPU
controller, two 8 GiB/no-swap/two-CPU solvers, MemAvailable50 GiB startup and
22 GiB runtime, 20 GiB reserve, 3600 s inner/3750 s outer/120 s cleanup.

This document authorizes no CFD, retry, model load, training, PPO selection,
threshold change, or admission. A favorable b00 result motivates the fixed b01
replication; it does not predetermine its outcome.
