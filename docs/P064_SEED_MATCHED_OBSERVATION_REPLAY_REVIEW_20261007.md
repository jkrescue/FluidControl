# P064 two-seed matched-observation replay — accepted diagnostic

Scope: CPU-only deterministic policy-map comparison on the complete saved old-B
b00 observation trajectory. This is not a physical-control result, surrogate
admission, or evidence of drag/lift benefit.

## Execution identity

- Unit: `fluid-control-p064-seed-matched-observation-replay-20261007.service`
- Invocation: `10a5f9fdeee5443d9e53fd39fc4a82e6`
- Exit: 0 (`Result=success`)
- Approval SHA-256: `e7541e62b46c101c92054466c1051ac0b6d356d27ce0d62aaea07c1af89ba534`
- Result SHA-256: `71ffa80731f45a91791a16ebdc918ea3aaed39ebf9a4b7edf74a69571db30d6d`
- Raw-record SHA-256: `c3279d1b1f4a290032b701691bc7c9ca09192a4fc81fa63e1a69ea999c8a986b`
- Worker SHA-256: `3bc56d3fb6d1e9e9591d6e4ad769ad65edd4f616d4637a27200c19ac7f50c59d`
- Reviewed deployment driver SHA-256: `83e08d66aa6c52b4f0164b9ab41eb3a161a52d7b50fa67db078d6b65cf2cbb26`

The unit used CPU2, 4 GiB/no-swap and no visible GPU. Peak cgroup memory was
469,487,616 bytes; minimum observed MemAvailable was 123,167,002,624 bytes;
wall time was 1.660 seconds. No FNO, CFD, training, or action-filter state was
executed.

## Verified result

All 800 old-B observations were used in fixed order. Replayed old-policy raw,
reflected and odd/projected requests reproduced the saved values exactly
(maximum error 0). Root independently recomputed the 800-row odd-component
algebra and RMS values to rounding precision.

For the full 800 rows:

- old-seed odd RMS: `0.4689618222`; new-seed odd RMS: `0.0439747855`;
- old-seed even RMS: `0.4021438833`; new-seed even RMS: `0.5912380307`;
- paired odd mean absolute difference: `0.4524903175`;
- paired even mean absolute difference: `0.2828062527`.

The six predeclared windows show the same qualitative decomposition difference;
for example, the fixed 600-row primary window has old/new odd RMS
`0.4718928604` / `0.0440522548`.

This supports only the narrow statement that PPO seed changed the learned
policy's reflection-even/odd action decomposition on matched observations and
that deployment projection strongly attenuates the new-seed requests. The
observations are endogenous to the old policy, so this does not establish
closed-loop causality, physical lift/drag performance, or a seed distribution.
Lift/drag conclusions remain pending the independent physical terminal review.

The prospective remedy remains unresolved. A naive deterministic projection in
an environment wrapper can break SB3 sampled-action/log-prob consistency; no
custom policy, augmentation, loss, architecture, retraining, or CFD is
authorized by this diagnostic.
