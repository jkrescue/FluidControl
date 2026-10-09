# Diverse-real-start H5 PPO — independent terminal review

Actual unit `fluid-control-exploratory-diverse-h5-ppo-20261006.service`, invocation `3a34c4d621244e4bbacdf1816b5b1374`, independently queried at terminal: PID0, active/exited, Result=success, ExecMainStatus=0. This is surrogate training completion, not physical improvement or admission.

Approval SHA `760e1f9e81494bdd8c3742cd0ce77e168b0df2bf288bd04546412096721f41e2`; executed immutable trainer `aae8c9a4311112439251b695001c7601ffd3d7cd3010937f86bd9a0bfebf3040`, supervisor `de87c86b813c3ec460bed2b74ee63a5d922f1a32fc2beb2b8e789b3d234ed5d0`, adapter `a01c0af63b6f8d1c23ce5a7f8dd3a85d0d8f9410ac68e4a8e07b228606465319`. Independent pre-execution trainer fixtures: 13 PASS0.05s; actual24-packet CPU receipt `f85f84a4b82e0c21eaf011281e0b98b570bfaa083805c604e1fbe04aaa14583b` binds reset data.

Output `artifacts/exploratory_diverse_h5_ppo_training_20261006/payload/`:

- Result `cd5775e4647280b77803de9a5ced6abdf6378cded4f676f935bd9836350c3640`.
- Final policy `8dc8cabf2104654345f270e3fb86edca7752cf4c883112c0a4cbd3a181acea9b`.
- Identity VecNormalize `6988d4d161bc69c8bbd89d477e9320ad9ef264d35c9dee0bbf63954d4cdfce70`.
- All six result-listed artifact byte hashes independently recomputed and matched; no policy/model deserialization by this review.

Actual4096 transitions, 32 PPO epochs, 64 optimizer hooks and816 complete length5 episodes. Each phase logged reset counts `[35,34,34,34,34,34]`. Independently counted transition slot coverage per environment `[174,170,170,170,170,170]`: 204 complete episodes plus4 steps in the final partial episode, not lost transitions. All24 real starts were represented. Trainer records changed policy tensors and unchanged frozen K1 digest `207932d9da5a9e4f256e16997fbfb4b74a31aaf63db0ca3087d8fe9b62892b0c`, guarded in reviewed execution code; this review did not independently reload tensors. Historical high/TF32 load identity followed by explicit highest/no-TF32 inference override is recorded.

Supervisor actual return0/error=null and12GiB/noSwap cgroup. Independently recomputed minimum MemAvailable from172 rows:119470489600 bytes, above22GiB runtime reserve. Trainer wall83.8070s. No CFD ran during training.

Reset-distribution intervention produced actual applied action support−0.75…+0.75; rate-limited fraction88.0615%. Unlike zero-only training, mean-bias and fluctuation reward penalties were active (mean components−0.51679624 and−0.05877138). These are training diagnostics, not evidence of better CFD control. The next separately approved paired124-cycle final-policy trial tests transfer without MPC substitution or online FNO; original10% physical reference remains unchanged and12.4D/U is not80D/U formal admission.
