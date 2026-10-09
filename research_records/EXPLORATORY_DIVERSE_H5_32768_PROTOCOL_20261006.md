# Fixed diverse-real-start H5 PPO budget comparison — preparation only

One factor:4096→32768 transitions from the same fresh seeded PPO initialization, not resumed policy/optimizer. Same24 verified packets, deterministic six-slot phase cycle, frozen officialK1, H5, canonical62 reward, physical69 observation, action constraints, seed20261006, all PPO hyperparameters and final-policy-only saving. Old4096 source/artifacts remain unchanged.

Expected64 rollouts×512 transitions,256 PPO epochs,512 optimizer steps; each environment8192 transitions gives1638 full episodes plus2 partial steps,1639 resets with counts `[274,273,273,273,273,273]`. All observed accounting must be reported; no cherry-picked checkpoint.

Unchanged resources:12GiB/noSwap,1CPU, allocator.06, MemAvailable50GiB startup/22GiB runtime (20reserve),1800s supervisor/worker and1950s outer with existing stop grace. Longer budget does not justify relaxing guards. No execution authorized by the pending spec.

The prior value loss cannot be compared directly across different reset distributions and neither result proves convergence. This experiment tests whether more optimization on the same diverse distribution improves the eventual separately approved physical trial; no promised benefit or changed physical10% criterion. No model/data/CFD generation in this preparation.
