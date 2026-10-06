# FC-E056 — fixed24 deterministic H5 policy comparison

Actual one diagnostic unit `fluid-control-diverse-policy-h5-comparison-20261006.service`, invocation `f4ba411c1bcf4cae9ebd178ad0c30a1f`, completed PID0/active-exited/Resultsuccess/ExecMainStatus0. No optimization or CFD, no change to concurrently running FC-E055. The inherited `protocol` describes training provenance; actual `evaluation_protocol` is24starts×5steps×2policies, zero optimizer steps and240s bound, not32768-step training.

Approval `a9c3eaf77f9dc2a444459e2a17da3dc8ea556ce0eb0d952e09cae1880e2fd08f`; immutable worker `630bc478ed17475539b21e206242598d7bbaa044b3acaf00829d95eb1424008c`, supervisor `d90b77374ea14ee4bcb108936401182e85168c8079214ff74129b906d775cd2c`. Final persisted metadata/source/runtime/policy hashes checked before execution; five syntheticCPUtests passed. Source captured in Git after execution, not asserted launchHEAD.

Result `artifacts/diverse_policy_h5_comparison_20261006/payload/result.json` SHA `3f8c6f7e5b03877601a3b25b26409e9d6f943fcbaec62600fa51343995922b9f`. Fixed final4096policy8dc8cabf…acea9b versus32768policy5ab92ebe…06d4a, same verified24real train reset packets, frozenK1/highest-noTF32, fresh deterministic phase cycles. No heldout cases or policy selection. All120steps perpolicy retain actions/forces/reward components; forces are surrogate responses, not actual-CFD accuracy measurements.

| Equal-case metric |4096policy|32768policy|
|---|---:|---:|
|Mean H5 return|−3.6902918374361167|−3.6875228003375486|
|Drag-gate component sum|−0.8006100965861433|−0.7885308339980662|
|Rear-lift fluctuation component sum|−0.2954832418696319|−0.29500373646751105|
|Rear-lift mean-bias component sum|−2.5893267563403075|−2.597430037011634|
|Actuation component sum|−0.0008254722793664734|−0.0018842206018089428|
|Rate component sum|−0.0032961772344497348|−0.004007308311417252|

Mean paired return delta is+0.0027690370985678316;7starts improve,10worsen,7exactlyequal. Phase00/02/04 macroreturns worsen; phase06 improves. The small net gain trades better drag penalty against worse bias/actuation/rate penalties; it does not justify a blanket budget-increase claim or establish convergence. H5 response versus62-sample reward history and incomplete history in69observation remain hypotheses, not proven causes. GPU deterministic evaluation is distinct from CPU deployed trajectories; physical conclusions await actualCFD.

Independent reviewer rehashed all47bound source and192runtime files and recomputed all24×5 returns, six components, matched identities andpaired deltas exactly from savedJSON. No independent model reload was performed. Policy/FNO unchanged digest checks were executed inside the approved worker; reviewer verified their recorded flags and source, not live tensors.

Supervisor return0/errornull; actual12GiB/noSwap.45memory samples minimumAvailable120472039424bytes; peak allocated GPU755589120bytes (<6GiB). Supervisor elapsed22.019s includes imports/load; worker7.069s starts afterdata/model setup. Outer260s/1CPU and runtimeAvailable22GiB reserve stayed intact. CUDAfree was not an admission floor. Original physical10%criterion and FC-E055 protocol unchanged; engineering diagnostic only, no scientific admission.
