# Current-frame adapter CPU review

Root accepted this project-only CPU bridge after full source/test inspection and an independent 14-test run (0.58 seconds). No official reader, sampler, existing control or model architecture was changed. The adapter is not an official PhysicsNeMo API.

Root found that the initial action implementation divided Python floats before FP32 conversion: 446 of 1,501 checked physical-action values differed from the DataPipe's FP32-first arithmetic by one ULP. The final module casts physical actions to FP32 before dividing by .75. Exact .01/.02 regression assertions now distinguish those orders; older .225 assertions also use the canonical FP32 formula rather than a rounded Python .3 literal.

Accepted module SHA256: `2945d38773100ca05e66d4be268839fad7f2d78fca016ab867fdd5c7b742eaa4`. Original stage SHA `e4b9a2aedfb000e126ffe35492e6484865c7b08dd6b0cb215cd502d85be82dff` is superseded. Plan SHA `29d872feef4b77adb971451568f0ef1763458dd19c45bfb7f75e3efda554c67e` is unchanged. Canonical test integration changes only its import to `fluid_control.online_current_frame`.

Coverage is synthetic CPU normalization, exact real-helper input packing, packet rejection, no packet mutation, and action constraints—not actual VTK/HDF/model/CFD/GPU execution. Input authenticity remains a future orchestrator obligation: expected time/mask/normalization digest must be independently bound to the real completed CFD state. Packing expects an unmodified result of normalization; full mask validation belongs to normalization. K4 observed history, model acceptance, autonomous action selection and closed-loop physical success are not supplied here. No scientific thresholds or resource floors changed.
