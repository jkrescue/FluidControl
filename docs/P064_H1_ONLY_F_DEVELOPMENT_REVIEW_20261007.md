# FC-E099 — F fixed development review: H1 improves, retention fails

Independent saved-array arithmetic ACCEPT; candidate promotion rejected under the original rule. Lead retains B and the verified controllers. F improves pooled H1 rearCl and totalCd MAE, but fixed-six AR retention fails and pooled H5 worsens. No threshold relaxation, continuation, new PPO or CFD follows automatically. This does not reverse the completed800-step real-CFD engineering reproduction or complete the broader prediction goal.

## Actual execution and identities

- Unit `fluid-control-p064-h1-only-development-h1-h5-20261007.service`, invocation `0f75311df5d248fab0dd12e6ce76c2ed`, PID0/exited/exit0 independently observed before reading result.
- Approval `docs/P064_H1_ONLY_F_DEVELOPMENT_APPROVAL_20261007.json`: `3c048003d6c8fd927a9d9829242a28e38c3afc83a49810d9e156a5b8dc7a5f8c`.
- Output `artifacts/p064_h1_only_development_h1_h5_20261007`; result `642445fc3ad17491d3898c1bd3f49f34062b277302896796b67090a473482cee`; supervisor `14c931d02363a1e540d4ec17f353e9d32df693cedc94c04db797523213a61e39`.
- F manifest `cabd2794ba187e8f303980543fc32f1386ec26da97c02819ba087b583692cf3b`; training review [E098](P064_H1_ONLY_F_TERMINAL_REVIEW_20261007.md), SHA `e40fc80981b5af2457f5e9e1091d62e317219daf5d9f0a02a4121f5b0a6a1413`.
- Existing B comparison result `artifacts/p064_arm_b_development_h1_h5_20261006/result.json`, SHA `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665`; B was not rerun.

Independent audit `audit_f_dev_arrays.py` SHA `656eef2a4cbf942c6b26340123352732619f25559696089b1785dadff266f623` mechanically adapts the reviewed E arithmetic only by F labels. CPU audit invocation `60dc8716943747eb8003f0b7a0bf9ca6` exited0 under2GiB/noSwap/CPU1/120s, CUDA hidden.449 source/192 runtime/10 input hashes and16 NPZ hashes pass; all80 endpoints' four-force MAE, total-Cd MAE, hold-current-force persistence, per-channel field SSE/reference sums, per-origin/per-phase/pooled summaries recompute. Maximum absolute numeric roundoff2.6135239750146866e-8 occurs in large field sums, within declared tolerance.

Fixed b01/b03 starts0,100,…700 and all H1–H5 retained. Truth fields, initial fields, mask/grid, rounded saved times, actual current/next actions, truth forces, and predicted flow are exact against B. Time-to-selection error is bounded by1e-5 from existing VTK rounding. Highest/noTF32 after load, zero optimizer steps; no model/HDF reread or forward by the audit. Minimum Available120117587968 bytes,23 memory observations,11.0113s; supervisor return0/errornull. This is an already-open development panel, not unseen/generalization evidence.

## Force errors (MAE, not percentage force error)

| Panel | H | B rearCl | F rearCl | B totalCd | F totalCd |
|---|---:|---:|---:|---:|---:|
| pooled |1|.138998316601|.136573601048|.038065373898|.033974312246|
| pooled |2|.131030313205|.131668329239|.028274118900|.025661773980|
| pooled |3|.132373675704|.135255103465|.025319509208|.023066688329|
| pooled |4|.148796733469|.150898524094|.027975853533|.028224274516|
| pooled |5|.165744980914|.170506666473|.030762493610|.031422402710|
| b01 |1|.155280457810|.147753668949|.044178001583|.040205702186|
| b01 |2|.136143652722|.133409876376|.030229948461|.024893313646|
| b01 |3|.133854428306|.135969684459|.021978601813|.017241388559|
| b01 |4|.157143987715|.159607004374|.026937693357|.025945432484|
| b01 |5|.180411564419|.186979007092|.033760644495|.032872281969|
| b03 |1|.122716175392|.125393533148|.031952746212|.027742922306|
| b03 |2|.125916973688|.129926782101|.026318289340|.026430234313|
| b03 |3|.130892923102|.134540522471|.028660416603|.028891988099|
| b03 |4|.140449479222|.142190043814|.029014013708|.030503116548|
| b03 |5|.151078397408|.154034325853|.027764342725|.029972523451|

Pooled H1 improves both primary metrics, but b03 H1 rearCl worsens. Pooled H1 persistence remains better than F: rearCl .0903335185722, totalCd .0203182175756. At H5 F is better than persistence .439982240088/.101057592779, but worse than B on both pooled force metrics. No statistical-significance claim is made.

Flow is frozen and arrays are exactly B, not a field improvement: pooled velocity relative L2 H1 .0103161044193/H5 .0430400805913; pressure relative L2 .0320416849890/.137005729854. Mixed u/v/p MAE is not interpreted as a homogeneous physical quantity.

## Fixed-six retention and decision

Independently recomputed means of saved six rows (indices160,816,923,975,1077,1233), with identities/history/flow hashes exact against B:

- Original H1 normalized objective: B .003976855262105043 → F .0036982664023526013, improves.
- Original AR normalized objective: B .008946200483478606 → F .00955009322691088, worsens.

These are original high/TF32 saved training-panel diagnostics, not physical force MAE or new inference. F's actual backward objective is H1 only; diagnostic total remains equal H1/AR. Saved scalar loss correctness is not independently established by a new forward. The predeclared joint H1 improvement criterion is met pooled, but the required H1 **and AR** non-regression is not; Lead rejects replacement of B. H5 degradation and phase differences reinforce the limitation without changing the rule. The hypothesis that removing AR supervision can help immediate controlled-force prediction receives limited descriptive support, not a causal/general conclusion or full acceptance.

Basic closed-loop delivery remains available via [the guarded canonical guide](CANONICAL_CLOSED_LOOP_QUICKSTART.md), with [800-step actual reproduction](CANONICAL_B01_REPRODUCTION_TERMINAL_REVIEW_20261007.md) passing all six original physical windows. Overall surrogate precision remains unfinished; historical failures and original10% physical bias standard remain intact. No training or CFD is running from these completed F jobs.
