# E terminal audit follow-up: full-path cache regression

The consumed [terminal report](P064_RESPONSE_AUX_E_TERMINAL_REVIEW_20261007.md) SHA `ac4c4ede971b7db0a475fed25956da63ebda55d04284498f627f50a5f066ba61` remains unchanged. Its R2 audit passed. Subsequently the cache guard was tightened from the unique output-directory name to the full absolute candidate output path, with a regression covering the same JSON basename and the same directory name under another absolute root.

Seven CPU tests passed. Actual R3 audit unit `fluid-control-p064-response-aux-terminal-audit-r3-20261007.service`, invocation `b6fa603cc6b44397b4b7083902ecad87`,8GiB/noSwap/CPU1/120s/CUDA-hidden, exited0 in2.048s; all original and auxiliary checks match R2. No candidate/model inference or training was repeated. Executed R3 wrapper SHA `547431a8e56bf0742753cbef420af8f4b64af30d5d71ad011396340b3fd49163`; canonical wrapper changes only the historical checker source path to its same-byte sibling for portability. The original checker is archived unchanged with SHA `8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587`.

Scalar auxiliary losses remain producer records, not independently reconstructed prediction-array losses. This engineering check does not substitute for prospective fixed-development evaluation.
