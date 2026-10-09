# FC-P013 diagnostic scheduling clarification

Lead decision before formal training: run the already planned six-window train
diagnostics as separate, read-only evaluations of the immutable P009 parent and
the eventual P013 terminal candidate. This is a scheduling change, not removal
of a metric or a relaxation of admission.

Preserve the original six window identities, H1/free-AR H100, physical force MAE,
tail62 mean lift/RMS error and field metrics from the frozen P011 diagnostic
implementation. Both results must be retained before interpreting the training
outcome. The diagnostics cannot select a checkpoint, change training duration,
or authorize PPO. They can run while the independent formal workflow is prepared.

Reason: repeated agent service-capacity failures interrupted integration of
in-training diagnostic callbacks. The full H100 optimizer objective/backward
has already passed the real-data resource probe. Separating read-only evaluation
avoids coupling that work to optional callback state restoration, while keeping
the same scientific comparison and all acceptance requirements.

Training remains one fixed 1368-window pass, terminal-only official saving,
strict frozen-flow invariance, full finite-gradient checks and official dual
fresh reload. Its result explicitly reports the separate diagnostics as pending,
not passed or complete. Original formal validation and real-CFD control criteria
remain mandatory. No inference of improved accuracy follows training completion.
