# FC-P008 fixed-feature force-row candidate

FC-P008 is a project calibration method around the pinned official PhysicsNeMo FNO. It is not a new official PhysicsNeMo model or API. Model construction and checkpoint load/save remain the official pinned APIs; existing Curator/DataPipe lineage remains authoritative. Reading curated train-only HDF files only caches existing CFD features and creates no new CFD data.

The immutable parent is FC-P003C epoch 2 (`f78c2f...697eb4`). The intended reviewed GPU execution will extract one true-state H1 pooled 128-feature row for every existing training transition: base20 16,000, train8 1,600, and train16 2,048, for 19,648 total. Validation, frozen test, and PPO are forbidden.

Source folds are assigned from actual time-zero and initial-state agreement against the four canonical base zero restarts, never from a filename. The canonical times are b00/b02/b04/b06 = 148/106/120/134. Time tolerance is `2e-5`; state max-absolute tolerance is `3e-7`; every HDF SHA is verified first.

The global endpoint weight for a row in family `f` is

`(exposure_f / (720+408+240)) / endpoint_count_f`,

where exposure is base20/train8/train16 = 720/408/240 and endpoint count is 16000/1600/2048. Thus family shares equal the existing regular sampler exposure shares. A fold removes one physical source phase across every family; remaining weights are renormalized only for fitting/standardization. OOF errors retain the original global endpoint weights and are pooled once over all phases. All four normalized force channels have equal selection weight. Ridge uses the fixed alpha set `{0,1e-8,1e-6,1e-4,1e-2,1}`, an unpenalized intercept, and exact ties choose the larger alpha. Each phase/family/channel is reported.

Feature extraction, candidate native replay, formal evaluation, and any later HydroGym policy runtime use the unchanged historical default TF32/high protocol. The highest-FP32 v2 diagnostic cache is not a formal baseline or candidate input. The captured native final-layer pointwise output followed by masked mean must reproduce the model raw force within the unchanged `2e-5` wiring tolerance. Pooled-feature CPU affine predictions are separately compared with native candidate outputs; their discrepancy is reported, not hidden.

After train-only alpha selection, one all-train fit may replace only `decoder_net.final_layer.linear` rows 3:7 and the corresponding four bias entries in a new checkpoint directory. All other tensors and the state rows 0:3 must remain byte-identical. The original parent remains read-only. Before any validation is authorized, a complete train-only native H1 replay must be finite, preserve state outputs, and bind the new official checkpoint. No scientific threshold changes, no PPO launch, and no physical-control claim follow automatically.
