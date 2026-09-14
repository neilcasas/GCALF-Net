# ADR 0003: Endpoint Re-designation, Grade Monitoring, and Grade-Head Patience

**Status:** Accepted | **Date:** 2026-09-14 | **Amends:** ADR 0001 A5; ADR 0002 D7-D8

## Context

The stopped fold-0 four-arm pilot established that the detection, box-regression,
and segmentation heads remain healthy: validation Dice rose from 0.47 to 0.60,
mAP from 0.099 to 0.286, and FROC at two false positives per case reached 0.69
at the 10 mm centroid criterion. None had converged at epoch 19/60. The masked
grade head did not exhibit overfitting. Its training cross-entropy fell only from
about 1.31 to 1.11, while validation cross-entropy was about 1.31; uniform
four-class cross-entropy is 1.386. The prior 12.83M-parameter head did memorize
(training CE 0.15--0.17) but generalized no better than the current 1.33M head:
the best validation CE ranges were 1.137--1.166 and 1.248--1.273 respectively.

The constraint is data supply, not label plumbing. Of 1,499 retained cases, 1,075
are benign/ISUP <=1 and have no lesion to grade. The remaining 424 csPCa-positive
cases contain 458 lesion instances, of which 441 are grade-supervised (GGG2 253,
GGG3 104, GGG4 37, GGG5 47); only 17 are heterogeneous and ungraded. PI-CAI has
only 40 ISUP-4 and 52 ISUP-5 patients in the public training set. Recovering the
17 lesions cannot change the rare-grade limitation.

Per-batch validation grade CE is also unsuitable for checkpoint selection. It is
affected by unequal batch mass, repeated anchors at a lesion voxel, stochastic
anchor subsampling, and zero-grade batch dilution. Its epoch-to-epoch variation
exceeds its total 20-epoch change. The corrected metric is therefore a
patch-level, pre-ensembling, pre-TTA monitor only; it is not the whole-volume
Phase 6 reported number.

## Decisions

1. **Superseded 2026-09-14 (Sec 1a).** Grade weighted F1 / balanced accuracy is
   the primary endpoint, matching the thesis's own stated title and hypothesis
   (`DATASCI17-THESIS-MASTERFILE.md`: "...for Gleason Grade Group Classification
   using bpMRI"; the pre-registered hypothesis that LFF+CAF "significantly
   improves lesion-level GGG2-5 classification performance"). Detection (FROC,
   lesion AUROC, case-level AUROC) and lesion-matched segmentation Dice are
   supporting evidence that the shared pipeline trained correctly, not the
   endpoint SOP 1-2 are decided on. Retain ADR 0002 D8's GGG4+5 merged analysis
   and report grade support, per-grade sensitivity, and uncertainty honestly.
   The 4-class GGG2-5 formulation is unchanged; no binary or merged reframing is
   substituted for the primary result.

1a. **Reversal, decided 2026-09-14.** This ADR's original Sec 1 (below,
   preserved for the record) re-designated detection and segmentation as
   primary after the fold-0 arms happened to separate on those two endpoints.
   That reasoning did not weigh the thesis's own declared research question,
   which is GGG classification, not detection or segmentation. On review, the
   original re-designation is reverted; grade is primary again, per the amended
   Sec 1 above.

   Consequence for Phase 1b (the detection-map painting defect,
   `boxes_to_detection_map` inverting checkpoint ordering at any single score
   threshold): it no longer gates a matrix launch. Verified by code trace:
   `gcalf_eval/grade_metrics.py::match_grade_predictions` and
   `gcalf_eval/seg_metrics.py::match_and_score_segmentation` both match directly
   against `pred_boxes`/`pred_scores`/masks and never call
   `boxes_to_detection_map`. The defect is confined to the PI-CAI aggregate
   detection map (`lesion_ap`, `auroc`, `picai_score`), now a doubly-secondary
   figure -- detection is supporting evidence, and nnDetection's internal FROC/
   mAP (computed every epoch, independent of this path) already serves that
   role. Resolve Phase 1b opportunistically from cached matrix predictions;
   report the PI-CAI map figure only if a representation passes the pre-declared
   selection rule (Phase 1b of the working plan) before the thesis write-up.

   **Original Sec 1, superseded, preserved for the record:** "Detection (FROC,
   lesion AUROC, and case-level AUROC over all 1,499 cases) and lesion-matched
   segmentation Dice are the endpoints answering SOP 1--2. Grade weighted F1 is
   secondary. Retain ADR 0002 D8's GGG4+5 merged analysis and report grade
   support, per-grade sensitivity, and uncertainty honestly."
2. Select `model_best_grade` by `val_grade_balanced_accuracy`, a lesion-matched
   macro recall over supervised GGG2--5 matches, with mode `max`. Log
   `val_grade_weighted_f1` and `val_grade_matched_lesions` beside it. Do not use
   weighted F1 for selection.
3. Aggregate `train_grade` and `val_grade` by their grade-supervision weight,
   matching the denominator of weighted cross-entropy. This corrects measurement
   only; it does not change labels, sampling, loss, optimizer, seed, or the
   validation loader.
4. Amend ADR 0001 A5: `EarlyStopping` remains prohibited. A configured
   `GradeHeadFreezeCallback` may freeze only the grade head after N consecutive
   non-improving finite values of `val_grade_balanced_accuracy`; detection and
   segmentation continue to the scheduled 50+10 epochs. The callback uses
   `requires_grad=False` and `.eval()` on the grade branch and never terminates
   the trainer.
5. N is deliberately not inferred from fold-0. The fresh-matrix configuration
   must declare one integer `trainer_cfg.grade_freeze_patience` in a committed
   protocol-lock change before any Phase 5 process starts. It remains unset for
   the Phase 4 diagnostic pilot. This ADR does not authorize a matrix launch
   without that value.
5a. **Protocol lock, decided 2026-09-14: grade-head freezing is DISABLED.**
   `trainer_cfg.grade_freeze_patience: null` is declared explicitly in
   `nndet/conf/train/gcalf_baseline.yaml`, which all four arms inherit, so the
   decision is recorded in the configuration each run resolves from rather than
   being implied by absence. This satisfies Sec 5's requirement that the value be
   committed before Phase 5.

   Rationale. The grade head is data-limited, not under-regularized: training
   cross-entropy reaches only 1.11 against a uniform-prediction value of
   ln(4)=1.386, and the pre-`d9d1159` 12.83M-parameter head memorized to 0.15
   while generalizing *worse* (best validation CE 1.137-1.166 versus 1.248-1.273
   for the current 1.33M head). Freezing a branch that is already near chance has
   little expected effect on the shared backbone. Leaving it disabled keeps the
   training protocol byte-identical to the existing `_c2_` fold-0 runs, removing
   one moving part from a matrix whose primary endpoints are now detection and
   segmentation. `GradeHeadFreezeCallback` remains in the codebase, tested and
   unused, available without an ADR change if later evidence warrants it.

## Statistical Consequences

Per Sec 1a, SOP 3's tests (Shapiro-Wilk then ANOVA/Tukey, or Friedman plus
Bonferroni-Wilcoxon at alpha 0.05/6) consume paired per-fold **grade weighted
F1** for baseline vs. full, as pre-registered before this ADR's original
detection/segmentation re-designation. Report detection (mAP, FROC) and
segmentation (Dice) as supporting per-fold trajectories, not as the SOP 1-2
decision metric.

Fold-0 arm order is not reproducible across the two available builds on
detection/segmentation (between-arm spread approximately 0.025 mAP and 0.03
Dice, within-arm epoch-to-epoch FROC variation substantially larger); grade is
data-limited (GGG4 approximately 30, GGG5 approximately 38 training lesions per
fold). Both risk an underpowered five-fold-level test. Patient-level paired
bootstrap confidence intervals required by ADR 0002 D7 carry the complementary,
patient-level uncertainty and are especially load-bearing here. A predeclared
null or a wide-CI, inconclusive result on GGG2/GGG3 specifically is reportable
and literature-consistent -- PDHD-Net (Wang et al., 2025) reported the same
weakness at that boundary on a richer four-channel private-data input. Endpoints
must not be changed again after the five-fold results are known.

## Consequences

- All five folds, including a fresh fold-0 rerun, use this one measurement and
  selection protocol. The stopped fold-0 results are not pooled with them.
- Existing checkpoint tensors remain compatible; old `model_best_grade` selections
  are invalid because their monitor was noisy. PyTorch Lightning 1.4.2 keys both
  `ModelCheckpoint` states by callback type, so corrected runs must use fresh
  train directories rather than resume an old run.
- Before Phase 5, exercise `gcalf_eval/seg_metrics.py` on real `do_seg=true`
  predictions and pass the four-arm fold-0 diagnostic gate.
- Full runs recompute BatchNorm running statistics after the ten SWA weight
  snapshots. This is one no-backprop training-loader pass, not an eleventh SWA
  optimization epoch. The `swa_epochs: 0` diagnostic pilot registers no SWA
  callback and therefore has no such pass.
