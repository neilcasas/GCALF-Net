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

1. Detection (FROC, lesion AUROC, and case-level AUROC over all 1,499 cases) and
   lesion-matched segmentation Dice are the endpoints answering SOP 1--2. Grade
   weighted F1 is secondary. Retain ADR 0002 D8's GGG4+5 merged analysis and
   report grade support, per-grade sensitivity, and uncertainty honestly.
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

Fold-0 arm order is not reproducible across the two available builds. The
between-arm spread is approximately 0.025 mAP and 0.03 Dice, while within-arm
epoch-to-epoch FROC variation is substantially larger. Five fold-level values per
arm may be unable to resolve such effects under the predeclared Shapiro-Wilk then
ANOVA/Tukey or Friedman plus Bonferroni-Wilcoxon procedure (alpha 0.05/6). Record
the exact new-primary metric consumed by each SOP 3 test. Patient-level paired
bootstrap confidence intervals required by ADR 0002 D7 carry the complementary,
patient-level uncertainty. A predeclared null result is reportable; endpoints
must not be changed after the five-fold results are known.

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
