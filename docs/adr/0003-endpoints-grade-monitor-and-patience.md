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

   Rationale. `freeze_grade_head` only disables gradients for grade-head
   parameters. The grade loss remains in the summed multitask objective, so its
   gradients still pass through the frozen branch's differentiable operations
   into the shared FPN and encoder. `.eval()` changes neither GroupNorm nor a
   dropout layer here. It therefore does not provide the asserted backbone
   protection and is irreversible for the rest of a run. Leaving it disabled is
   the only supported protocol state; `GradeHeadFreezeCallback` remains tested
   but unused unless its implementation is changed and separately validated.

## Remediation amendment — 2026-09-15

The stopped four-arm grade pilot demonstrated degenerate class non-emission:
every arm omitted at least one GGG class and GGG4 was emitted only three times
across the matched lesions. This is not a reportable low-accuracy result. The
following amendment applies before any fresh fold-0 or matrix run; the stopped
pilot remains diagnostic and is never pooled into CV.

1. `model_best_grade` is a deterministic final snapshot, not an argmax over
   `val_grade_balanced_accuracy`. For a scheduled 50+10 run it is written after
   the ten-epoch SWA callback has completed (`grade_checkpoint: post_swa`); the
   13-epoch remediation diagnostic has `swa_epochs: 0` and records its final
   state (`grade_checkpoint: final`). `model_best` remains the detection-mAP
   selection. The old balanced-accuracy monitor is logged only for diagnosis.
2. The training module writes a `grade_anchor_counts.json` record (the latest
   completed epoch), counting only grade-supervised anchors *after*
   detection-positive subsampling. Fix A
   may use those four recorded counts as mean-one inverse-frequency CE weights
   through `grade_class_weight_source: anchor` and explicit
   `grade_anchor_class_counts`; it must not reuse lesion counts as an anchor
   proxy. The default remains the old lesion prior until this evidence exists.
3. If Fix A fails, `gcalf_grade_remediation` enables Fix B, uniform GGG2--5
   sampling among supervised foreground lesions. It is a fresh 13-epoch,
   fold-0-only diagnostic. Fix C (ordinal loss) is not authorized unless both
   preceding diagnostics fail their gate.

   **2026-09-16 deviation.** Fix A's pilot was not run. Stage 2's measured
   sampled-anchor distribution (`GGG2/3/4/5` = 58156/23248/8991/10725,
   `evidence/stage2-anchor-20260916-v2/train.log`) is within 2--5% of the
   existing lesion-based prior at every class (anchor weights
   `[0.260, 0.650, 1.681, 1.409]` vs. lesion weights
   `[0.256, 0.622, 1.747, 1.376]`) -- Fix A's premise, that the CE sees a
   materially different class prior than lesion counts imply, does not hold
   under measurement. Proceeding directly to Fix B on this evidence, by
   explicit decision, rather than running a pilot predicted to reproduce the
   locked configuration's failure. This is a disclosed protocol deviation,
   not a silent skip.
4. The grade operating point is fixed at `--grade-score-threshold 0.05` before
   the diagnostics. It is a score cutoff, not an F1-optimised threshold; all
   folds use it and report false positives per case at that cutoff.
5. A remediation diagnostic passes only when its grade confusion-matrix
   **columns** emit every GGG2--5 class at least once, its selected epoch has
   `val_grade_balanced_accuracy > 0.33`, mAP changes by no more than 0.02, and
   `val_cls` changes by no more than 0.005 versus its corresponding stopped-pilot
   arm. Failure restores the locked configuration and is reported as a null.
6. The canonical `raw_splitted/labelsTr` metadata must be copied off-instance
   and pass `scripts/audit_grade_metadata.py` before any data-changing recovery:
   exactly 441 supervised lesions, 17 heterogeneous/unsupervised lesions, and
   GGG2/3/4/5 = 253/104/37/47. The M5 record is produced only after the
   single-GPU multiprocessing throughput measurement by
   `scripts/record_m5_budget.py`; without that concrete record, folds 1--4 and
   the matrix remain blocked.

## Remediation correction — 2026-09-16

The original Fix B diagnostic is void as a test of grade-balanced sampling. Its
configuration enabled balanced sampling but retained lesion-prior inverse-frequency
CE weights: sampled anchors were GGG2/3/4/5 = 204/82/30/38 and the weights were
`[0.2555, 0.6356, 1.7373, 1.3716]`. Effective emphasis was
`[0.078, 0.165, 0.412, 0.345]` (5.3x max/min), versus the locked configuration's
`[0.249, 0.242, 0.263, 0.247]` (1.09x). Fix B′ completes the authorized Fix B
diagnostic with sampled-anchor counts `[30118, 26390, 23403, 24923]` and
near-uniform weights `[0.863, 0.985, 1.110, 1.043]`.

For item 5, the authoritative measurements are the whole-volume `run_eval`
confusion-matrix columns and its `grade_balanced_accuracy`. The patch monitor's
column emission is superseded. The former Fix B assessment substituted
whole-volume 0.211 for the declared patch monitor (~0.26--0.29); both fail and
the substitution is disclosed. Recover `val_cls` and the monitor from MLflow for
the diagnostic and corresponding stopped-pilot arm.

Fold 0 has 73 matched lesions (41/18/5/9); one GGG4 lesion moves balanced
accuracy by 0.05 and approximate SE is 0.07. The >0.33 gate is not lowered, but
near-threshold results are inconclusive; pooled five-fold CV is answerable. Fix
C is closed: macro OvR AUROC 0.524 and the epoch 10--13 plateau provide no
ranking signal, while item 3's both-diagnostics-failed condition was never met.

Sec 2's selection wording is superseded by amendment item 1's deterministic
`post_swa` snapshot. Stages 1--3 ran under D9 deviation in
`/workspace/.conda/gcalf`, not pinned `gcalf:m0`. Correct the Stage 2 path to
`evidence/stage2-anchor-20260916-v2.log`.

The pre-Phase-5 real `do_seg=true` consequence is amended: no eligible trained
segmentation prediction exists before the matrix. The first completed matrix arm
must run `seg_metrics.py` on preserved `do_seg=true` predictions before its fold
is counted in analysis; this timing amendment is explicit.

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
