# ADR 0005: Final grade configuration — CORAL with lesion transfer

**Status:** Accepted for the final grade intervention; matrix launch is gated

**Date:** 2026-09-22

**Supersedes:** ADR 0004's pending rollout gates for the grade head. ADR 0004's
implementation details remain historical context; this ADR is the protocol that
governs the final run.

## Decision

Adopt the CORAL ordinal grade head and grade-stratified lesion-centred transfer
augmentation together as one intervention. The intervention is configured by
`nndet/conf/train/gcalf_final.yaml`:

- `model_cfg.head_grade_kwargs.grade_loss_type: coral`;
- `DataLoader3DLesionTransfer`, with `p_paste: 0.5`, source grades GGG4/5,
  `region_mode: dilated`, feathered intensity blending, and ADC excluded from
  intensity jitter. The declared final run has `shuffle_labels: false`; the
  required fold-0 control enables that flag to randomize the pasted GGG4/5
  labels while preserving the transfer geometry;
- `trainer_cfg.grade_class_weight_source: anchor`, with four positive counts
  measured by the Phase 1 post-paste telemetry before launch; and
- `grade_balanced_sampling: false`.

The lesion bank is built for the complete cohort, then filtered to the training
fold at loader construction using PI-CAI patient identifiers. Validation always
receives `lesion_transfer_cfg: null` and the natural class prior.

This is one reported comparison against the locked baseline. The design does
not identify separate effects for CORAL and lesion transfer, and the thesis must
not attribute separate credit to either component.

## Rationale and limits

Transfer changes context, location, pose, and host prostate, but does not create
new lesion appearances. The high-grade appearance support therefore remains
bounded by the approximately 68 real GGG4/GGG5 lesions. A positive result can
localise part of the prior null to exposure/context; it cannot establish
generalisation to unseen high-grade morphology. A negative result strengthens
the data-supply ceiling conclusion.

Transfer replaces grade-balanced sampling rather than stacking another sampling
correction. Anchor class weights are mandatory because lesion-derived weights
would invert the effective emphasis after pasting. GGG4 and GGG5 effective
shares must not exceed GGG2, and GGG2 and GGG3 sensitivity each have a 0.05
maximum allowed drop versus fold-0 baseline.

## Day-0 Phase 1 probe

The required diagnostic is a patient-disjoint multinomial logistic regression
on grade-head features from the existing fold-0 diagnostic checkpoint, evaluated
with macro OvR AUROC and a 2,000-resample patient bootstrap (seed 2026).

**Result at implementation time:** not available in this checkout. No diagnostic
checkpoint or feature export is present under the repository workspace, so no
numeric result is fabricated here. The matrix is not launch-ready until the
remote fold-0 probe records its point estimate and CI in this ADR:

```
Point estimate: pending remote Phase 1 run
95% patient bootstrap CI: pending remote Phase 1 run
Gate: CI lower bound > 0.5 => proceed; otherwise record the null expectation
```

Either result leaves the build unchanged. A lower bound at or below 0.5 is
evidence that the encoder does not encode grade and makes a null combined run
the expected outcome.

## Fold-0 detection safety gate and fallback

After the final intervention completes fold 0, compare its detection metrics
with the existing fold-0 baseline before launching folds 1–4:

- continue only when `lesion_ap` point drop is at most 0.02 and `picai_score`
  point drop is at most 0.03;
- otherwise stop and relaunch the locked baseline configuration (`ce`, transfer
  off), then report grade as the pre-registered null.

This gate protects the primary csPCa detection endpoint because synthetic
lesions are full detection targets. It is an infrastructure/safety decision,
not a grade-result selection rule.

The final grade adjudicator remains macro OvR AUROC with a patient-level,
2,000-resample bootstrap using seed 2026. Weighted F1 is a guardrail only: it
must not drop by more than 0.03 from the fold-0 baseline. The required control
is one fold-0 `full` run with labels shuffled on pasted lesions; an improvement
there voids any grade-supervision claim.

## Consequences

- The disabled loader path remains the base identity hook, so existing arms do
  not receive transfer implicitly.
- The bank builder and anatomy backfill are read-only by default and require
  explicit `--write` after frame validation.
- Matrix launch remains blocked until anatomy containment validation, bank
  construction, Phase 1 telemetry, measured anchor counts, and the Day-0 probe
  are recorded.

## Addendum: transfer-path safety fixes (2026-09-23)

The transfer implementation now keeps the declared feathered, dilated region
as intensity-blend support, but derives the synthetic instance label from the
transformed lesion mask itself. Pasted blend footprints reject any collision
with a real positive instance, and candidate placements require at least 0.95
of lesion voxels inside the gland and at least 0.50 inside the selected target
zone. Placement retries are bounded at eight centre redraws. These changes do
not alter the declared intervention or its locked blend-support values.

The fold-0 detection safety gate is now enforced between fold 0 and folds 1--4
by comparing only `lesion_ap` and `picai_score` with the same-wave baseline.
The transfer worker stream is wired to `exp.seed`, with the process ID removed
from its seed derivation, so changing the experiment seed changes the
augmentation stream reproducibly.

Because collision and anatomy rejection change the effective paste rate, the
Phase 1 post-paste telemetry and `grade_anchor_class_counts` are launch
preconditions that must be re-measured after these fixes. Existing measurements
must not be reused without that re-measurement.

This addendum records, without resolving, a protocol tension found during the
fix: ADR 0003:56-66 downgrades `lesion_ap`/`picai_score` to doubly-secondary
figures because of the known `boxes_to_detection_map` defect, while ADR
0005:76-77 uses those same two numbers as the fold-0 safety-gate criterion.
