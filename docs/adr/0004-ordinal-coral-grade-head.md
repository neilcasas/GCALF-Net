# ADR 0004: Ordinal CORAL Grade Head

**Status:** Accepted for implementation; matrix authorization pending | **Date:** 2026-09-22 | **Amends:** ADR 0003 remediation amendment

## Context

The grade endpoint is GGG2--5, an ordinal label space, but the current head is
a four-logit softmax classifier. The clean Fix B′ measurement is null:
macro one-vs-rest AUROC 0.5321, patient-bootstrap 95% CI [0.4330, 0.6323],
and quadratic-weighted kappa 0.1278. GGG5 AUROC is 0.450. Fix A was refuted
by measured anchor/lesion-prior agreement; the original Fix B was confounded;
Fix B′ failed cleanly. None of these changes altered the hypothesis space of
the grade classifier.

ADR 0003's remediation amendment closed Fix C unless both preceding
diagnostics failed. That condition was written before Fix B′ landed and is now
formally reopened because this is a specification correction for an ordinal
endpoint, not another class-reweighting or sampling intervention. No official
matrix result exists yet, and the matrix has not launched.

## Decision

Implement a CORAL grade head behind `grade_loss_type`:

- `ce` remains the default and preserves the existing four-logit head and
  weighted cross-entropy behavior exactly.
- `coral` uses one shared bias-free feature projection and three learned
  thresholds, trained with the three cumulative binary targets
  `1[y > k]`.
- CORAL probabilities are converted to four grade probabilities only through
  the grade head. Negative adjacent differences are clamped and renormalized,
  with a cumulative counter logged whenever clamping occurs.
- The existing lesion class weights
  `[0.256, 0.622, 1.747, 1.376]` remain unchanged. Their selected-class value
  weights each sample, and the loss divides by the same weight sum used by CE.

The implementation is additive: labels, grade mappings, endpoints, operating
threshold, optimizer settings, patch/batch settings, and checkpoint selection
are unchanged. The matrix lock currently requires
`model_cfg.head_grade_kwargs.grade_loss_type: ce`; this prevents an unrecorded
mode change while the gates are pending.

## Pre-declared gates

### Phase 1: linear probe

Train the declared 13-epoch fold-0 diagnostic, extract the grade-head input
features through the existing matcher path, and fit patient-disjoint
multinomial logistic regression. Report macro OvR AUROC with the specified
patient-level bootstrap (2,000 resamples, seed 2026).

The result is **pending** in this implementation amendment. The rule is fixed:

- CI lower bound `> 0.5`: proceed to the CORAL implementation/refit phase.
- CI lower bound `<= 0.5`: stop the CORAL path, launch the matrix locked, and
  report grade as the pre-registered null.

### Phase 3: detector-frozen refit

Compare CE and CORAL on the same Phase 1 checkpoint and patient split. Report
weighted F1, QWK, macro OvR AUROC, balanced accuracy, MAE, adjacent accuracy,
per-class sensitivity, and the confusion matrix. GGG2 sensitivity must not
fall by more than 0.05. The result and the selected matrix mode are **pending**
and must be recorded here before the matrix starts. Fold 0 does not adjudicate
the five-fold endpoint.

## Explicitly unchanged

The primary endpoint remains lesion-level grade weighted F1. Post-SWA
checkpoint selection, `grade_score_threshold: 0.05`, class weights,
patch/batch/LR settings, detection/regression/segmentation heads, and GGG2--5
label semantics remain unchanged. CORAL, CORN, EMD, and soft-label variants
are not substituted for one another.

The 2026-09-11 four-arm numbers and all prior diagnostic runs remain
non-poolable. They are model-last or otherwise diagnostic measurements and are
not five-fold CV evidence.

## Consequences

The implementation reduces the grade head's free final-layer parameters from
`4*C + 4` to `C + 3`, while retaining the existing feature extractor. It gives
the ordinal endpoint an inductive bias that the previous reweighting and
sampling fixes did not provide. The data-supply limit remains binding: a
successful implementation does not guarantee a positive result.

No cloud provisioning, metadata audit, Phase 1 probe, Phase 3 refit, or GPU
smoke run is claimed by this ADR. Those require the absent prepared data and
checkpoints and must be recorded with their actual outputs before matrix
authorization.
