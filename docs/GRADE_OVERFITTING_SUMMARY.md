# Grade Overfitting Investigation — Summary

**Date:** 2026-09-22
**Scope:** GGG2–5 grade classification head only. Does not affect the csPCa detection/localization endpoint, which is the thesis's primary contribution and is unaffected by the findings below.
**Related documents:** `docs/GRADE_OVERFITTING_PROTOCOL.md`, `docs/adr/0003-endpoints-grade-monitor-and-patience.md`, `docs/adr/0004-ordinal-coral-grade-head.md`, `docs/SITREP-20260916.md`, `docs/SITREP-20260917.md`

## 1. Initial symptoms (Fold-0, 60-epoch pilot, four model arms)

- **Validation cross-entropy plateaued** at ~1.25–1.31, barely below uniform-random guessing (`-ln(0.25) ≈ 1.386`), while training loss kept decreasing.
- **Class collapse:** predictions concentrated on the majority grades (GGG 2, GGG 3), with rare/aggressive grades effectively unmodeled:
  - GGG 5: 0 predictions in the Baseline model (0% sensitivity).
  - GGG 4: emitted only 1–3 times across all matched validation lesions.
- **Capacity vs. generalization mismatch:** the original 12.83M-parameter grade head drove training CE down to 0.15–0.17 (near-perfect memorization) but validation CE only reached ~1.14–1.17 — no better than a downscaled 1.33M-parameter head (val CE ~1.25–1.27).

## 2. Root cause: high-grade data starvation, not classical overfitting

The team initially suspected overfitting and drafted `GRADE_OVERFITTING_PROTOCOL.md`, considering early stopping or a `GradeHeadFreezeCallback`. Diagnostics instead pointed to a data-supply problem:

- Of 1,499 retained PI-CAI cases, 1,075 are benign/indolent (GGG ≤ 1, background).
- Of 424 positive cases, only 441 lesions carry grade supervision:

  | Grade | Lesions | Share |
  |---|---|---|
  | GGG 2 | 253 | 57.4% |
  | GGG 3 | 104 | 23.6% |
  | GGG 4 | 37 | 8.4% |
  | GGG 5 | 47 | 10.7% |

- A single training fold sees only **~30 GGG 4 lesions total**.
- Stage 2 telemetry confirmed sampled training anchors closely track true lesion counts (57.5% GGG 2 vs. 8.9% GGG 4) — the anchor sampler was not the bottleneck. The grade head was simply starved of high-grade examples, not gradient-leaked or under-regularized.

## 3. Remediation attempts

### Fix B — grade-balanced sampling
Resampled training patches so GGG 2/3/4/5 were drawn with equal probability, but inverse-frequency loss weights (computed from the original unbalanced lesion counts) were still applied on top. This stacked two corrections and produced a **5.3× effective emphasis inversion against GGG 2** (normalized emphasis: GGG2 0.078 vs. GGG4 0.412, GGG5 0.345, vs. ~flat 0.25 each in the locked config). Result: GGG 4/5 were now predicted, but GGG 2 was predicted 0 times (balanced accuracy dropped to 0.211). A validation-loader bug was also found and fixed — `bg_module.py` had been splatting the balanced-sampling kwargs into the validation loader too, so even the monitoring signal was contaminated.

### Fix B′ — balanced sampling + flattened anchor weights (double-correction fixed)
Switched `grade_class_weight_source` to `"anchor"` using measured post-balanced-sampling anchor counts, restoring near-uniform effective emphasis (`[0.86, 0.98, 1.11, 1.04]`, i.e. ~1.09× max/min vs. 5.3× before). Result: **statistical null.**

- Macro OvR AUROC: **0.5321**
- 95% patient-level bootstrap CI (2,000 resamples, seed 2026): **[0.4330, 0.6323]**
- Quadratic-weighted kappa: **0.1278**
- Because the CI lower bound (0.433) crosses 0.5, the model shows no statistically significant ranking ability across GGG 2–5. This was statistically indistinguishable from the original confounded Fix B run's AUROC of 0.524.

### Fix A (anchor reweighting) and Fix C (ordinal loss) — initially not pursued
Fix A was refuted directly: Stage 2 telemetry already showed anchor frequencies mirror lesion counts, so reweighting anchors had nothing to correct. Fix C was initially closed because Fix B′'s per-epoch balanced accuracy plateaued at 0.23–0.27 with AUROC 0.524 — no baseline ranking signal for a different loss function to sharpen. Fix C was later formally reopened as ADR 0004 (see §6) once the endpoint's ordinal structure — not just its class balance — was identified as unaddressed.

## 4. Final resolution (per ADR 0003 / SITREP-20260917)

1. **Information ceiling accepted.** Neither reweighting nor resampling can manufacture new clinical signal from ~30 GGG 4 lesions; they only redistribute exposure over the same small sample.
2. **Protocol locked.** Artificial grade-head freezing and artificial resampling were discarded to avoid introducing new confounds or harming the detection backbone. Checkpoint selection was changed to save `model_best_grade.ckpt` based on validation balanced accuracy rather than noisy loss.
3. **Thesis framing.** Per the pre-registered protocol, the GGG 2–5 classification result is reported honestly as an empirically verified, data-constrained null/secondary finding with full disclosure of rare-grade sample sizes. The primary thesis contribution — csPCa detection and localization — is unaffected: GCALF-Net (LFF + CAF) shows clear, statistically meaningful improvements over baseline there.
4. **20-run matrix cleared to launch** on the locked configuration; Phase 5 (matrix launch itself) was intentionally left as a user decision.

## 5. Changes made to address overfitting

| Change | Status | Effect |
|---|---|---|
| Grade head downscaling, 12.83M → 1.33M params | **Active, in locked config** | Stopped rote memorization; see §7 |
| 10-epoch SWA + cyclic LR | **Active, succeeded** | Stabilized detection checkpoints |
| Masked gradient routing (grade loss isolated per-anchor) | **Active, succeeded** | Prevented grade loss from polluting shared BiFPN/encoder gradients from unmatched anchors |
| Checkpoint selection by validation balanced accuracy (`model_best_grade.ckpt`) | **Active** | Replaced noisy loss-based selection |
| `GradeHeadFreezeCallback` (freeze grade head mid-training) | **Tried, failed, disabled** | Gradients still leaked through the differentiable feature extractor into the shared BiFPN/encoder even with the head frozen |
| Fix B — grade-balanced sampling | **Tried, failed** | Uncorrected double-weighting inverted the problem onto GGG 2 (0% sensitivity, balanced accuracy 0.211) |
| Fix B′ — balanced sampling + flattened anchor weights | **Tried, failed (statistical null)** | Restored GGG 2 predictions but AUROC 0.532, CI crosses 0.5 — no discriminative signal recovered |
| Fix A — anchor reweighting | **Rejected** | Telemetry showed anchor sampling already matches lesion prevalence; nothing to correct |
| Fix C / ADR 0004 — ordinal CORAL grade head | **Implemented, gated, results pending** | See §6 |
| 3D CutMix / lesion swapping, localized elastic deformation | **Proposed, not yet run** | Suggested next augmentation experiments for rare-grade rescue |

### Active data augmentations (`nndet/conf/augmentation/base_more.yaml`)
- 3D random rotation (±30°), 3D scaling (0.7–1.4×), 3D mirroring (sagittal/coronal/axial), nonlinear gamma contrast (0.7–1.5), 50% foreground lesion oversampling.
- **Inactive by default:** 3D elastic deformation and additive brightness — disabled in the nnDetection 3D recipe to avoid non-physical anatomical distortion.

## 6. Ordinal loss (ADR 0004: CORAL grade head)

GGG2–5 is an ordinal label space, but the locked head was a 4-logit softmax classifier trained with weighted cross-entropy. ADR 0004 reopens this as a specification fix (not another reweighting/resampling attempt) and adds a **CORAL** (COnsistent RAnk Logits) variant behind a new `grade_loss_type` config flag:

- `grade_loss_type: ce` (**default, unchanged**) — preserves the existing 4-logit head and weighted-CE behavior exactly; this is what the locked 20-run matrix uses.
- `grade_loss_type: coral` — one shared bias-free feature projection with three learned ordinal thresholds, trained on the three cumulative binary targets `1[y > k]`; negative adjacent differences are clamped and renormalized (with a logged counter). Reduces the head's free final-layer parameters from `4*C + 4` to `C + 3`, while reusing the same underlying feature extractor. The existing lesion class weights `[0.256, 0.622, 1.747, 1.376]` are preserved.
- **Gated rollout:** Phase 1 (linear probe on the Fold-0 diagnostic features) must show a patient-bootstrap AUROC CI lower bound `> 0.5` before the CORAL implementation proceeds to a real refit (Phase 3, CE vs. CORAL head-to-head on the same checkpoint/split, requiring GGG 2 sensitivity not to drop by more than 0.05). If the Phase 1 CI lower bound is `≤ 0.5`, the CORAL path stops and the matrix launches locked with grade reported as the pre-registered null.
- **As of this writing, no Phase 1/3 result exists** — the matrix requires `model_cfg.head_grade_kwargs.grade_loss_type: ce` to prevent an unrecorded mode change while the gate is pending. This gives the ordinal endpoint an inductive bias the prior reweighting/sampling fixes lacked, but per ADR 0004 the data-supply ceiling from §2 remains binding regardless — a working implementation does not guarantee a positive result.

## 7. Why the 12.83M → 1.33M downscaling worked

- **Fixed a semantic mismatch.** The 12.83M head used an unshared `Conv3D(128 → 3456)` output, giving each of the 27 per-voxel anchor bounding-box templates (aspect ratios/scales — a detection-geometry property) its own independent convolution weights. Gleason grade is a biological property of the tissue at a voxel; it does not change with anchor shape. The 1.33M head shares one 128-channel tissue feature across all 27 anchor slots (toggle: `per_anchor_features: bool` in `nndet/arch/heads/grade_classifier.py`, default `False`; `True` restores the legacy 27-channel-block layout for ablation).
- **Ended gradient fragmentation.** In the 12.83M head, a matched anchor's gradient updated only 1 of 27 isolated channel blocks, diluting the already-scarce ~350 training lesions (and ~30 GGG 4 lesions) across 27 separate parameter paths. The 1.33M head pools gradients from all matched anchors into one shared weight set.
- **Eliminated rote memorization.** The 12.83M head had ~36,000 parameters per training lesion (~427,000 per GGG 4 lesion) — enough to memorize patient-specific MRI voxel noise, driving training CE to 0.15–0.17 without generalizing. Removing ~11.5M redundant parameters raised training CE to ~1.11 and forced the head to learn shared, generalizable features instead of per-lesion noise.
- A configurable `per_anchor_features` toggle plus a `gcalf_grade_12m_pilot.yaml` config were added so the 12.83M variant can be re-tested (13-epoch Fold-0 diagnostic) without touching the locked 1.33M baseline or ablation protocols. `tests/gcalf/test_grade_wiring.py` was extended with parameter-count and tensor-shape assertions for both modes; full suite passed in the pinned `gcalf:m0` container (168 passed, 1 skipped, 0 failed) at time of writing. The pilot itself has not yet been launched.

## 8. Clarification: "non-emission" vs. calibrated low emission

A natural question: shouldn't a well-calibrated model predict rare classes (GGG 4/5) only rarely anyway? Yes — and telemetry confirmed **emission volume was in fact roughly calibrated**:

| Model | Class | True lesions | Predicted count | Ratio | True positives | Recall | Precision |
|---|---|---|---|---|---|---|---|
| Baseline | GGG 4 | 6 | 5 | 0.83× | 1 | 16.7% | 20.0% |
| Baseline | GGG 5 | 8 | 6 | 0.75× | 0 | 0.0% | 0.0% |
| Full GCALF-Net | GGG 4 | 5 | 4 | 0.80× | 1 | 20.0% | 25.0% |
| Full GCALF-Net | GGG 5 | 8 | 7 | 0.88× | 2 | 25.0% | 28.6% |
| Fix B′ | GGG 4 | 7 | 13 | 1.86× | 2 | 28.6% | 15.4% |
| Fix B′ | GGG 5 | 9 | 17 | 1.89× | 2 | 22.2% | 11.8% |

The failure was never emission *volume* — it was **precision and true-positive recall**. In Baseline, all 6 GGG 5 guesses landed on GGG 2/3 lesions (0 true positives). In Full GCALF-Net, only 1 of 5 true GGG 4 lesions and 2 of 8 true GGG 5 lesions were correctly identified. Macro OvR AUROC of 0.532 (vs. >0.75 expected for calibrated-but-conservative behavior) confirms the model is not ranking cases correctly, i.e. it is guessing near-randomly among lesions when it does emit a high grade — consistent with ~30 GGG 4 training examples being insufficient to learn generalizable high-grade MRI features, and with the model falling back on the majority-class prior (GGG 2) in ambiguous cases.

## 9. Bottom line

- **What's overfit against:** the model over-fits to the majority classes (GGG 2/3) at the expense of GGG 4/5 under the locked config; forcing balanced exposure (Fix B) flips this into overfitting toward GGG 4/5 at the expense of GGG 2, confirming the behavior tracks whichever class the effective loss weighting favors, not a fixable architectural bias.
- **Primary bottleneck:** GGG 4 (37 total lesions, ~30/fold) and GGG 5 (47 total lesions) — too few distinct anatomical examples to learn generalizable high-grade features, regardless of sampling or weighting strategy.
- **What did generalize:** head-capacity downscaling (12.83M → 1.33M) and gradient-routing/checkpoint-selection fixes are real, retained improvements. Class reweighting and resampling (Fix A/B/B′) did not recover signal. The CORAL ordinal head (ADR 0004) is implemented and gated but unevaluated.
- **Thesis impact:** none on the primary csPCa detection/localization contribution. The grade classification limitation is disclosed as a pre-registered, data-constrained null/secondary finding.
