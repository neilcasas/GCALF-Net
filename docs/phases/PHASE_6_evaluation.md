# Phase 6 — Evaluation & Statistical Analysis

**Milestone:** M9 · **Depends on:** Phase 5 (ablation runs) · **Blocks:** thesis results chapter
**Goal:** compute detection, grade, and (optional) segmentation metrics for all four configs; run
the defended significance test *and* patient-level bootstrap CIs (ADR 0002 D7); produce the
comparison tables and figures answering SOPs 1–3.

**Definition of done:**
- [ ] **V:** picai_eval FROC/lesion-AUROC/case-level AUROC computed per config, over all 1,500 cases.
- [ ] **V:** 4×4 grade confusion matrix (grade-supervised matched lesions only), weighted F1
      (primary), macro-F1, per-grade sensitivity/precision, quadratic-weighted κ, per config —
      with misses and false positives reported alongside, never folded in.
- [ ] **V:** Shapiro-Wilk → ANOVA/Tukey or Friedman/Bonferroni-Wilcoxon: GCALF-full vs baseline,
      p-value reported (SOP 3).
- [ ] **V:** patient-level paired bootstrap CIs and effect sizes for every pairwise comparison.
- [ ] Comparison table + confusion-matrix figures + FROC curves exported.

---

## 6.1 Detection / case-level — picai_eval, all 1,500 cases

`picai_eval/src/picai_eval/eval.py::evaluate` on detection maps from `scripts/predict.py`:
- **FROC** (sensitivity vs. false-positives-per-case) — the PI-CAI standard.
- **Lesion-level AUROC.**
- **Case-level AUROC** — max lesion confidence per case, the standard PI-CAI benign-vs-csPCa
  endpoint. This is free once detection works and uses the full 1,500-case cohort, unlike every
  grade metric below.
- Answers "does it find clinically significant disease" for every config, independent of grading.

## 6.2 Grade classification (grade-supervised lesions only — the thesis headline)

Per matched positive detection, restricted to lesions with `grade_supervised: true`
(`ARCHITECTURE.md §8`). `gcalf_eval/run_eval.py`:

- **4×4 confusion matrix**, GGG2–5, over matched, grade-supervised lesions. Report **two numbers
  beside it, not inside it**: missed lesions (unmatched grade-supervised ground truth, per grade)
  and false positives (unmatched detections, split by benign vs. positive case). State the
  matching criterion and score threshold used. Always report the grade-matched denominator next
  to the full detection denominator (§6.1's 1,500) so a reader cannot mistake one for the other.
- **Weighted F1** — primary endpoint, per the thesis.
- **Macro-F1**, balanced accuracy, per-grade sensitivity/recall and precision.
- **GGG2 vs. GGG3 sensitivity** — call this out explicitly; it is the clinical threshold the
  thesis's problem statement targets.
- **Quadratic-weighted Cohen's κ** — grades are ordinal; κ_w penalizes far-off errors more than
  adjacent-grade errors, unlike plain accuracy or unweighted κ.
- **One-vs-rest AUROC** per grade + macro.
- **95% CIs via bootstrap** — mandatory given the Phase 1 audit's per-grade counts (floor: GGG4=20,
  GGG5=18 total cases, roughly 4 held out per fold each before recovery). Lean on macro/balanced
  metrics over raw accuracy.
- If Phase 1's audit triggered the pre-registered GGG4+5 merged secondary analysis (ADR 0002 D8),
  report it as a clearly labeled secondary table, never substituted for the primary 4-class result.

## 6.3 Segmentation (only if the segmentation head is trained)

Dice (DSC) vs. lesion delineations, via `picai_eval`/nnDetection's segmenter output, for matched
positive detections. State out-of-scope explicitly if not trained.

## 6.4 Ablation comparison (SOP 1, 2)

From `collect_results.py` (Phase 5): rows {baseline, lff_only, caf_only, gcalf_full}, baseline as
reference, Δ columns. Pairwise readings: baseline vs. lff_only → LFF's contribution; baseline vs.
caf_only → CAF's contribution; baseline vs. gcalf_full → combined.

## 6.5 Statistics (SOP 3) — both the defended test and bootstrap CIs, per ADR 0002 D7

1. **Defended route, run exactly as specified, no amendment:** collect paired per-fold weighted-F1
   (or the thesis's designated primary metric) for baseline and gcalf_full. **Shapiro-Wilk** for
   normality on the 5 fold-level values. If normal: one-way repeated-measures **ANOVA** + two-sided
   **Tukey HSD**. If not (the likely outcome — 5 points rarely reject Shapiro-Wilk either way, but
   route through whichever branch the test actually selects): **Friedman** + all six paired,
   two-sided **Wilcoxon signed-rank** tests, Bonferroni threshold `0.05/6 = 0.00833`.
2. **Patient-level paired bootstrap, reported alongside, not instead:** using the out-of-fold
   predictions preserved in Phase 5, resample patients (keeping all of a resampled patient's
   lesions together) to build confidence intervals and paired permutation-test p-values for every
   comparison. This is the honest uncertainty estimate the 5-fold-level test cannot provide at
   n=5.
3. State the significance threshold (α=0.05) for both routes and whether each comparison is
   significant under each. Report effect sizes and all planned comparisons, not only favorable ones.

## 6.6 Figures & tables (`gcalf_eval/make_figures.py`)

- Confusion matrices (one per config, 4×4, normalized) + miss/false-positive counts beside each.
- FROC curves (all configs, one axis).
- Per-grade sensitivity bar chart (baseline vs. full, GGG2–5).
- Case-level AUROC comparison (all four configs, full 1,500-case cohort).
- Master comparison table as CSV + rendered PNG/LaTeX.
- Bootstrap CI plot per comparison (§6.5.2).
- All written under `gcalf_experiments/_results/`.

## 6.7 Compute note (cloud)

Evaluation reuses Phase 5's trained checkpoints — cheap (inference + metrics). Pull checkpoints
locally and run eval on CPU/small GPU if training ran on cloud; keep the same `gcalf_configs/*.yaml`
so local and cloud evaluation are identical.

## 6.8 Risks & fallbacks

| Risk | Fallback |
|---|---|
| Too few grade-supervised GGG4/5 in a fold → unstable per-grade metric | Report macro/balanced + wide CIs; pool folds for per-grade sensitivity; document explicitly. |
| picai_eval expects a specific prediction format | Follow its `evaluate()` docstring; convert nnDetection detection maps with `nndet_generate_detection_maps.py`. |
| Defended test non-significant | Report honestly; a non-significant trend with tight bootstrap CIs and qualitative Grad-CAM benefit still answers the SOPs. |
| Detection and grade-matched denominators get conflated in a figure | Fix the evaluation granularity once (grade metrics always over grade-supervised matched lesions) and apply identically to every config's figure. |

## 6.9 Deliverables & commit

- Branch `feat/eval` → commit "eval: picai_eval + grade metrics + defended stats + bootstrap CIs + figures".
- Files: `gcalf_eval/{run_eval,collect_results,make_figures}.py`, `gcalf_experiments/_results/`.

**Next:** `PHASE_7_gradcam.md`.
