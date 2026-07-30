# Phase 6 — Evaluation & Statistical Analysis

**Milestones:** M8, M10 (result tables) · **Depends on:** Phase 5 (ablation runs) · **Blocks:** thesis results chapter
**Goal:** compute detection, classification, and (optional) segmentation metrics for all four configs; run the statistical significance test; produce the comparison tables and figures that answer SOPs 1–3.

**Definition of done:**
- [ ] picai_eval FROC/AUROC computed per config.
- [ ] 5-class GGG metrics (5×5 confusion matrix, macro-F1, balanced acc, per-class sensitivity, quad-κ) per config, with misses and false positives reported alongside.
- [ ] Wilcoxon signed-rank: GCALF-full vs baseline, p-value reported (SOP 3).
- [ ] Comparison table + confusion-matrix figures + FROC curves exported.

---

## 6.1 Detection / lesion-level — picai_eval

`picai_eval/src/picai_eval/eval.py::evaluate` on the detection maps from `scripts/predict.py`:
- **FROC** (sensitivity vs false-positives-per-case) — the PI-CAI standard.
- **lesion-level AUROC**.
- Feed predicted lesion candidates + ground-truth lesion masks; picai_eval handles hit-matching.
- Answers "does it find lesions" — reported for every config.

## 6.2 Five-class GGG classification (the thesis headline)

Per-lesion (detection model) or per-case (classifier fallback). `gcalf_eval/run_eval.py`:
- **Confusion matrix — 5×5** over ground-truth lesions matched to detections, GGG1–5 (masterfile Table 3.2). The detection model has **no background class** (`classifier_classes=5`, see `THESIS_PLAN.md §0`), so there is no background row or column to draw. Report the detection-side error modes as two numbers beside the matrix — **missed lesions** (unmatched ground truth, per GGG) and **false positives** (unmatched detections, split by benign vs positive case) — and state the matching criterion and score threshold used. A "6×6 with background" matrix would describe a model that does not exist; if the classifier-fallback model is the one being evaluated, it *is* 6-way over `case_ISUP` and should be labelled as such.
- **Macro-F1**, **balanced accuracy**, **per-class sensitivity/recall** and **precision**.
- **GGG2 vs GGG3** sensitivity — call this out explicitly; it's the clinical threshold the whole thesis targets.
- **Quadratic-weighted Cohen's κ** — the grades are ordinal, so κ_w is the right agreement metric (penalizes far-off errors more).
- **AUROC** one-vs-rest per class + macro.
- **95% CIs via bootstrap** — mandatory given GGG4=40, GGG5=52; report CIs, lean on macro/balanced over raw accuracy.

## 6.3 Segmentation (only if lesion masks are used as targets)

- **Dice (DSC)** vs lesion delineations, via `picai_eval` / nnDetection's segmenter output. Report if the segmentation head is trained; otherwise state it's out of scope.

## 6.4 Ablation comparison (SOP 1, 2)

From `collect_results.py` (Phase 5), the master table: rows {baseline, lff_only, caf_only, gcalf_full}, baseline as reference, Δ columns. Pairwise readings:
- baseline vs lff_only → LFF contribution.
- baseline vs caf_only → CAF contribution.
- baseline vs gcalf_full → combined.

## 6.5 Statistical significance (SOP 3)

Masterfile §F cites Wilcoxon (1945) + Shapiro-Wilk (1965):
1. Collect **paired per-fold** (or bootstrap-resample) metric values for baseline and gcalf_full.
2. **Shapiro-Wilk** for normality → if normal, paired t-test; else **Wilcoxon signed-rank** (the masterfile's default).
3. Report the p-value for GCALF-full > baseline on the primary metric (macro-F1 or GGG2/3 sensitivity). `picai_eval/statistical_helper.py` and `stat_util/` have bootstrap helpers you can reuse.
4. State the significance threshold (α=0.05) and whether the improvement is significant.

## 6.6 Figures & tables to export (`gcalf_eval/make_figures.py`)

- Confusion matrices (one per config, 5×5, normalized) + the miss/false-positive counts beside each.
- FROC curves (all configs on one axis).
- Per-class sensitivity bar chart (baseline vs full, GGG1–5).
- The master comparison table as CSV + a rendered PNG/LaTeX for the thesis.
- All written under `gcalf_experiments/_results/`.

## 6.7 Compute note (cloud)

Full evaluation reuses the trained checkpoints from Phase 5; it's cheap (inference + metrics). If training ran on cloud (see `THESIS_PLAN.md §11–12`), pull checkpoints locally and run eval on CPU/small GPU. Keep the same `gcalf_configs/*.yaml` so local and cloud evaluation are identical.

## 6.8 Risks & fallbacks

| Risk | Fallback |
|---|---|
| Too few GGG4/5 in a fold → unstable per-class metric | Report macro/balanced + wide CIs; pool folds for per-class sensitivity; document. |
| picai_eval expects a specific prediction format | Follow its `evaluate()` docstring; convert nnDetection detection maps with `nndet_generate_detection_maps.py`. |
| Significance not reached | Report honestly; a non-significant but positive trend + qualitative Grad-CAM benefit still answers the SOPs. |
| Per-lesion vs per-case metric mismatch across configs | Fix the evaluation granularity once (per-lesion for detection model) and apply identically to all configs. |

## 6.9 Deliverables & commit

- Branch `feat/eval` → commit "eval: picai_eval + 5-class GGG metrics + Wilcoxon + figures".
- Files: `gcalf_eval/{run_eval,collect_results,make_figures}.py`, `gcalf_experiments/_results/`.

**Next:** `PHASE_7_gradcam.md`.
