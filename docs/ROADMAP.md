# GCALF-Net Roadmap

`SPEC.md` says *what* GCALF-Net is. This file says *when it is done*: the spec split into 11 milestones, each with one goal and one test that decides pass/fail. The step-by-step *how* stays in `phases/PHASE_*.md` — this is the ledger, not a third copy of the plan.

**Rules.** Milestones are sequential; M4 (full baseline) must close before M5 (LFF) starts. A milestone closes only when its test runs and passes — not when the code "looks right". Every test below is a command or an assertion, never a judgement call. Tick the box, record the tagged commit, move on.

| M | Milestone | Phase | Goal (one sentence) |
|---|---|---|---|
| M0 | Environment | [0](phases/PHASE_0_environment.md) | One reproducible env where `nndet`, PI-CAI tools and medcam all import |
| M1 | Data pipeline | [1](phases/PHASE_1_data_pipeline.md) | A validated nnDetection task with 3-channel bpMRI and per-lesion GGG labels |
| M2 | Baseline forward | [2](phases/PHASE_2_baseline.md) | The adapted 3-channel model builds, runs, and can memorize 2 cases |
| M3 | Baseline tiny train | [2](phases/PHASE_2_baseline.md) | Train→predict→eval works end-to-end on a 6-case task |
| M4 ⭐ | Baseline full train | [2](phases/PHASE_2_baseline.md) | Real baseline numbers on PI-CAI — the thesis's first result |
| M5 | LFF | [3](phases/PHASE_3_lff.md) | Learnable 3D Fourier filter replaces the wavelet module, baseline-invariant |
| M6 | CAF | [4](phases/PHASE_4_caf.md) | Bidirectional windowed Q/K/V cross-attention replaces channel-light fusion |
| M7 | Full GCALF + ablation | [5](phases/PHASE_5_integration_ablation.md) | All four configs run on identical folds/seeds |
| M8 | Evaluation | [6](phases/PHASE_6_evaluation.md) | Detection + 5-class metrics + significance test for every config |
| M9 | Grad-CAM | [7](phases/PHASE_7_gradcam.md) | 3D CAMs wired to the right class/layer, blinded packets exported |
| M10 | Thesis outputs | [6](phases/PHASE_6_evaluation.md), [7](phases/PHASE_7_gradcam.md) | Every SOP answered by a named table or figure |

---

## M0 — Environment

**Goal.** A single pinned environment (Python 3.8, torch 1.10.1/CUDA 11.3, nnDetection `csrc` compiled) in which every dependency of the thesis imports, plus the empty `gcalf/` code tree committed.

**Test.**
```bash
python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py tests/test_csrc_cuda.py
python -c "import nndet, nndet._C; print(nndet.__file__)"     # must resolve inside GCALF-Net/, not PDHD-Net/
python -c "import picai_prep, picai_eval, medcam"
```
**Closes when.** The tests run in the GPU Docker container (the CUDA test does not skip), and `environment.yml` + `env.lock.txt` are committed on `feat/env-and-data`.
**Blocks.** Everything. Do not start M1 with a half-working env — `csrc` is the #1 blocker (`SPEC.md §14`).

## M1 — Data pipeline

**Goal.** `nnDet_raw/Task2xx_PICAI/` populated from PI-CAI with 3 co-registered modalities per case, per-lesion GGG stored as 0-indexed foreground classes (`lesion_ISUP k → class k-1`), benign cases carrying `"instances": {}`, and the official 5-fold splits loaded.

**Test.**
```bash
python gcalf_data/sanity_checks.py    # all asserts green
```
covering: identical spacing/orientation across T2W/ADC/DWI; channel order `_0000/_0001/_0002`; every instance class ∈ {0..4}; instance-volume ids == `case.json` keys; per-class counts match the marksheet after the `k-1` shift; no `patient_id` crosses folds.

**Closes when.** Sanity checks pass, `nndet_prep` emits a plan with `in_channels=3` and `classifier_classes=5`, and `docs/data_report.md` records the benign/label decisions.
**Watch.** `classifier_classes=6` means `build_labels.py` invented a benign class — fix the data, never the plan.

## M2 — Baseline forward pass

**Goal.** The unmodified released architecture (`wavelet` + `channel_light`) instantiates from the M1 plan at 3 channels and trains at all.

**Test.** Forward `(1,3,D,H,W)` through `RetinaUNetV001` with no shape error, shapes matching `SPEC.md §5`; then **overfit 2 cases** and assert training loss → ~0.
**Closes when.** The overfit run drives loss to near zero. This is the only test that proves labels, loss and head are wired together — a model that cannot memorize 2 cases is broken, not undertrained.

## M3 — Baseline tiny train

**Goal.** The whole pipeline — prep → train → predict → eval — executes unattended on `Task9xx_PICAI_TINY` (4–6 cases, mixed grades + 1 benign).

**Test.** 2-epoch run under `nndet/conf/train/smoke.yaml` → `scripts/predict.py` → `gcalf_eval/run_eval.py` produces a populated `metrics.csv`.
**Closes when.** Numbers exist. Their *value* is meaningless here — this milestone tests plumbing, not accuracy.

## M4 ⭐ — Baseline full train

**Goal.** The adapted 3-channel PDHD-Net trained on real PI-CAI folds at the shipped schedule (50 epochs × 2500 batches + 10 SWA), evaluated, and reported. **This is the reference column every later result is measured against.**

**Test.** `gcalf_experiments/baseline_*/metrics.csv` contains FROC, 5×5 confusion matrix, macro-F1 and per-class sensitivity; fold-0 measured **seconds/step and wall time** recorded.
**Closes when.** Baseline metrics reported and commit tagged `baseline-v1`.
**Gate.** The fold-0 pilot number selects one of the three pre-committed budget options (`SPEC.md §12`) — **choose and record before launching the matrix**, never after seeing results.

## M5 — LFF

**Goal.** `LearnableFrequencyFilter3D` replaces `WaveletSpatialFusion` at stages [1,3,4] via the registry, with the wavelet path still the default so the refactor cannot move the baseline.

**Test.** `pytest tests/gcalf/test_lff.py` — shape in == shape out across several `(D,H,W)`; identity at init (`delta_H=0`, `allclose` atol 1e-4); gradients reach `delta_weight`; **a centre-weighted grid low-passes** (constant survives, Nyquist checkerboard is suppressed — this is the test that catches a missing `ifftshift`); finite under autocast on CPU and CUDA. Plus a fixed-seed regression test proving baseline output is byte-identical after the registry refactor.
**Closes when.** Tests pass, `lff_only` completes tiny train/resume/predict/eval, per-stage peak memory recorded.

## M6 — CAF

**Goal.** `WindowedCrossAttentionFusion3D` replaces `MemoryEfficientFusion` at stages [2,5] with genuine bidirectional cross-attention — CNN queries Swin, Swin queries CNN.

**Test.** `pytest tests/gcalf/test_caf.py` — output shape matches CNN dims for divisible *and* padded shapes; **padding invariance** (refilling the padded region with a different constant leaves the valid region unchanged — fails if `key_padding_mask` is missing); **residual counted once** (zero attention outputs + identity `out_proj` ⟹ output == aligned CNN feature, not a multiple of it); both branch projections receive gradients; stage-2 and stage-5 tensors pass the declared memory gate.
**Closes when.** Tests pass and `caf_only` completes tiny train/resume/predict/eval.
**Watch.** If windowed attention will not fit, walk the documented fallback ladder and report the resolved architecture. BiFusion is never relabeled CAF.

## M7 — Full GCALF-Net + ablation matrix

**Goal.** Both modifications on together, then all four configs (`baseline`, `lff_only`, `caf_only`, `gcalf_full`) trained on the **same folds and seeds** — pure config work, no new model code.

**Test.** `gcalf_eval/collect_results.py` emits one table, rows = 4 configs, with the baseline as reference column and Δ per metric; each run dir carries `config_snapshot.yaml`, `git_commit.txt`, `env.txt`.
**Closes when.** The four-way table exists and every run's config snapshot differs only in the two module flags.
**Invariant.** Never drop folds for some configs only — that breaks the paired statistics in M8.

## M8 — Evaluation & statistics

**Goal.** Every config scored on detection (picai_eval FROC/AUROC) and 5-class GGG classification, with the significance test that answers SOP 3.

**Test.** Per config: 5×5 confusion matrix over matched lesions with misses and false positives reported *beside* it (never as a 6th row), macro-F1, balanced accuracy, per-class sensitivity (**GGG2 vs GGG3 is the headline**), quadratic-weighted κ, bootstrap 95% CIs. Across configs: **Wilcoxon signed-rank on paired per-fold metrics**, GCALF-full vs baseline, p-value reported.
**Closes when.** Comparison table, confusion-matrix figures and FROC curves are all regenerable by one script.

## M9 — Grad-CAM

**Goal.** 3D CAMs from `medcam.inject`, attached to the classifier layer and the correct target class, rendered into blinded per-case packets for the three urologists.

**Test.** `pytest tests/test_gradcam.py` — CAM is non-empty and matches input spatial shape; CAM **changes when `label` changes** (not constant across classes); spatial argmax falls inside the lesion mask for a known strong positive (or `medcam.evaluate` overlap > chance); injected model's forward output is unchanged vs un-injected.
**Closes when.** Tests pass and blinded PNG packets (3 slices × 3 sequences, no GT/prediction visible) are exported.
**Note.** If per-anchor logits make the detection head awkward, run the CAM study on the classifier-fallback model (`SPEC.md §14`) — its clean `(B, num_classes)` output is the recommended path.

## M10 — Thesis outputs

**Goal.** Every SOP is answered by a specific, named artifact.

**Test.** Each row below points at a file that exists:

| SOP | Answered by |
|---|---|
| SOP 1 — adapted PDHD-Net baseline | M4 baseline metrics table |
| SOP 2 — LFF / CAF contribution | M7 four-way ablation table |
| SOP 3 — combined significance | M8 Wilcoxon p-value |
| SOP 4 — clinical interpretability | M9 urologist Likert results |

**Closes when.** All four rows resolve, and the ablation plots, confusion matrices and CAM figures are exported at thesis quality.

---

## Contingency

If detection plumbing blocks progress past M2–M3, switch to the **classifier-fallback model** (`SPEC.md §14`): `Encoder` (LFF/CAF swaps intact) + global pool + linear GGG head on `case_ISUP`. Every milestone from M5 onward still closes on its own test — the four-way ablation and the Grad-CAM study survive untouched. Only FROC and Dice are lost, and they become "additional results if detection works."
