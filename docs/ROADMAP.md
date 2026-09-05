# GCALF-Net Roadmap

`ARCHITECTURE.md` says *what* GCALF-Net is and *why*. This file says *when it is done*: the
architecture split into milestones, each with one goal and one test that decides pass/fail. The
step-by-step *how* is in `phases/PHASE_*.md`.

**Rules.** Milestones are sequential; M4 (built baseline, full train) must close before M6 (LFF)
or M7 (CAF) starts. A milestone closes only when its test runs and passes — not when the code
"looks right." Every gate below is tagged **L** (runs in the pinned `gcalf:m0` image, CPU-only) or
**V** (requires a GPU — Vast.ai). No **V** gate is evidence for a milestone whose **L** gate has
not passed. See `docs/adr/0002-*.md` for why each milestone is shaped this way.

| M | Milestone | Phase | Goal | Gates |
|---|---|---|---|---|
| M0 | Environment | [0](phases/PHASE_0_environment.md) | `nndet`, PI-CAI tools, medcam all import; CUDA build verified | L, V |
| M1 | Data pipeline | [1](phases/PHASE_1_data_pipeline.md) | Validated task: `csPCa` detection class (1,500 cases) + grade metadata (220+ audited lesions) | L |
| M2 | Baseline build (FDR + WAF) | [2](phases/PHASE_2_baseline.md) | FDR built, WAF wired, replacing the released wavelet/channel-light modules | L, V |
| M3 | Baseline forward + overfit | [2](phases/PHASE_2_baseline.md) | Grade head loss routing correct; 2-case overfit drives loss to ~0 | L, V |
| M4 ⭐ | Baseline full train | [2](phases/PHASE_2_baseline.md) | Real FDR+WAF baseline numbers on PI-CAI — thesis's first result | V |
| M5 | Budget ladder decision | [2](phases/PHASE_2_baseline.md) | Fold-0 pilot measurement selects a pre-committed matrix rung | V |
| M6 | LFF | [3](phases/PHASE_3_lff.md) | Learned real-valued gain replaces FDR's fixed mask, baseline-invariant | L, V |
| M7 | CAF | [4](phases/PHASE_4_caf.md) | Bidirectional windowed Q/K/V replaces WAF's self-attention | L, V |
| M8 | Full GCALF + ablation matrix | [5](phases/PHASE_5_integration_ablation.md) | All four configs run on identical folds/seeds | V |
| M9 | Evaluation | [6](phases/PHASE_6_evaluation.md) | Detection + grade metrics + defended stats + bootstrap CIs, every config | L, V |
| M10 | Grad-CAM | [7](phases/PHASE_7_gradcam.md) | 3D CAMs on the grade head, blinded urologist packets exported | L, V |
| M11 | Thesis outputs | [6](phases/PHASE_6_evaluation.md), [7](phases/PHASE_7_gradcam.md) | Every SOP answered by a named table or figure | — |

---

## M0 — Environment

**Goal.** Two environments per `ARCHITECTURE.md §12`: the pinned `gcalf:m0` image (Python 3.8,
torch 1.10.1/CUDA 11.3, `nndet` `csrc` compiled) where every reported result must reproduce, and a
disposable modern-CUDA scratch environment for LFF/CAF math prototyping on the local GPU. Neither
substitutes for the other.

**L gate.**
```bash
python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py
python -c "import nndet, nndet._C; print(nndet.__file__)"     # resolves inside GCALF-Net/, not PDHD-Net/
python -c "import picai_prep, picai_eval, medcam"
```
**V gate.**
```bash
docker run --rm --gpus all gcalf:m0 python -m pytest -q tests/test_csrc_cuda.py   # must run, not skip
```
This has never passed (`docs/m0-verification.md` records the Docker/CDI failure that blocked it).
Re-run it before trusting any downstream CUDA result.
**Blocks.** Everything. `csrc` is the #1 historical blocker — do not start M1 with a half-working env.

## M1 — Data pipeline

**Goal.** A task with one `csPCa` detection foreground class over all 1,500 cases, and grade
metadata (`grade`, `grade_source`, `grade_supervised`) on every positive instance — 220 baseline
graded lesions plus whatever the unifocal linkage recovery audit (Phase 1) adds from the 205
Pooch25 cases. Official 5-fold splits loaded and independently verified.

**Known gap.** `gcalf_data/{build_labels,prepare_picai,sanity_checks}.py` currently implement the
rejected native-4-class design (ADR 0002 D2's rejected alternative). They need real rework, not a
patch — see Phase 1 for the scope.

**L gate.**
```bash
python -m gcalf_data.sanity_checks    # all asserts green
```
covering: identical spacing/orientation across T2W/ADC/HBV after resampling to a common grid;
`dataset.json["labels"] == {"0": "csPCa"}`; every instance's detection class is `0`; every graded
instance's `grade ∈ {2,3,4,5}`; instance-volume IDs == `case.json` keys; no `patient_id` crosses
folds; every held-out fold contains every grade.
**Closes when.** Sanity checks pass, the linkage-recovery audit report is committed, and
`docs/data_report.md` records the final grade-supervised lesion count per grade.
**Watch.** Any GGG1 (ISUP 1) foreground detection instance, or any grade assigned by inference
rather than the audit's 1-lesion/1-component rule, means the labels were fabricated — fix the
data, never the plan.

## M2 — Baseline build: FDR + WAF

**Goal.** Build the fixed FFT frequency decomposition (FDR) and wire the existing (but dead)
`WindowAttentionFusion` (WAF) as the frozen control, per `ARCHITECTURE.md §5, §7`. This is new
code, not a config flag — treat it as the largest single engineering item in the plan.

**L gate.** `pytest tests/gcalf/test_fdr.py tests/gcalf/test_waf.py` — mask radius exact,
low/high split reconstructs the input losslessly, orientation gates pass (constant → low branch,
Nyquist checkerboard → high branch), WAF shape/gradient tests pass, CPU forward/backward through
the full encoder at `(1,3,32,256,256)`.
**V gate.** CUDA forward/backward parity with the CPU result; peak memory at stages `[1,3,4]`
(FDR) and `[2,5]` (WAF) recorded.
**Closes when.** Both modules pass their tests and are wired as the `baseline` config's default.

## M3 — Baseline forward pass + grade-head overfit

**Goal.** The two-head model (detection + masked grade head, `ARCHITECTURE.md §8`) instantiates,
forward-passes, and — critically — the grade head's masked loss routing is provably correct.

**L gate.** Forward `(1,3,32,256,256)` with no shape error. **Overfit 2 real cases** (one
grade-supervised positive, one benign): assert final loss ≤10% of initial, the positive's grade
logit matches its label at ≥0.9 probability, the benign case contributes zero grade-head gradient
(assert directly on `grade_head` parameter grads after a benign-only batch).
**V gate.** Same overfit on GPU; confirm no numerical divergence under `precision: 16`.
**Closes when.** The overfit gate passes. This is the single test that proves the masked loss
routing works — a model that cannot memorize 2 cases, or that leaks gradient into the grade head
from an unsupervised lesion, is broken, not undertrained.

## M4 ⭐ — Baseline full train

**Goal.** FDR+WAF baseline trained on real PI-CAI folds at the shipped schedule (50 epochs × 2500
batches + 10 SWA), evaluated, reported. **This is the reference column every later result is
measured against**, and it is the paper's architecture adapted to 3-channel bpMRI — not the
released `WaveletSpatialFusion`/`ChannelWiseLightFusion` code (ADR 0002 D4). State that plainly in
the thesis.

**V gate.** `gcalf_experiments/baseline_*/metrics.csv` contains FROC, case-level AUROC, the 4×4
grade confusion matrix (grade-supervised lesions only) with misses/FPs reported alongside,
weighted F1, per-grade sensitivity; fold-0 measured seconds/step and wall time recorded.
**Closes when.** Baseline metrics reported, commit tagged `baseline-v1`.
**Gate → M5.** Do not launch the 20-run matrix from this milestone's numbers alone — M5 decides
the ladder rung first.

## M5 — Budget ladder decision

**Goal.** Turn the fold-0 pilot's measured seconds/step into a committed matrix plan, per
`ARCHITECTURE.md §9` and ADR 0002 D10 (planned rung: full 20-run matrix, $150–350 budget).

**V gate.** Record, before any other fold starts: measured seconds/step, projected full-matrix
GPU-hours and cost, and which of the three pre-committed rungs (full / halved batches-per-epoch,
all runs / 14-run reduced) is selected. This record is written into every subsequent `run.json` —
choosing after seeing fold results is test-set tuning.
**Closes when.** The rung is recorded and unanimous across all subsequent run configs.

## M6 — LFF

**Goal.** `LearnableFrequencyFilter3D` replaces FDR at stages `[1,3,4]` via the registry; FDR
remains the default so the refactor cannot move the baseline.

**L gate.** `pytest tests/gcalf/test_lff.py` — shape in==out; identity at init (`delta_H=0`); the
**orientation gate** (centre-weighted grid low-passes: constant survives, Nyquist checkerboard is
suppressed — catches a missing `ifftshift`); gradients reach `delta_weight`; finite under autocast
on CPU. Plus a fixed-seed regression test proving the FDR baseline's output is byte-identical
after the registry refactor.
**V gate.** Finite gradients and stable logging on one short CUDA run; per-stage peak memory
recorded.
**Closes when.** Tests pass; `lff_only` completes tiny train/resume/predict/eval.

## M7 — CAF

**Goal.** `WindowedCrossAttentionFusion3D` replaces WAF at stages `[2,5]` with genuine
bidirectional cross-attention (CNN queries Swin, Swin queries CNN) in place of WAF's
self-attention.

**L gate.** `pytest tests/gcalf/test_caf.py` — output shape matches CNN dims for divisible and
padded shapes; padding invariance (fails without `key_padding_mask`); residual counted exactly
once (zero attention outputs + identity `out_proj` ⟹ output == aligned CNN feature, not a multiple
of it); both branch projections receive gradients.
**V gate.** Stage-2/stage-5 tensors pass the declared memory gate; short CUDA training segment
with finite losses.
**Closes when.** Tests pass; `caf_only` completes tiny train/resume/predict/eval.
**Watch.** If windowed attention will not fit, walk the fallback ladder (`ARCHITECTURE.md §7`)
and report the resolved architecture. BiFusion is never relabeled CAF.

## M8 — Full GCALF-Net + ablation matrix

**Goal.** Both modifications on together, then all four configs (`baseline`, `lff_only`,
`caf_only`, `gcalf_full`) trained on the same folds/seeds at the M5-committed rung — pure config
work, no new model code.

**V gate.** `gcalf_eval/collect_results.py` emits one table, rows = 4 configs, baseline as
reference column with Δ per metric; each run directory carries `config_snapshot.yaml`,
`git_commit.txt`, `env.txt`; re-running `baseline.yaml` after the registry refactor matches
`baseline-v1` exactly (a drifted baseline invalidates the whole comparison — check this before
trusting any ablation result).
**Closes when.** The four-way table exists and no fold was dropped for only some configs.

## M9 — Evaluation & statistics

**Goal.** Detection + grade metrics for every config, the defended significance test, and
patient-level bootstrap CIs (ADR 0002 D7 — both, not one instead of the other).

**L gate.** Metric and statistical-decision-tree tests on fixed synthetic fixtures, covering every
branch (normal → ANOVA/Tukey; non-normal → Friedman/Wilcoxon).
**V gate.** Evaluate each completed fold; aggregate only after the full matrix is present; compute
paired bootstrap CIs from preserved out-of-fold predictions.
**Closes when.** Weighted F1 (primary), macro-F1, per-grade sensitivity/precision, quadratic
Cohen's κ, FROC, case-level AUROC, the defended p-value, and bootstrap CIs are all reported per
config, with the detection and grade-matched denominators shown side by side (never collapsed).

## M10 — Grad-CAM

**Goal.** 3D Grad-CAM on the frozen full GCALF-Net's grade head (`ARCHITECTURE.md §11`), blinded
review packets for 3 urologists, 30 cases stratified across GGG2–5.

**L gate.** Hook/target/shape test on synthetic tensors; rendered overlay pipeline runs
end-to-end on a synthetic case.
**V gate.** Held-out inference on real cases; export of the approved, blinded review set.
**Closes when.** CAMs localize to matched lesions on known positives; packets + rating sheets are
in urologists' hands. **Start recruiting the three urologists at M0** — they are the only
dependency outside this team's control and they sit at the end of the chain.

## M11 — Thesis outputs

**Goal.** Every SOP answered by a named table or figure; reproducibility audit complete.

**Closes when.** Each run is linked to its code revision, environment, dataset-manifest version,
fold, preprocessing config, seed, checkpoints, predictions, metrics, and statistical outputs.
Patient data, derived imaging, checkpoints, and experiment artifacts stay out of git. Every
blocked or amended protocol requirement (this roadmap's own amendments included) is recorded
explicitly in the thesis, not silently absorbed.
