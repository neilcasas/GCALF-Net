# ADR 0002: GGG2–5 Lesion-Level Protocol, Masked Grade Head, and a Built (Not Released) FDR/WAF Baseline

**Status:** Accepted | **Date:** 2026-09-05 | **Source:** `/grilling` session, code-verified against both `PDHD-Net/` and `GCALF-Net/`

## Context

ADR 0001 fixed the architecture and cloud contract assuming five-class GGG1–5 lesion-level
grading (`classifier_classes = 5`). Between ADR 0001 and this session the project:

1. discovered PI-CAI has no spatial GGG1 mask (ISUP ≤1 is encoded as background — see
   `picai_labels/README.md`), pivoted to a four-class **GGG2–5** native-instance protocol
   (commits `cf3390b`, `1b18cab`, `95e202e`), then
2. pivoted again to restore GGG1–5 with a separate grade head (commits `2b218f8` onward,
   the dirty working tree preserved on branch `ggg1-5-pivot`), then
3. reverted to GGG2–5 in this session, but with a different data/model design than either
   prior attempt.

This ADR records the reverted-to design and the reasons, so the history above is not repeated.
It **amends** ADR 0001 where noted and otherwise leaves it standing.

Two hard facts, measured directly from the data (not from any document), drove every decision
below:

- **Mask reality.** Of 425 PI-CAI positive (ISUP ≥2) cases, only 220 carry graded spatial masks
  (`csPCa_lesion_delineations/human_expert/original`, voxel values `{2,3,4,5}`). The other 205
  (`Pooch25`) and all 1,500 `AI/Bosma22a` masks are **binary** (`{0,1}`) — presence only, no grade.
  Per-grade file counts in the graded set: GGG2 135, GGG3 52, **GGG4 20, GGG5 18**.
- **Code reality.** Neither PDHD-Net nor GCALF-Net contains an FFT-based frequency module or a
  wired window-attention fusion. `grep -rn "torch.fft\|fftshift\|rfftn" nndet/` is empty in both
  repos; `modular.py:172` comments `移除FFT频域处理` ("FFT frequency-domain processing
  removed"); `WindowAttentionFusion` is imported at `modular.py:11` and never instantiated. The
  live encoder is Haar-wavelet fusion (`WaveletSpatialFusion`, stages `[1,3,4]`) plus ECA-style
  channel-light fusion (`MemoryEfficientFusion`/`ChannelWiseLightFusion`, stages `[2,5]`).

## Decisions

### D1 — GGG2–5 lesion-level grading, approved

The thesis protocol is GGG2–5 (four foreground grade classes), not GGG1–5 and not case-level
GGG1–5. Adviser/panel sign-off is recorded as obtained. GGG1 is not "excluded" from the study —
PI-CAI's own reference standard encodes ISUP ≤1 as background/non-csPCa, and GCALF-Net inherits
that definition. State the model as detecting **clinically significant PCa (csPCa, ISUP ≥2)** and
grading it GGG2–5, not as a five-class model with one class removed.

### D2 — Two-head model: csPCa detection (all 1,500 cases) + masked grade head (grade-supervised lesions only)

Supersedes ADR 0001 decision 1's implicit single-classifier framing and the four-class
native-instance design of commits `cf3390b`/`1b18cab`. The model has:

- **One detection foreground class, `csPCa`**, trained on all 425 human-annotated positives
  (220 graded + 205 Pooch25 binary) plus all 1,075 benign/GGG1 negatives. This is nnDetection's
  existing anchor objectness + box regression + segmentation path, unchanged, using its existing
  sigmoid focal loss (background imbalance is what focal loss exists for — do not use CE here).
- **A separate 4-logit grade head** over each matched positive detection, trained with
  class-weighted cross-entropy **only** on lesions with a resolved GGG2–5 label, masked to zero
  loss and zero gradient elsewhere. This is where the thesis's class-weighted CE requirement
  lives — on a small, class-imbalanced softmax, not on the anchor classifier.

Rejected: native `classifier_classes = 4` foreground instances (the shipped `cf3390b` design).
It requires dropping all 205 Pooch25 cases from detection training (no class to assign an
ungraded lesion to), discarding 48% of positive cases for a problem the masked head avoids
entirely.

Consequence: `dataset.json["labels"]` is `{"0": "csPCa"}` — the planner derives
`classifier_classes = 1` for the anchor head, not 4 or 5. The grade head is separate model code,
not something the nnDetection planner or plan pickle expresses. Grade labels travel as instance
metadata (`grade`, `grade_source`, `grade_supervised`), not as the instance's detection class.

### D3 — Unifocal linkage recovery audit, run once, before the endpoint is frozen

220 graded lesions is the floor, not necessarily the final number. Phase 1 runs a
connected-component audit over all 425 positive masks, cross-referenced against `marksheet.csv`
`lesion_ISUP`: any Pooch25 (binary) case where the marksheet lists **exactly one** lesion and the
binary mask has **exactly one** connected component receives that lesion's marksheet grade,
unambiguously — no inference, no pooling across components. Multifocal Pooch25 cases (multiple
components or multiple marksheet lesions) are **not** touched by this rule and stay
grade-unsupervised. Report the recovered count per grade before deciding between the plain
4-class endpoint and a pre-registered GGG4+5 merged secondary analysis (see D8).

### D4 — Build FDR (fixed) + wire WAF as the frozen control; LFF and CAF are single-variable changes on top of it

Amends ADR 0001's implicit assumption (and `SPEC.md §0`'s "two `ModuleList` slots" framing) that
FDR/WAF already exist. They do not, in either repository. The frozen baseline is:

- **FDR**: `fftn` → `fftshift` → fixed radial low-pass mask at `|f| ≤ 0.15` → complementary
  `X_low = M·X`, `X_high = (1-M)·X` → `ifftshift` → `ifftn`, replacing `WaveletSpatialFusion` at
  stages `[1,3,4]`.
- **WAF**: `WindowAttentionFusion` (already written, dead code in the released repo) instantiated
  and wired at stages `[2,5]`, replacing `MemoryEfficientFusion`.

This makes LFF (`§`, Phase 3) exactly FDR with a learned mask — one variable — and CAF (Phase 4)
exactly WAF with true bidirectional Q/K/V — one variable. The reported "baseline PDHD-Net" is
therefore the paper's architecture, adapted to 3-channel bpMRI, not the authors' released
`WaveletSpatialFusion`/`ChannelWiseLightFusion` code. State this plainly in the thesis methods
section; if FDR+WAF underperforms the released wavelet/channel-light code, report that finding
rather than switching baselines after the fact.

### D5 — Fusion at stages 2 and 5 only, identical across all four arms

Confirmed, not changed: matches the currently wired stages, avoids near-full-resolution 3D
cross-attention (the primary OOM risk per `SPEC.md §14`/ADR 0001 consequence 4), and adds the
fewest parameters to a model whose grade head is trained on 220–340 lesions. "Every corresponding
scale" in the phase docs means every scale that carries fusion, i.e. 2 and 5 — not all six
encoder stages. Fix the standing defect where `modular.py:158-165` constructs a fusion module for
every stage but invokes only 2 and 5 (dead parameters in every checkpoint) as part of the FDR/WAF
build in Phase 2.

### D6 — Preprocessing: N4 on T2W only; common grid; center-crop 640→256; fixed 3.0 mm slice spacing, pad/crop to 32; nnDetection plans on top

Supersedes the isotropic-1mm-then-crop-to-32 contract implied by the earlier `ROADMAP.md` T2 (that
contract truncates a ~50 mm gland to 32 mm of coverage after 1 mm resampling — a real defect
caught by the now-deleted `FEASIBILITY_REPORT.md §10.2`). The adopted contract:

1. **N4 bias correction on T2W only.** ADC is a quantitative diffusion map; N4 (built for
   receive-coil intensity bias in structural MRI) distorts its physical values. HBV is derived
   from the same acquisition as ADC and is treated the same way.
2. Resample T2W/ADC/HBV onto a **common reference grid** before any cropping — they ship at
   different native resolutions (T2W ~0.5 mm in-plane / ~640×640; ADC/HBV coarser) and "crop to
   256×256" is meaningless until they share a grid.
3. **In-plane: center-crop** (not resample) 640→256 around the prostate, using the whole-gland
   mask (`picai_labels/anatomical_delineations/whole_gland/`, present for all 1,500 cases) to
   center the crop. This preserves native ~0.5 mm in-plane detail instead of inventing or
   destroying it.
4. **Slice axis: resample to a fixed 3.0 mm spacing** (near-native for PI-CAI T2W, invents
   minimal through-plane detail), then pad/crop to 32 slices = 96 mm of coverage — comfortably
   more than a ~50 mm gland, and physically comparable across patients (unlike interpolating
   every patient's varying native slice count to a fixed count of 32, which gives every patient a
   different effective mm/slice and makes lesion extent in voxels incomparable across patients).
5. Per-case, per-modality z-score normalization, after steps 1–4.
6. **Masks always nearest-neighbour interpolation, never linear** — linear interpolation on
   `{0,2,3,4,5}`-valued masks produces non-integer values that silently fabricate nonexistent
   grade classes.
7. This geometry is the **raw nnDetection task input**, not a hand-planned final patch. Let
   `nndet_prep`'s planner choose target spacing and patch size from the data, as it is designed
   to and as `picai_baseline/nndetection_baseline.md`'s tested recipe assumes. Record the
   resolved plan values in the dataset manifest; do not fight the planner with a second, hidden
   resampling step.

### D7 — Statistics: keep the defended normality-gated test as primary, add patient-level paired bootstrap CIs

Shapiro-Wilk on 5 fold-level values has effectively no power to reject, so the defended
ANOVA/Tukey-or-Friedman/Wilcoxon route will almost always run through its non-normal branch. Run
it exactly as specified (no amendment needed) and additionally report patient-level paired
bootstrap confidence intervals and effect sizes for every comparison, resampling patients and
retaining all their lesions. This requires preserving out-of-fold predictions for every run
(already required by ADR 0001 decision 11's recovery contract; extend it to predictions, not only
checkpoints).

### D8 — Rare-grade endpoint: 4-class primary stays, decided after D3's audit

GGG4 (20 graded cases) and GGG5 (18) yield roughly 4 held-out cases per fold each before the D3
audit runs. Freeze the choice between (a) 4-class primary with per-grade CIs and a named sample-
size limitation, or (b) a pre-registered GGG4+5 merged secondary analysis, **after** D3's recovered
counts are in hand — not before. Do not add this decision as a fifth class or make it the primary
endpoint outright; it is a secondary analysis if added at all.

### D9 — Local execution: pinned `gcalf:m0` (CPU-only) for everything that must match the reported result, plus a separate modern-CUDA scratch environment for LFF/CAF math prototyping

The local RTX 4050 is `sm_89`; the pinned CUDA 11.3 toolchain builds through `sm_86`
(`docs/VAST_TESTING.md`), and it is 6 GB regardless. All gates that must match the cloud/reported
result — unit tests, config parsing, manifest/fold validation, real preprocessing, synthetic
CPU forward/backward, the 2-case CPU micro-overfit — run inside `gcalf:m0`, CPU-only. A second,
disposable modern-torch/CUDA environment (not the pinned stack, never installed into
`GCALF-Net/`'s reported results) may be used to prototype LFF/CAF tensor math (shape, gradient,
orientation-gate correctness) on the 4050 before porting the validated implementation into the
pinned environment. This does not modernize the reported runtime (ADR 0001's rejected
alternative "Modernize runtime first" still stands) — it is scratch space for math iteration only,
and nothing built there is installed or reported as the thesis result.

### D10 — Budget: $150–350, full 20-run matrix is the planned rung

The pre-committed ladder from ADR 0001 amendment A5 stands unchanged: full matrix → halve
`num_train_batches_per_epoch` for all 20 runs → 14-run reduced matrix. The full matrix
(≈250–420 GPU-hours at 3090/A5000 spot pricing) fits this budget with headroom for 2–3 failed
instance-hours and durable checkpoint storage. The fold-0 pilot measurement (Phase 2) is still
what triggers a step down the ladder — this decision fixes which rung is *planned*, not a
license to skip the pilot.

### D11 — Documentation and git hygiene

- `pdhd-upstream` tag placed at `e2330cf` (2025-11-29, "Update README: Change architecture to C +
  Swin Transformer + FFT + BiFPN"), the last commit before thesis work began at `2320bc9`
  (2026-07-30). `git diff pdhd-upstream` in `GCALF-Net/` is now the exact thesis changeset.
- The uncommitted GGG1–5 five-class rewrite (separate grade head design, deleted GGG2–5 tests,
  rewritten phase docs) is preserved on branch `ggg1-5-pivot`, not merged into `main`.
- `docs/SPEC.md`, the prior `docs/ROADMAP.md`, and the eight prior `docs/phases/PHASE_*.md` files
  are deleted, superseded by `docs/ARCHITECTURE.md`, the rewritten `docs/ROADMAP.md`, and the
  rewritten phase docs. `docs/GLOSSARY.md`, `docs/CLOUD_DEPLOYMENT_PLAN.md`,
  `docs/VAST_TESTING.md`, `docs/m0-verification.md`, `docs/data_report.md`, and this ADR set
  remain live. `docs/DATASCI17-THESIS-MASTERFILE.md` is known-stale (a newer version is pending)
  and is not used as a source of truth for any decision above.

## Consequences

- `gcalf_data/build_labels.py`, `gcalf_data/prepare_picai.py`, `gcalf_data/sanity_checks.py`, and
  `tests/gcalf/{test_data_preparation,test_data_sanity_checks}.py` currently implement the
  native-4-class design rejected by D2. They are **not rewritten by this ADR** — that is Phase 1
  implementation work — but they must not be treated as done; `docs/phases/PHASE_1_data_pipeline.md`
  records the gap explicitly.
- The grade head, its loss routing, and its masking logic are new model code with no prior
  implementation in either repository (unlike LFF/CAF, which at least have GFNet/DCA/UCTransNet
  references to adapt).
- FDR and WAF are also new code (D4); the integration surface is larger than ADR 0001 assumed.
- Every downstream evaluation artifact (confusion matrix, FROC, Grad-CAM target) must report the
  detection denominator (1,500 cases) and the grade-matched denominator (220–340 lesions)
  side by side, never collapse them into one accuracy number.

## Rejected Alternatives

| Alternative | Reason rejected |
|---|---|
| Native `classifier_classes = 4` (foreground-only GGG2–5 instances) | Forces dropping all 205 Pooch25 positives from detection training; no grade to assign an ungraded lesion. |
| Case-level GGG1–5 head | Reverses the approved GGG2–5 lesion-level amendment; discards the 178 marksheet cases with genuinely mixed-grade multifocal disease. |
| Keep released wavelet + channel-light as "the baseline" | Leaves the defended hypotheses (naming FDSF's fixed spherical mask and window-based self-attention) literally false; makes each ablation change two things at once (mechanism family and learnability). |
| Fusion at all six encoder stages | Matches phase-doc prose literally but is the main OOM risk in 3D and adds parameters a 220–340-lesion grade head cannot support. |
| 1 mm isotropic resampling + depth-32 crop | Truncates a ~50 mm gland to 32 mm of coverage; caught by measurement, not assumption. |
| Patient-level paired bootstrap as the sole/primary statistical test | Requires a protocol amendment beyond what was approved this session; kept as a secondary report alongside the defended test instead (D7). |
| GGG4+5 merge as the primary endpoint | A third protocol amendment; abandons the GGG4-vs-GGG5 distinction without exhausting the recovery audit (D3) first. |

## Review Triggers

Revisit this ADR if: the D3 unifocal audit recovers materially fewer lesions than expected and
the 4-class endpoint becomes non-viable even with CIs; the masked grade head cannot be trained
end-to-end after a real implementation attempt; FDR+WAF cannot be made to converge as a control
(forcing a return to the released wavelet+channel-light baseline, per ADR 0001's rejected
alternative reconsidered); or the fold-0 pilot forces a ladder step-down inconsistent with D10's
planned rung. Any revision must update this ADR, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, the
affected phase docs, and the thesis chapter together.
