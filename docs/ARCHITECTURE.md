# GCALF-Net — Architecture

**Thesis:** GCALF-Net: Modified PDHD-Net with Adaptive Frequency Filtering and Cross-Attention
Fusion for Gleason Grade Group Classification using bpMRI
**Protocol:** GGG2–5 lesion-level detection + grading, PI-CAI (bpMRI: T2W, ADC, HBV)
**Date:** 2026-09-05 (supersedes `SPEC.md`, deleted)

> This is the master "what and why." `docs/ROADMAP.md` is the milestone ledger with pass/fail
> gates; `docs/phases/PHASE_0…7.md` are the step-by-step execution plans; `docs/adr/0002-*.md`
> is the decision record this file implements — read the ADR first if you want the reasoning
> behind a decision rather than the decision itself.

---

## 0. Reality check — what the code actually contains

Verified directly against both `PDHD-Net/` (pristine) and `GCALF-Net/` (this repo). Neither
contains an FFT-based frequency module or a wired window-attention fusion.

| Thesis/paper says | Code actually has | Consequence |
|---|---|---|
| FDR: 3D FFT + fixed spherical low/high mask | `nndet/arch/encoder/WaveletFusion.py::WaveletSpatialFusion` — Haar **wavelet**, not FFT. `torch.fft`/`fftshift`/`rfftn` appear nowhere in `nndet/`. `modular.py:172` comments `移除FFT频域处理` ("FFT frequency-domain processing removed"); `fft_low_ratio` at `modular.py:62` is a dead parameter. | **FDR must be built.** This is not a config flag away — it is new code (§5). |
| WAF: window self-attention fusion | `nndet/arch/encoder/window_attention_fusion.py::WindowAttentionFusion` exists, fully written, and is **imported at `modular.py:11` but never instantiated**. Live fusion is `MemoryEfficientFusion`→`ChannelWiseLightFusion` (ECA-style channel attention + grouped conv), stages `[2,5]` only. | **WAF must be wired.** The class exists; it just needs to replace `ChannelWiseLightFusion` in the module list. |
| 5-class GGG1–5 lesion grading | PI-CAI's expert masks encode ISUP ≤1 as background (value 0), indistinguishable from benign tissue. Only 220 of 1,500 cases carry graded spatial masks (`{2,3,4,5}`); the other 205 positive cases (`Pooch25`) and all 1,500 `Bosma22a` AI masks are binary `{0,1}`. | **GGG2–5 is the ceiling, not a simplification** (ADR 0002 D1). A GGG1 class cannot be built from this data without inventing labels. |
| One classifier head over `classifier_classes` foreground grades | Native instance classes would require dropping every ungraded positive lesion (all 205 Pooch25 cases) — no class to assign them. | **Two-head design**: one detection foreground class (`csPCa`, all 425 positives + 1,075 negatives) plus a separate 4-logit grade head trained only on grade-resolved lesions (ADR 0002 D2). |
| Fusion "at every corresponding scale" | Live fusion wiring is stages `[2,5]` of 6; `modular.py:158-165` constructs a fusion module for **every** stage but invokes only 2 and 5 — four modules of dead parameters in every checkpoint. | Fusion stays at `[2,5]` (ADR 0002 D5); fix the dead-parameter construction while building FDR/WAF. |
| 1000-epoch training | `nndet/conf/train/v001.yaml`: `max_num_epochs: 50`, `num_train_batches_per_epoch: 2500`, `swa_epochs: 10`. No `EarlyStopping` exists or may be added. | ≈150k optimizer steps per run fixes the compute budget (§9). |

**Bottom line.** The integration surface is larger than a naive reading of the paper suggests:
FDR (new), WAF (wire existing), a grade head (new, no upstream reference), LFF (new, built on
FDR), CAF (new, built on WAF). Budget for five new modules, not two `ModuleList` swaps.

---

## 1. Repository role mapping

| Repo | Role | What to do with it |
|---|---|---|
| **GCALF-Net** | Main working repository. Copy of `PDHD-Net/` at tag `pdhd-upstream` (= commit `e2330cf`), upstream remote removed. | Build here. `git diff pdhd-upstream` is the thesis changeset. |
| **PDHD-Net** | Released reference, kept pristine. | Read-only. Never edit; never install alongside GCALF-Net (`nndet` package name collides). |
| **GFNet** | Reference (learnable Fourier filter). | Copy `GlobalFilter`'s idea (2D) into a new 3D module (§6). |
| **TransFuse** | Fusion fallback/reference. | `BiFusion_block` is a named, non-cross-attention fallback only (§7 fallback ladder). |
| **Dual-Cross-Attention / UCTransNet** | Secondary CAF references. | Read for the Q/K/V channel cross-attention math; reimplement 3D, don't import (both are 2D). |
| **M3d-Cam (`medcam`)** | Tool. | `medcam.inject(model, layer=…)` for Grad-CAM (§8). |
| **Z-SSMNet** | Reference only. | Cross-check PI-CAI prep and official splits; do not install (fights PDHD-Net's pins). |
| **picai_prep / picai_eval** | Tools. | Install as editable packages. |
| **picai_labels** | Data. | `clinical_information/marksheet.csv`, `csPCa_lesion_delineations/`, `anatomical_delineations/whole_gland/`. |
| **picai_baseline** | Reference recipe. | Follow `nndetection_baseline.md` for the PI-CAI → nnDetection conversion steps. |

**Environment anchor:** PDHD-Net's `requirements.txt` pins a 2021-era stack
(`pytorch_lightning<=1.4.2`, `nnunet==1.7.1`, `SimpleITK<2.1.0`) → torch ~1.10/CUDA 11.3. This is
fixed; do not modernize it mid-thesis (it would confound baseline-vs-GCALF attribution — ADR 0001).
GFNet/TransFuse/UCTransNet/DCA classes are copied in, never `pip install`ed. Z-SSMNet is read-only.

---

## 2. Project structure

```
GCALF-Net/
├── nndet/arch/encoder/
│   ├── modular.py                     # the ONE integration file
│   ├── gcalf/
│   │   ├── fdr.py                     # NEW: fixed FFT frequency decomposition
│   │   ├── waf.py                     # wires the existing WindowAttentionFusion
│   │   ├── lff.py                     # NEW: FDR + learned mask
│   │   ├── caf.py                     # NEW: WAF + true bidirectional Q/K/V
│   │   ├── grade_head.py              # NEW: masked 4-logit GGG2-5 head
│   │   └── registry.py                # build_frequency_module(...), build_fusion_module(...)
│   ├── WaveletFusion.py               # keep, reference-only baseline no longer used
│   ├── channel_lightweight_fusion.py  # keep, reference-only fusion no longer used
│   └── window_attention_fusion.py     # WAF implementation, now wired via gcalf/waf.py
├── gcalf_configs/{baseline,lff_only,caf_only,gcalf_full}.yaml
├── gcalf_data/                        # PI-CAI prep, label linkage, sanity checks (rework needed — Phase 1)
├── gcalf_eval/                        # picai_eval wrapper, grade metrics, collect_results, gradcam
├── gcalf_experiments/                 # outputs, gitignored
├── cloud/vast/                        # Vast.ai host/bootstrap/download/export scripts
└── scripts/{train,predict,preprocess,run_m3_smoke.sh}
```

`modular.py` never names a concrete module. It calls `registry.build_frequency_module(...)` and
`registry.build_fusion_module(...)`, selected through Hydra `model_cfg.encoder_kwargs.gcalf_cfg`.
One code path, config-selected modules — see §4.

---

## 3. Data contract

**Ground truth** (`marksheet.csv`, 1,500 rows; measured directly, not from documentation):

```
case_ISUP:   {0: 847, 1: 228, 2: 234, 3: 99, 4: 40, 5: 52}
lesion_ISUP: {0: 592, 1: 311, 2: 260, 3: 109, 4: 41, 5: 55}   (comma-separated per case)
```

**Mask reality** (measured by decoding voxel data directly):

| Annotation set | Cases | Voxel values | Grade-usable |
|---|---|---|---|
| `human_expert/original` (+ `resampled`) | 1,295 (220 non-empty) | `{0,2,3,4,5}` | **Yes** |
| `human_expert/Pooch25` | 205 (all non-empty) | `{0,1}` | No — binary |
| `AI/Bosma22a` | 1,500 (408 non-empty) | `{0,1}` | No — binary |

287 of 1,500 marksheet rows are multifocal (comma-separated `lesion_ISUP`); of those, 178 have
genuinely different grades across lesions in the same case — real information a case-level label
would discard. This is why the prediction unit is the lesion, not the case.

**Cohort roles** (all 1,500 cases train the detector; only graded lesions train the grade head):

| Group | n | Detection role | Segmentation | Grade head |
|---|---|---|---|---|
| Benign (ISUP 0) + GGG1 (ISUP 1) | 1,075 + 228 | negative | negative | masked out |
| Positive, graded (`human_expert`) | 220 (+ audit recovery, §Phase 1) | positive | positive | **trained** |
| Positive, binary only (`Pooch25`) | 205 | positive | positive | masked out |

**Label schema** (per case, nnDetection instance format):

```json
{
  "instances": {
    "1": {"class": 0, "grade": 2, "grade_source": "human_expert/original", "grade_supervised": true},
    "2": {"class": 0, "grade_supervised": false}
  }
}
```

`dataset.json["labels"] = {"0": "csPCa"}` — one detection foreground class. The planner derives
`classifier_classes = 1` for nnDetection's own anchor head. Grade is carried as instance metadata,
consumed only by the separate grade head (§4.3), never by the planner.

**Preprocessing** (ADR 0002 D6 — nearest-neighbour for every mask, never linear):

1. N4 bias correction on **T2W only** (ADC/HBV are quantitative maps; N4 distorts their values).
2. Resample T2W/ADC/HBV onto a common reference grid (they ship at different native resolutions).
3. In-plane: **center-crop** 640→256 around the prostate, centered using the whole-gland mask
   (`anatomical_delineations/whole_gland/`, present for all 1,500 cases). No in-plane resampling —
   this preserves native ~0.5 mm detail.
4. Slice axis: resample to a **fixed 3.0 mm spacing**, then pad/crop to 32 slices (96 mm coverage).
   Fixed spacing, not fixed slice count, keeps lesion extent in voxels comparable across patients.
5. Per-case, per-modality z-score normalization.
6. Emit as the nnDetection **raw** task; let `nndet_prep`'s planner choose target spacing and patch
   size on top (do not hand-plan a second time — follow `picai_baseline/nndetection_baseline.md`).

**Splits:** official PI-CAI patient-disjoint 5-fold splits (`picai_baseline`/`Z-SSMNet`), verified
independently for no patient leakage and every grade present in every held-out fold.

---

## 4. Modular design for ablation

```yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: fdr       # fdr | lff
      fusion_type: waf                 # waf | caf
      freq_stages: [1, 3, 4]
      fusion_stages: [2, 5]
      fdr: {radius: 0.15}
      lff: {grid_size: [4, 8, 8], groups: 8}
      waf: {window_size: [2, 7, 7], num_heads: 4}
      caf: {window_size: [2, 7, 7], num_heads: 4, dropout: 0.0}
```

| Config | `frequency_filter_type` | `fusion_type` |
|---|---|---|
| Baseline (built FDR+WAF) | `fdr` | `waf` |
| LFF only | `lff` | `waf` |
| CAF only | `fdr` | `caf` |
| Full GCALF-Net | `lff` | `caf` |

One `Encoder` class; `nndet/arch/encoder/gcalf/registry.py` is the only string→module mapping.
Same experiment-directory convention as before: `gcalf_experiments/<config>_seed<NN>/` with
`checkpoints/`, `logs/metrics.csv`, `predictions/`, `config_snapshot.yaml`, `git_commit.txt`.

---

## 5. FDR (fixed) and the built baseline

**Purpose:** the frozen control. Replaces `WaveletSpatialFusion` at stages `[1,3,4]`.

**Definition:**
1. `X = fftn(x, dim=(-3,-2,-1))`, then `fftshift` to center frequencies.
2. Normalized radial frequency coordinates `r = sqrt((d/D)² + (h/H)² + (w/W)²)` on the centered grid.
3. Fixed spherical mask `M = 1[r ≤ 0.15]`.
4. `X_low = M·X`, `X_high = (1-M)·X`.
5. `ifftshift` each, then `ifftn` back to spatial domain.
6. Route `X_low` toward the Swin branch, `X_high` toward the CNN branch (matching the paper's
   frequency-routing claim — verify against the thesis's exact routing direction before coding).

Run FFT in fp32 (`torch.cuda.amp.autocast(enabled=False)` around the FFT block, as in GFNet).

**Unit tests:** mask radius is exact at the boundary; `X_low + X_high` reconstructs `X` (aliasing-
free split); shape preserved for odd/even `(D,H,W)`; a constant input passes almost entirely
through `X_low`; a Nyquist-checkerboard input passes almost entirely through `X_high` (this is the
orientation gate — it catches a missing `fftshift`/`ifftshift` pair, the same defect class flagged
for LFF below).

---

## 6. Learnable Frequency Filter (LFF)

**Purpose:** replace FDR's fixed mask with one learned, shared, real-valued gain — the *only*
change from §5, so LFF vs. baseline is a single-variable ablation.

**Reference:** `GFNet/gfnet.py::GlobalFilter`, ported 2D→3D.

**Design corrections (ADR 0001 amendments A2/A3, still binding):**
- **Centered frequency coordinates.** `rfftn` orders the two full axes `[0,+f,…,Nyquist,−f,…,−1]`;
  interpolating a coarse learned grid directly over that raw index space cannot express a
  low-pass response and smooths across the Nyquist discontinuity. Define the grid on centered
  coordinates and `ifftshift` the interpolated result along `D,H` before multiplying.
- **Real-valued gain, not complex.** `x` real ⟹ `X` Hermitian ⟹ `irfftn` applies the Hermitian
  projection `(H(f)+conj(H(−f)))/2` on a non-Hermitian complex gain — half the learned parameters
  alias onto the other half and are not identifiable. A real gain is exactly Hermitian, halves the
  parameter count, stays frequency-selective, and needs no orientation correction on the trailing
  rFFT axis (it already runs DC→Nyquist monotonically).
- Parameterize as `H = 1 + delta_H`, `delta_H` zero-initialized (exact identity at start, no
  separate spatial residual — an all-ones filter *plus* a residual returns ≈2x, not x).

```python
class LearnableFrequencyFilter3D(nn.Module):
    """Grouped, shape-tolerant, zero-phase, real-valued learnable frequency filter."""

    def __init__(self, in_channels, out_channels, grid_size=(4, 8, 8), groups=8):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        self.in_channels = in_channels
        self.groups = groups
        self.delta_weight = nn.Parameter(torch.zeros(1, groups, *grid_size))  # real, zero-init
        self.proj = (nn.Identity() if in_channels == out_channels
                     else nn.Conv3d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            spectrum = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
            delta = F.interpolate(self.delta_weight, size=spectrum.shape[-3:],
                                   mode="trilinear", align_corners=True)
            delta = torch.fft.ifftshift(delta, dim=(-3, -2))  # centered grid -> rfftn ordering
            gain = (1.0 + delta).repeat_interleave(self.in_channels // self.groups, dim=1)
            filtered = torch.fft.irfftn(spectrum * gain, s=x.shape[-3:],
                                         dim=(-3, -2, -1), norm="ortho")
        return self.proj(filtered.to(input_dtype))
```

**Unit tests** (`tests/gcalf/test_lff.py`): shape in == shape out across several `(D,H,W)`;
`delta_weight=0` ⟹ output ≈ input (`atol=1e-4`); gradient reaches `delta_weight`; **orientation
gate** — a grid that is 1 at centre, −1 elsewhere must pass a constant volume through and suppress
a Nyquist checkerboard (inverts under a missing `ifftshift` — the single most valuable test in the
file); finite under autocast on CPU and CUDA; a fixed-seed regression test proving the FDR
baseline's output is byte-identical after LFF's registry refactor.

---

## 7. WAF (wired) and Cross-Attention Fusion (CAF)

**WAF** (baseline fusion, stages `[2,5]`): instantiate the existing
`nndet/arch/encoder/window_attention_fusion.py::WindowAttentionFusion` in place of
`MemoryEfficientFusion`. It already implements shifted-window self-attention; the work here is
wiring, not writing.

**CAF** (replaces WAF, the second single-variable change): true bidirectional Q/K/V cross-attention
— CNN windows query Swin keys/values, Swin windows query CNN keys/values — where WAF has
self-attention within one aligned feature. `TransFuse::BiFusion_block` (gating + Hadamard) is a
named non-cross-attention fallback only, never relabeled CAF.

```python
class WindowedCrossAttentionFusion3D(nn.Module):
    def __init__(self, cnn_channels, transformer_channels, out_channels,
                 window_size=(2, 7, 7), num_heads=4, dropout=0.0):
        super().__init__()
        if out_channels % num_heads != 0:
            raise ValueError("out_channels must be divisible by num_heads")
        self.window_size = tuple(window_size)
        self.cnn_align = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        self.swin_align = nn.Conv3d(transformer_channels, out_channels, 1, bias=False)
        self.cnn_queries_swin = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout, batch_first=True)
        self.swin_queries_cnn = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(out_channels)
        self.out_proj = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat, swin_feat):
        cnn = self.cnn_align(cnn_feat)
        swin = self.swin_align(swin_feat)
        cnn_w, pad_mask, meta = window_partition_3d(cnn, self.window_size)
        swin_w, _, _ = window_partition_3d(swin, self.window_size)
        cnn_cross, _ = self.cnn_queries_swin(cnn_w, swin_w, swin_w, key_padding_mask=pad_mask, need_weights=False)
        swin_cross, _ = self.swin_queries_cnn(swin_w, cnn_w, cnn_w, key_padding_mask=pad_mask, need_weights=False)
        fused_w = self.norm(cnn_cross + swin_cross)          # attention terms only
        fused = window_reverse_3d(fused_w, meta)
        return self.out_proj(torch.cat([cnn, fused], dim=1)) + cnn   # residual counted ONCE
```

**Two rules that must hold, both non-obvious and both caught only by the tests below:**
- **The aligned CNN feature reaches the output through exactly one path** (the concat + trailing
  `+ cnn`). Also adding it inside the attention sum triples it.
- **Padding is masked, not attended as real keys.** `window_size=(2,7,7)` rarely divides stage
  feature maps evenly; unmasked zero-padding lets the model learn an input-size-dependent bias.
  Both `MultiheadAttention` calls take `key_padding_mask`; a window that is entirely padding
  produces NaN under softmax masking — drop such windows or discard their result on reverse.

**Known simplification:** no relative position bias, no shifted-window pass — attention is
permutation-invariant inside a window and carries no cross-window information. Record as an
architecture limitation; the first upgrade if CAF underperforms for reasons other than memory.

**Unit tests** (`tests/gcalf/test_caf.py`): window partition/reverse round-trips exactly for
divisible and padded shapes; padding invariance (refilling padded region with a different constant
leaves the valid region's output unchanged); no NaN from fully-padded windows; residual counted
once (zero both attention outputs + identity `out_proj` ⟹ output == aligned CNN feature exactly);
gradients reach both alignment convs and both attention projections; CUDA peak memory at real
stage-2/stage-5 shapes passes a declared gate.

**Fallback ladder** (apply in order, record any architecture change): reduce window to `(2,4,4)`;
activation checkpointing around CAF; true CAF at stage 5 only; TransFuse BiFusion as a separately
named fallback experiment. Never silently relabel BiFusion as CAF.

---

## 8. Grade head, loss routing, and inference

**Grade head:** a small 4-logit head reading the feature of each matched positive detection
(same attachment point nnDetection already uses for its own classifier — reuse that plumbing,
add a parallel head rather than reusing the anchor classifier's channel count).

**Loss:**
```text
total_loss = objectness_loss + box_loss + segmentation_loss + grade_loss
```
- `objectness_loss`, `box_loss`, `segmentation_loss`: nnDetection's existing sigmoid focal /
  regression / Dice-CE paths, unchanged, over the single `csPCa` class, on all 1,500 cases.
- `grade_loss`: class-weighted cross-entropy over the 4 grade logits, computed **only** for
  matched positive detections with `grade_supervised: true`; masked (zero loss, zero gradient)
  for every other detection, including true positives on ungraded lesions and all negatives.
  Derive class weights from each fold's **training partition only**, normalize to mean 1.
  Log the number of grade-supervised lesions contributing to every batch — a batch with zero
  such lesions is a legitimate, expected state, not a bug.

**Inference:** detect `csPCa` first (standard nnDetection prediction), then assign the
highest-probability GGG2–5 grade to each retained detection via the grade head. Every detection
gets a grade, including false positives on benign cases — those grades are not meaningful and
must not be included in grade accuracy; report the detection denominator (all retained detections)
and the grade-matched denominator (detections matched to a grade-supervised ground-truth lesion)
side by side (§10).

---

## 9. Training plan

- **Loop:** nnDetection's PyTorch-Lightning `RetinaUNetModule`, unchanged, plus the grade head's
  loss term wired in alongside the existing detection/segmentation losses.
- **Schedule** (authoritative — the config, not any README): `nndet/conf/train/v001.yaml`:
  `max_num_epochs: 50`, `num_train_batches_per_epoch: 2500`, `swa_epochs: 10`, SGD
  `initial_lr: 0.01` with poly decay, `precision: 16`. ≈150k optimizer steps per run. No
  `EarlyStopping` — none may be added; a per-config stopping rule would invalidate the four-way
  comparison. `nndet/conf/train/smoke.yaml` (2 epochs × 20 iters + 2 SWA) is the tiny-run config.
- **Compute budget:** 20 runs (4 configs × 5 folds × 1 primary seed) ≈ 250–420 GPU-hours at the
  shipped schedule. The fold-0 pilot measures seconds/step and selects one of three pre-committed
  ladder rungs (full matrix → halved batches-per-epoch, all 20 runs → 14-run reduced matrix,
  never dropping folds for only some configs) — **choose before launching, not after seeing
  results.** Planned rung: full matrix, within a $150–350 budget (ADR 0002 D10).
- **AMP:** everywhere except the FFT block in FDR/LFF (`autocast(enabled=False)` around it).
- **Reproducibility:** fixed seed; snapshot `config.yaml`, `git rev-parse HEAD`, `pip freeze`,
  dataset-manifest SHA-256, and split-file SHA-256 into every `gcalf_experiments/<exp>/`.
  Preserve out-of-fold predictions for every run (needed for §10's paired bootstrap CIs).

---

## 10. Evaluation plan

- **Detection (all 1,500 cases):** `picai_eval` FROC + lesion-level AUROC + case-level AUROC
  (max lesion confidence per case — the standard PI-CAI benign-vs-csPCa endpoint, free once
  detection works).
- **Grading (grade-supervised lesions only):** 4×4 confusion matrix over matched, grade-supervised
  lesions; report missed lesions (unmatched ground truth, per grade) and false positives (split
  benign vs. positive case) **beside** the matrix, never folded into it. Weighted F1 (primary,
  per the thesis), macro-F1, per-grade sensitivity/precision, quadratic-weighted Cohen's κ
  (grades are ordinal), one-vs-rest AUROC per grade + macro. Report 95% bootstrap CIs everywhere —
  mandatory given GGG4/GGG5 counts.
- **Segmentation:** Dice vs. lesion delineations, for matched positive detections.
- **Statistics** (ADR 0002 D7): Shapiro-Wilk → one-way repeated-measures ANOVA + Tukey HSD if
  normal, else Friedman + Bonferroni-corrected Wilcoxon (α/6 = 0.00833), on fold-level scores —
  exactly as the thesis specifies. **Additionally**, patient-level paired bootstrap CIs and paired
  permutation tests on preserved out-of-fold predictions, resampling patients and keeping all
  their lesions together. Report both; the bootstrap numbers are the honest uncertainty estimate
  the 5-point defended test cannot provide.
- **Ablation table:** rows = {baseline, LFF-only, CAF-only, full}, baseline as reference column,
  Δ per metric, identical folds/seeds across all four.

---

## 11. Grad-CAM / explainability

Use `medcam` (M3d-Cam), not a hand-rolled implementation — `medcam.inject(model, layer=…,
label=…, backend='gcam')`. Attach to the last conv feeding the **grade head** (§8), targeting the
predicted grade of a matched positive detection via `label=`. Because the grade head has a clean
per-detection 4-logit output (not the anchor classifier's per-anchor score), there is no
per-anchor `label`-selection ambiguity to work around — this design avoids the classifier-fallback
detour ADR 0001/`SPEC.md §14` needed under the native-class design.

Overlay on T2W; export blinded review packets (no GT/prediction label visible) for the 30-case,
GGG2–5-stratified, 3-urologist PI-RADS v2 Likert review.

---

## 12. Local testing / development

**Two environments, never merged (ADR 0002 D9):**
1. **`gcalf:m0`, pinned, CPU-only.** Every gate whose result must match what gets reported: unit
   tests, config parsing, module-selection assertions, manifest/fold validation, real
   preprocessing on fold-0 cases, synthetic CPU forward/backward at `(1,3,32,256,256)`, the
   2-case CPU micro-overfit. `local-only`/`L` gates in the roadmap and phase docs run here.
2. **A disposable modern-torch/CUDA scratch environment**, for prototyping LFF/CAF tensor math
   (shape, gradient, orientation-gate correctness) on the local RTX 4050 before porting the
   validated implementation into the pinned stack. Nothing here is installed into `GCALF-Net/`'s
   dependencies or reported as a thesis result — it is scratch space for iteration speed only.

**Why CPU-only for the pinned stack:** the local GPU is `sm_89`; the pinned CUDA 11.3 toolchain
builds through `sm_86` (`docs/VAST_TESTING.md`), and it has 6 GB regardless. Every CUDA-dependent
gate — the `nndet/csrc` extension build, `tests/test_csrc_cuda.py` (never yet run, per
`docs/m0-verification.md`), CAF/FDR CUDA memory profiling — is a cloud (`V`) gate, not local.

## 13. Cloud testing / deployment

Authoritative runbook: `docs/CLOUD_DEPLOYMENT_PLAN.md`. Pinned legacy Docker image on a
provider-neutral NVIDIA GPU VM, S3-compatible storage via AWS CLI v2, immutable dataset manifests,
data staged to local NVMe before training, checkpoint/log sync after every epoch. The budget
ladder is in §9; the pre-committed rungs live in `docs/CLOUD_DEPLOYMENT_PLAN.md` and are
unchanged by this document.

---

## 14. Risk assessment & fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| FDR/WAF build takes longer than budgeted | High | This is the largest new-code item in the plan (§0); if it slips past its phase gate, step down the budget ladder rather than compressing later phases. |
| Grade head does not learn on 220–340 lesions | Medium | Widen with the D3 unifocal-recovery audit before concluding the head is broken; report per-grade CIs regardless. |
| `nnDetection csrc`/old-torch env won't build | High | Docker image with the exact pinned base; CPU fallback for NMS in smoke tests; this is the #1 environment blocker historically. |
| GPU memory (3D + attention) | High | Smaller CAF/WAF windows, activation checkpointing, stage-5-only true attention, batch 1 + accumulation, AMP everywhere except FFT. |
| GGG4/5 tiny even after D3 audit | Certain (floor is 20/18) | Report CIs; consider the pre-registered GGG4+5 merged secondary analysis (ADR 0002 D8) once audit counts are known. |
| Full 20-run matrix over budget | Med | Pre-committed ladder (§9); never drop folds for only some configs. |

---

### Appendix — files to touch vs. reference

**Create:** `nndet/arch/encoder/gcalf/{fdr,waf,lff,caf,grade_head,registry}.py`,
`gcalf_configs/*.yaml`, `tests/gcalf/test_{fdr,waf,lff,caf,grade_head}.py`.
**Edit:** `nndet/arch/encoder/modular.py` (registry calls), `gcalf_data/*` (rework for the D2
two-head data contract — see `docs/phases/PHASE_1_data_pipeline.md`).
**Adapt from:** `GFNet/gfnet.py::GlobalFilter` (§6), DCA/UCTransNet (§7 Q/K/V math).
**Wire, don't rewrite:** `nndet/arch/encoder/window_attention_fusion.py::WindowAttentionFusion`.
**Keep untouched, reference-only:** `WaveletFusion.py`, `channel_lightweight_fusion.py`,
`swimTransformer.py`, `decoder/BiFPN.py`.
**Install as tools:** `picai_prep`, `picai_eval`, `medcam`. **Reference-only:** `Z-SSMNet`.
