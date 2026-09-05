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
| FDSF: input-level 3D FFT + fixed spherical low/high mask and branch shunting | `nndet/arch/encoder/WaveletFusion.py::WaveletSpatialFusion` — Haar **wavelet**, not FFT. `torch.fft`/`fftshift`/`rfftn` appear nowhere in `nndet/`. `modular.py:172` comments `移除FFT频域处理` ("FFT frequency-domain processing removed"); `fft_low_ratio` at `modular.py:62` is a dead parameter. | **FDSF must be built.** This is not a config flag away — it is new code (§5). |
| WAF: window self-attention fusion | `nndet/arch/encoder/window_attention_fusion.py::WindowAttentionFusion` exists, fully written, and is **imported at `modular.py:11` but never instantiated**. Live fusion is `MemoryEfficientFusion`→`ChannelWiseLightFusion` (ECA-style channel attention + grouped conv), stages `[2,5]` only. | **WAF must be wired.** The class exists; it just needs to replace `ChannelWiseLightFusion` in the module list. |
| 5-class GGG1–5 lesion grading | PI-CAI's expert masks encode ISUP ≤1 as background (value 0), indistinguishable from benign tissue. Only 220 of 1,500 cases carry graded spatial masks (`{2,3,4,5}`); the other 205 positive cases (`Pooch25`) and all 1,500 `Bosma22a` AI masks are binary `{0,1}`. | **GGG2–5 is the ceiling, not a simplification** (ADR 0002 D1). A GGG1 class cannot be built from this data without inventing labels. |
| One classifier head over `classifier_classes` foreground grades | Native instance classes would require dropping every ungraded positive lesion (all 205 Pooch25 cases) — no class to assign them. | **Two-head design**: one detection foreground class (`csPCa`, all 425 positives + 1,075 negatives) plus a separate 4-logit grade head trained only on grade-resolved lesions (ADR 0002 D2). |
| Five paper encoder levels and WAF at corresponding scales | Level count is planner-derived (`modular.py:63`; `c002.py:196-204`) and can be five or six. `modular.py:192` fuses at `[2,5]`, while `BiFPN.py:224-225` unpacks `p3..p7, _` — it **drops the sixth input**. So on a six-level plan the stage-5 fusion output is exactly the one the decoder throws away, and the only wired fusion that reaches the head is stage 2; on a five-level plan stage 5 never executes at all. | Freeze the level count; make WAF/CAF locations valid and identical; require every encoder output to reach BiFPN (ADR 0002 D5). |
| 1000-epoch training | `nndet/conf/train/v001.yaml`: `max_num_epochs: 50`, `num_train_batches_per_epoch: 2500`, `swa_epochs: 10`. No `EarlyStopping` exists or may be added. | ≈150k optimizer steps per run fixes the compute budget (§9). |

**Bottom line.** The integration surface is larger than a naive reading of the paper suggests:
FDSF (new), WAF (wire existing), a grade head (new, no upstream reference), LFF (new, built on
FDSF), CAF (new, built on WAF). Budget for five new modules, not two `ModuleList` swaps.

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
│   │   ├── fdsf.py                    # NEW: input-level fixed FFT separation and shunting
│   │   ├── waf.py                     # wires the existing WindowAttentionFusion
│   │   ├── lff.py                     # NEW: FDSF fixed response replaced by learned response
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
| Benign (ISUP 0) + GGG1 (ISUP 1) | 847 + 228 = 1,075 | negative | negative | masked out |
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

**Preprocessing** (ADR 0002 D6 — identical at training, validation, test, and deployment):

1. N4 bias correction on **T2W only**. ADC is quantitative; HBV is a diffusion-weighted magnitude
   or derived image, not a quantitative map in the same sense, but both remain outside N4 by
   protocol.
2. Build **one** reference grid from the corrected T2W: native in-plane geometry, **3.0 mm along
   the slice axis**. Resample ADC/HBV onto it with **linear** interpolation and both masks with
   nearest-neighbour, in a single pass. Never interpolate a mask linearly. Do not use a
   cubic/B-spline kernel on ADC or HBV — it overshoots at edges and can produce out-of-range or
   negative values on the same quantitative map step 1 declines to N4. Folding the slice spacing
   into this grid costs ADC/HBV one interpolation instead of two.
3. Validate the resampled whole-gland mask, then derive the in-plane crop centre from its centroid.
   **The crop is a fixed 128 mm physical field of view, not a fixed voxel count** (see the FOV note
   below), so its voxel extent varies per case. It stays a pure index operation, preserving native
   T2W in-plane spacing. An empty or implausible gland mask uses a predeclared, target-independent
   fallback (the T2W geometric centre) and is recorded.
4. **The lesion mask never selects or changes the crop.** It is unavailable at inference and may
   be used only after the crop for retention QC. Union- and lesion-centred fallbacks are prohibited.
5. Pad/crop the slice axis to 32 slices about the **geometric** centre — 96 mm of coverage at the
   3.0 mm spacing already fixed in step 2. Pad with **zeros**; the value is not cosmetic (step 8).
   Geometric depth centring remains the adopted rule while the exhaustive positive-case audit shows
   no additional depth clipping; changing to gland-z centring requires a new full-cohort audit.
6. For every positive case, record lesion voxels and connected components before and after the
   in-plane and depth operations. A non-empty lesion becoming empty is a hard QC failure; partial
   clipping is retained and reported under the predeclared exclusion rule below, never hidden by a
   label-guided crop.
7. Emit as the nnDetection **raw** task. **This geometry is not the model input.** `nndet_prep`
   applies three further transforms: `crop_to_nonzero` (`nndet/io/crop.py:288`) trims the zero
   padding back off per case, the planner resamples to its own target spacing, and training
   extracts patches. What the contract guarantees downstream is therefore the 3.0 mm slice spacing
   (so the planner's target z spacing is 3.0 mm and no second z resample occurs), a gland-centred
   field of view of constant physical size, and exactly one normalization pass — not a literal
   fixed-shape array arriving at the network.
8. **Normalize once:** do not z-score in the raw-task builder. Let nnDetection's `nonCT` scheme
   perform per-case, per-modality zero-mean/unit-variance normalization after its planned
   resampling. Step 5's padding value drives which scheme resolves:
   `determine_whether_to_use_mask_for_norm` (`nndet/planning/experiment/base.py:287-312`) enables
   the nonzero mask only when the median `crop_to_nonzero` size reduction is `< 3/4`, and that
   decides whether `normalize_other` z-scores over the nonzero region or the whole volume. Record
   the resolved scheme, target spacing, and patch size.

**Why a physical FOV and not 256 voxels.** T2W in-plane spacing measured over a 600-case sample of
the PI-CAI archive runs **0.234–0.625 mm** (modes 0.500 mm, n=273; 0.300 mm, n=166). A fixed
256-voxel crop therefore spans 60 mm of anatomy on one scanner and 160 mm on another — the same
defect step 5 avoids on the slice axis by fixing spacing rather than slice count. It is also,
measurably, what produced every retention exception in the 2026-09-05 audit: all four are
fine-spacing cases and none is at 0.5 mm or coarser.

| Case | In-plane spacing | FOV under a 256-voxel crop | Retention under that crop |
|---|---|---|---|
| `11280_1001303` | 0.234 mm | 60.0 mm | partial clip (58 / 6,406 voxels) |
| `11050_1001070` | 0.281 mm | 72.0 mm | **all 3,472 lesion voxels lost** |
| `10956_1000975` | 0.300 mm | 76.8 mm | partial clip (469 / 107,178) |
| `11174_1001197` | 0.342 mm | 87.6 mm | partial clip (40 / 32,365) |
| *modal case* | 0.500 mm | 128.0 mm | retained |

At 128 mm every case receives the coverage the modal case already had. That audit also found all
1,500 Bosma22b gland masks non-empty, and 421/425 positives fully retained. **These per-case counts
have not been re-measured under the 128 mm rule**; until `gcalf_data/audit_crop_retention.py` is
committed and run (§Phase 1), read the table as the evidence for the rule change and the counts as
pending re-audit.

**Predeclared exclusion rule** (fixed now, before the re-audit reports its numbers): a positive
case is excluded from all four ablation arms if a gland-centred 128 mm crop retains no lesion voxel
for a grade-supervised lesion. Partial clipping is retained and reported, never repaired by moving
the crop. Exclusions are frozen into the shared case set and reported once, with their fold.

**The gland mask is a deployment dependency.** The crop centre comes from the Bosma22b whole-gland
mask (`gcalf_data/prepare_picai.py:26` — the PI-CAI maintainers' own AI segmentation), which does
not exist for an unseen case. Running this pipeline at inference therefore requires shipping a
prostate-gland segmenter, and `docs/CLOUD_DEPLOYMENT_PLAN.md` carries it as such. Step 3's fallback
covers an *empty* gland mask; it does not cover a *wrong* one, and `11050_1001070` is the standing
example — both Bosma22b and Guerbet23 disagree with the lesion annotation there.

**Splits:** official PI-CAI patient-disjoint 5-fold splits (`picai_baseline`/`Z-SSMNet`), verified
independently for no patient leakage and every grade present in every held-out fold.

---

## 4. Modular design for ablation

```yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: fdsf      # fdsf | lff
      fusion_type: waf                 # waf | caf
      num_levels: 5
      fusion_levels: [0, 1, 2, 3, 4]   # starting point; frozen by M2 profiling
      fdsf: {radius: 0.15}
      lff: {grid_size: [4, 8, 8], groups: 1}   # in_channels is 3 at the input; 3 % 8 != 0
      waf: {window_size: [2, 7, 7], num_heads: 4}
      caf: {window_size: [2, 7, 7], num_heads: 4, dropout: 0.0}
```

| Config | `frequency_filter_type` | `fusion_type` |
|---|---|---|
| Baseline (built FDSF+WAF) | `fdsf` | `waf` |
| LFF only | `lff` | `waf` |
| CAF only | `fdsf` | `caf` |
| Full GCALF-Net | `lff` | `caf` |

One `Encoder` class; `nndet/arch/encoder/gcalf/registry.py` is the only string→module mapping. The
encoder emits exactly five feature levels, numbered `0..4`, and the decoder consumes all five.
Configuration validation rejects any other level count or out-of-range fusion level.

**`fusion_levels` is measured before it is frozen, not assumed.** `[0,1,2,3,4]` is the
paper-faithful *starting point*, not a settled value: level 0 carries no stride
(`modular.py:120`; `get_strides()` returns `[1,1,1]`), so fusing there means windowed attention
over the full-resolution feature map plus a trilinear upsample of the Swin feature to that size
(`modular.py:195-198`). Near-full-resolution 3D attention was ADR 0001's consequence 4 and the
primary OOM risk in this design; that constraint is still live and is not answered by declaring a
default. M2's V gate therefore **profiles every level before any comparative fold is trained**
(`ROADMAP` M2), and the measurement selects the frozen subset. Judge it at the planner's resolved
patch size, not at the raw-task volume (§3 step 7). Once frozen, the subset is identical for WAF
and CAF in every arm and never revisited after seeing model results.
Same experiment-directory convention as before: `gcalf_experiments/<config>_seed<NN>/` with
`checkpoints/`, `logs/metrics.csv`, `predictions/`, `config_snapshot.yaml`, `git_commit.txt`.

---

## 5. FDSF (fixed 3D FFT) and the built baseline

**Purpose:** the frozen frequency control. FDSF runs exactly once on the three-channel bpMRI input,
before both branches of the five-level encoder. It does not replace stage-local wavelet modules.

**Definition:**
1. `X = fftn(x, dim=(-3,-2,-1))`, then `fftshift` to center frequencies.
2. Normalized radial frequency coordinates `r = sqrt((d/D)² + (h/H)² + (w/W)²)` on the centered grid.
3. Fixed spherical mask `M = 1[r ≤ 0.15]`.
4. `X_low = M·X`, `X_high = (1-M)·X`.
5. `ifftshift` each, then `ifftn` back to spatial domain.
6. Route `X_low` toward the Swin branch and `X_high` toward the CNN branch, exactly as specified by
   the paper. This direction is fixed across all baseline and ablation runs.

Run FFT in fp32 (`torch.cuda.amp.autocast(enabled=False)` around the FFT block, as in GFNet).

**Unit tests:** mask radius is exact at the boundary; `X_low + X_high` reconstructs `X` (aliasing-
free split); shape preserved for odd/even `(D,H,W)`; a constant input passes almost entirely
through `X_low`; a Nyquist-checkerboard input passes almost entirely through `X_high` (this is the
orientation gate — it catches a missing `fftshift`/`ifftshift` pair, the same defect class flagged
for LFF below).

---

## 6. Learnable Frequency Filter (LFF)

**Purpose:** replace the input-level FDSF fixed spherical mask with one learned, shared,
real-valued response at the identical input location, emitting the **same complementary
`(x_low, x_high)` pair** — the *only* change from §5, so LFF vs. baseline is a single-variable
ablation. LFF is a learned *separation*, not a filter: matching FDSF's arity is what makes the
`lff_only` arm a drop-in swap rather than a different pipeline.

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
- **Parameterize as `H = M_fixed + delta_H`**, where `M_fixed` is §5's spherical mask at
  `r ≤ 0.15` and `delta_H` is zero-initialized. At init LFF is then *exactly FDSF*, not merely an
  identity map — a stronger baseline-invariance property than `H = 1 + delta_H` (which would start
  at `x_low = x, x_high = 0` and starve the CNN branch at step 0), and the reason §5's regression
  test can be a direct equality check. No separate spatial residual — an all-ones filter *plus* a
  residual returns ≈2x, not x.
- **Take the high branch by subtraction**, `x_high = x - x_low`, so complementary reconstruction
  holds by construction and FDSF's own lossless-split test (`PHASE_2 §2.2`) applies unchanged to
  both arms.

```python
class LearnableFrequencyFilter3D(nn.Module):
    """Grouped, shape-tolerant, zero-phase, real-valued learnable frequency separation.

    Same signature and same (x_low, x_high) arity as FDSF. At initialization the
    learned response *is* FDSF's fixed spherical mask, so the two are numerically
    identical and the baseline cannot move when the registry gains this branch.
    """

    def __init__(self, in_channels, radius=0.15, grid_size=(4, 8, 8), groups=1):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        self.in_channels = in_channels
        self.groups = groups
        self.radius = radius
        self.delta_weight = nn.Parameter(torch.zeros(1, groups, *grid_size))  # real, zero-init

    def forward(self, x):
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            spectrum = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
            # centered on D,H; monotonic DC->Nyquist on the trailing rFFT axis (§6 bullet 1)
            base = spherical_mask_rfft(spectrum.shape[-3:], self.radius,
                                       device=x.device)            # == §5's M
            delta = F.interpolate(self.delta_weight, size=spectrum.shape[-3:],
                                   mode="trilinear", align_corners=True)
            gain = (base + delta).repeat_interleave(self.in_channels // self.groups, dim=1)
            gain = torch.fft.ifftshift(gain, dim=(-3, -2))  # centered grid -> rfftn ordering
            x_low = torch.fft.irfftn(spectrum * gain, s=x.shape[-3:],
                                      dim=(-3, -2, -1), norm="ortho")
        x_low = x_low.to(input_dtype)
        return x_low, x - x_low          # complementary by construction
```

**Unit tests** (`tests/gcalf/test_lff.py`): both outputs keep the input shape across several
`(D,H,W)`; **`x_low + x_high == x`** to `atol=1e-5` (the same lossless-split assertion FDSF must
pass); **`delta_weight=0` ⟹ LFF's `(x_low, x_high)` equals FDSF's, not merely `x`** — the
init-equals-baseline gate; gradient reaches `delta_weight` through *both* returned tensors;
**orientation gate** — a grid that pushes the response to 1 at centre and −1 elsewhere must pass a
constant volume into `x_low` and a Nyquist checkerboard into `x_high` (inverts under a missing
`ifftshift` — the single most valuable test in the file); finite under autocast on CPU and CUDA;
a fixed-seed regression test proving the FDSF baseline's output is byte-identical after LFF's
registry refactor.

---

## 7. WAF (wired) and Cross-Attention Fusion (CAF)

**WAF** (baseline fusion): instantiate the existing
`nndet/arch/encoder/window_attention_fusion.py::WindowAttentionFusion` in place of
`MemoryEfficientFusion`. It already implements shifted-window self-attention; the work here is
wiring, not writing. It fuses corresponding CNN and Swin features at the frozen `fusion_levels`
(§4 — starting point `[0,1,2,3,4]`, settled by M2's profiling); the decoder receives all five
encoder outputs whether or not every level carries a fusion module.

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
gradients reach both alignment convs and both attention projections; CUDA peak memory at all five
real feature shapes passes a declared gate.

**Fallback ladder** (walk it against M2's measurements, and freeze the outcome before the four-arm
comparison): reduce window to `(2,4,4)`; activation checkpointing around both WAF and CAF; drop the
highest-resolution level(s) from `fusion_levels`, which is where the cost concentrates; TransFuse
BiFusion as a separately named fallback experiment. Never silently relabel BiFusion as CAF or
reduce only one ablation arm.

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
- **AMP:** everywhere except the FFT block in FDSF/LFF (`autocast(enabled=False)` around it).
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
`docs/m0-verification.md`), CAF/FDSF CUDA memory profiling — is a cloud (`V`) gate, not local.

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
| FDSF/WAF build takes longer than budgeted | High | This is the largest new-code item in the plan (§0); if it slips past its phase gate, step down the budget ladder rather than compressing later phases. |
| Grade head does not learn on 220–340 lesions | Medium | Widen with the D3 unifocal-recovery audit before concluding the head is broken; report per-grade CIs regardless. |
| `nnDetection csrc`/old-torch env won't build | High | Docker image with the exact pinned base; CPU fallback for NMS in smoke tests; this is the #1 environment blocker historically. |
| GPU memory (3D + attention) | High | Smaller shared CAF/WAF windows, activation checkpointing, a predeclared fusion-level subset shared by both modules, batch 1 + accumulation, AMP everywhere except FFT. |
| GGG4/5 tiny even after D3 audit | Certain (floor is 20/18) | Report CIs; consider the pre-registered GGG4+5 merged secondary analysis (ADR 0002 D8) once audit counts are known. |
| Full 20-run matrix over budget | Med | Pre-committed ladder (§9); never drop folds for only some configs. |

---

### Appendix — files to touch vs. reference

**Create:** `nndet/arch/encoder/gcalf/{fdsf,waf,lff,caf,grade_head,registry}.py`,
`gcalf_configs/*.yaml`, `tests/gcalf/test_{fdsf,waf,lff,caf,grade_head}.py`.
**Edit:** `nndet/arch/encoder/modular.py` (registry calls), `gcalf_data/*` (rework for the D2
two-head data contract — see `docs/phases/PHASE_1_data_pipeline.md`).
**Adapt from:** `GFNet/gfnet.py::GlobalFilter` (§6), DCA/UCTransNet (§7 Q/K/V math).
**Wire, don't rewrite:** `nndet/arch/encoder/window_attention_fusion.py::WindowAttentionFusion`.
**Keep untouched, reference-only:** `WaveletFusion.py`, `channel_lightweight_fusion.py`,
`swimTransformer.py`, `decoder/BiFPN.py`.
**Install as tools:** `picai_prep`, `picai_eval`, `medcam`. **Reference-only:** `Z-SSMNet`.
