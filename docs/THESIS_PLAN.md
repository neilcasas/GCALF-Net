# GCALF-Net — Technical Implementation Plan

**Thesis:** GCALF-Net: Modified PDHD-Net with Adaptive Frequency Filtering and Cross-Attention Fusion for Gleason Grade Group Classification using bpMRI
**Dataset:** PI-CAI (bpMRI: T2W, ADC, DWI) — 5-class Gleason Grade Group (GGG 1–5)
**Date:** 2026-07-30 (rev. 4 - code-verified corrections; working repo renamed `GCALF-Net`)

> This plan is grounded in an actual inspection of the cloned repos, not the paper text. Where the cloned code contradicts the thesis masterfile, that is flagged explicitly — read §0 first, it changes several downstream decisions.
>
> **This is the master overview.** The step-by-step execution plans live in `docs/phases/PHASE_0…PHASE_7.md` — each is a self-contained, runnable checklist for one project phase. Read this file for the "why/what", the phase files for the "how/when".
>
> **Rev. 3 decisions:** lesion-level nnDetection multi-task learning remains primary; LFF uses an interpolated low-resolution complex spectral grid; CAF means true bidirectional windowed Q/K/V cross-attention; the required experiment is four configurations over all five official folds; the legacy runtime is frozen; and cloud training uses Docker on a provider-neutral GPU VM with S3-compatible storage staged to local NVMe. See `docs/adr/0001-gcalf-architecture-and-cloud.md`.
>
> **Rev. 4 corrections (verified against the code, not the papers):** the working repository is `GCALF-Net/` (a copy of `PDHD-Net/` at tag `pdhd-upstream`; the original stays untouched as the released reference); `classifier_classes` is **5**, not 6 — nnDetection instance classes are foreground-only and 0-indexed, so `lesion_ISUP k → class k-1` and benign cases carry **zero instances**; the LFF spectral grid must be defined on shifted (centered) frequency coordinates and use a real-valued gain; CAF must apply its CNN residual **once** and mask padded window tokens; and the real training schedule is 50 epochs × 2500 batches (+10 SWA), which fixes the compute budget for the 20-run matrix. See §0 and the amendments in the ADR.
>
> **Documentation map:** `docs/README.md` indexes the plans, `docs/GLOSSARY.md` defines canonical terms, and `docs/CLOUD_DEPLOYMENT_PLAN.md` is the cloud runbook.

---

## 0. Reality check — what the repos actually contain (read this first)

The masterfile describes an idealized PDHD-Net. The cloned code is different in ways that matter:

| Masterfile says | Cloned `PDHD-Net/` actually has | Consequence |
|---|---|---|
| FDSF / FMB / FDR: FFT + spherical low/high mask, hyperparameter α | `nndet/arch/encoder/WaveletFusion.py` → `WaveletSpatialFusion` (Haar **wavelet**, not FFT). FFT path is present as dead params (`fft_low_ratio`) and **commented out** in `modular.py` (`移除FFT频域处理` = "removed FFT frequency processing") | Your "replace fixed frequency module with LFF" = replace **`WaveletSpatialFusion`** (the module actually wired at encoder stages 1,3,4). The paper's FFT/α module isn't in the code; do not go looking for it. |
| ConvSwin3D fused by WAF (window self-attention) | Fusion actually wired = `MemoryEfficientFusion` → `ChannelWiseLightFusion` (ECA channel attention + conv), at stages 2 and 5. `WindowAttentionFusion` (WAF) exists in `window_attention_fusion.py` but is **not used** by `modular.py`. | Your "replace WAF with CAF" = replace **`MemoryEfficientFusion`/`ChannelWiseLightFusion`**. If your thesis needs a literal "WAF baseline", you must wire `WindowAttentionFusion` in yourself — treat it as a third fusion option. |
| 3D-LDFPN decoder | `nndet/arch/decoder/BiFPN.py` (`ClassBiFPN`) + `decoder/base.py` | Matches. Keep unchanged. |
| PDHD-Net = classifier for GGG | PDHD-Net = **fork of nnDetection** (MIC-DKFZ). It is a detection framework (RetinaUNet: detection + per-lesion classification + segmentation), not a plain classifier. Entry points `scripts/train.py`, `scripts/predict.py`, model class `nndet/ptmodule/retinaunet/v001.py::RetinaUNetV001`. | "Classification" is a per-lesion head inside a detection model. A pure 5-class classifier is a **fallback** (§14), not the default. Your baseline is a detection+classification run. |
| 6 classes (background + GGG1–5) | `nndet/planning/architecture/boxes/base.py:89` sets `classifier_classes = len(dataset_properties["class_dct"])`, and `class_dct` is `dataset.json["labels"]` — **foreground instance classes only, 0-indexed**. PDHD-Net's own `README.md` already documents the intended task: `{"0":"GGG1","1":"GGG2","2":"GGG3","3":"GGG4","4":"GGG5"}`. RetinaUNet scores classes with sigmoid focal loss, so background is implicit and never a channel. | **`classifier_classes = 5`.** Labels map `lesion_ISUP k → class k-1`. A benign case is not "class 0" — it is a case with an **empty `instances` dict** and no boxes. The confusion matrix is 5×5 over detected lesions, with false positives on benign cases counted separately (§10). |
| Training runs ~1000 epochs | `nndet/conf/train/v001.yaml`: `max_num_epochs: 50`, `num_train_batches_per_epoch: 2500`, `swa_epochs: 10`, `precision: 16`. There is **no** `EarlyStopping` — the schedule is fixed poly-LR plus SWA. (The README's "1000 epochs" contradicts the shipped config; the config wins.) | ≈150k iterations per run fixes the compute budget: see §9 and the pre-committed reduced matrix in §12. |

**Bottom line:** the two modifications map cleanly onto two `nn.ModuleList` slots in one file (`nndet/arch/encoder/modular.py`). That is the entire integration surface for LFF and CAF. Everything else (nnDetection planning, training, detection heads, BiFPN, PI-CAI preprocessing) stays as-is.

---

## 1. Repository role mapping

| Repo | Role | What to do with it | Key paths |
|---|---|---|---|
| **GCALF-Net** | **Main working repository.** A copy of `PDHD-Net/` (tag `pdhd-upstream` marks the copy point; the upstream remote is removed). | **Build inside it.** All GCALF-Net code goes here. `git diff pdhd-upstream` is the exact "what we changed vs the released implementation" for the thesis. | `nndet/arch/encoder/{modular,swimTransformer,WaveletFusion,channel_lightweight_fusion,window_attention_fusion}.py`, `nndet/ptmodule/retinaunet/{base,v001}.py`, `scripts/{train,predict,preprocess}.py`, `nndet/conf/train/{v001,smoke}.yaml` |
| **PDHD-Net** | Released reference, kept pristine. | **Do not edit.** Read-only comparison point. | same paths, unmodified |
| **GFNet** | Reference only (learnable Fourier filter). | **Copy the idea**, not the file. Port `GlobalFilter` (2D) → new 3D module. | `GFNet/gfnet.py::GlobalFilter` (lines 49–74) |
| **TransFuse** | Fusion fallback/reference. | Its `BiFusion_block` is useful as a cheap non-QKV ablation, but it must not be called cross-attention. | `TransFuse/lib/TransFuse.py::BiFusion_block` (lines 20–72), `ChannelPool` (15) |
| **Dual-Cross-Attention** | Secondary CAF reference (channel + spatial cross-attention over a pyramid). | **Adapt the mechanism.** DCA is 2D and built for U-Net skip pyramids, not 2-branch fusion — don't import it, reimplement 3D. | `Dual-Cross-Attention/model/utils/dca.py::{ChannelAttention, SpatialAttention, CCSABlock, DCA}` |
| **UCTransNet** | Secondary CAF reference (channel cross-attention origin). | Read for the channel-wise cross-attention math (`Attention_org`). 2D, don't import. | `UCTransNet/nets/CTrans.py::{ChannelTransformer, Attention_org, Block_ViT}` |
| **M3d-Cam (`medcam`)** | **Tool (install).** Drop-in 2D/3D Grad-CAM, Grad-CAM++, Guided-BP. | `medcam.inject(model, backend='gcam', layer=…)`. Use for §8 instead of hand-rolling. | `M3d-Cam/medcam/{medcam_inject.py::inject, backends/grad_cam.py, backends/grad_cam_pp.py}` |
| **Z-SSMNet** | Reference (PI-CAI pipeline + loss). Official challenge solution on nnU-Net/nnDetection. | Cross-check data prep; reuse the **official splits** and **focal loss** idea. Don't adopt its MNet/SSL model (off-thesis). | `Z-SSMNet/src/z_ssmnet/{prepare_data.py, splits/picai/, z_nnmnet/training_docker/focal_loss.py, zonal_segmentation/}` |
| **picai_prep** | Tool (wrap). Preprocessing DICOM/MHA → nnUNet/nnDetection format. | Use as installed package / CLI. | `picai_prep/…/{dcm2mha,mha2nnunet,nnunet2nndet}.py` |
| **picai_labels** | Data (use). Labels + delineations. | Read `marksheet.csv`; use lesion/gland masks. | `clinical_information/marksheet.csv`, `csPCa_lesion_delineations/`, `anatomical_delineations/` |
| **picai_eval** | Tool (wrap). Detection/lesion metrics. | Use `evaluate()` API for FROC/AUROC. | `picai_eval/…/eval.py`, `metrics.py` |
| **picai_baseline** | Reference (docs/scripts). The canonical PI-CAI → nnDetection recipe. | **Follow `nndetection_baseline.md` step-by-step** for data prep — it is the tested path. | `nndetection_baseline.md`, `nnunet_baseline.md`, `src/` |

**Dependency conflicts (real, must resolve):**
- PDHD-Net `requirements.txt` pins `pytorch_lightning>=1.3.1,<=1.4.2`, `nnunet==1.7.1`, `SimpleITK<2.1.0`, `torchmetrics<=0.7.3`. This is a **2021-era stack** → implies torch ~1.8–1.10. **This is your environment anchor.** Everything else bends to it.
- GFNet (`timm` + `torch.fft`), TransFuse (`timm==0.3.2`), UCTransNet (`einops`), DCA (`einops`) are all **copy-in** pure-`nn.Module` references — never `pip install` them, so their pins don't touch your env. Only lift the specific classes.
- picai_prep / picai_eval are modern and light (SimpleITK, numpy, scipy) — compatible, install as packages.
- **`medcam`** deps are light (`nibabel`, `numpy`) and torch-version-agnostic (pure hooks) → installs cleanly into the pinned env. `pip install medcam` or `pip install -e M3d-Cam/`.
- **Z-SSMNet** is *not* installed — it targets its own nnU-Net fork and would fight PDHD-Net's pins. Treat it as read-only reference (copy the focal-loss class + splits if useful).
- **Risk:** nnDetection's `nndet_env`/CUDA extensions (`nndet/csrc/`) may need compilation against your torch/CUDA. Budget time for this; it's the #1 environment blocker (§14).

---

## 2. Proposed project structure

**Recommendation: build inside `GCALF-Net/`** — the copy of PDHD-Net made for this thesis (do not start a clean repo). Reason: GCALF-Net is two module swaps in an existing framework; a clean repo would mean reimplementing nnDetection's planning/training/detection machinery — weeks of work for zero thesis value. Add a thin `gcalf/` package for the new modules + a top-level experiments/config area, and keep everything else untouched so the baseline stays reproducible.

**One canonical location for new model code: `nndet/arch/encoder/gcalf/`.** It is inside the installed package, so it imports without touching `setup.py`. Do not also create a top-level `gcalf/`.

```
GCALF-Net/                             # git repo root (copy of PDHD-Net @ tag pdhd-upstream)
├── nndet/arch/encoder/
│   ├── modular.py                     # ← the ONE integration file (edit forward + __init__ to honor config)
│   ├── gcalf/                         # ← NEW: all thesis modules live here
│   │   ├── __init__.py
│   │   ├── lff.py                     # LearnableFrequencyFilter3D (replaces WaveletSpatialFusion)
│   │   ├── caf.py                     # WindowedCrossAttentionFusion3D
│   │   ├── classifier_model.py        # §14 reduced-scope fallback (Encoder + pool + linear head)
│   │   └── registry.py                # build_frequency_module(...), build_fusion_module(...)
│   ├── WaveletFusion.py               # keep (baseline freq option)
│   ├── channel_lightweight_fusion.py  # keep (baseline fusion option)
│   └── window_attention_fusion.py     # keep (optional literal-WAF baseline)
├── gcalf_configs/                     # ← NEW: ablation configs (YAML)
│   ├── baseline.yaml                  # wavelet + channel_light
│   ├── lff_only.yaml
│   ├── caf_only.yaml
│   └── gcalf_full.yaml
├── gcalf_data/                        # ← NEW: PI-CAI prep wrappers + sanity checks
│   ├── build_labels.py               # marksheet.csv → per-case/per-lesion GGG
│   ├── prepare_picai.py              # picai_prep pipeline driver
│   └── sanity_checks.py              # spacing/orientation/channel/label asserts
├── gcalf_eval/                        # ← NEW
│   ├── run_eval.py                   # picai_eval wrapper + 5-class classification metrics
│   └── gradcam.py                    # 3D Grad-CAM
├── gcalf_experiments/                 # ← NEW: outputs, gitignored
│   ├── <exp_name>/checkpoints/
│   ├── <exp_name>/logs/              # csv + tensorboard
│   ├── <exp_name>/predictions/
│   └── <exp_name>/config_snapshot.yaml + git_commit.txt + env.txt
├── notebooks/                         # ← NEW: debugging only (shape checks, gradcam viz)
├── scripts/{train,predict,preprocess}.py   # existing nnDetection entry points
└── requirements.txt / setup.py        # existing
```

Design principle for on/off: `modular.py` never names a concrete module. It calls `registry.build_frequency_module(...)` and `registry.build_fusion_module(...)`, which return the configured implementations. **One code path, config-selected modules.** The real configuration path is Hydra `model_cfg.encoder_kwargs` -> `RetinaUNetModule._build_encoder(..., **model_cfg["encoder_kwargs"])` -> `Encoder(gcalf_cfg=...)`; it is not read directly from the nnDetection plan.

---

## 3. Modular design for ablation

**Config flags (put under Hydra `model_cfg.encoder_kwargs.gcalf_cfg`):**

```yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: wavelet   # wavelet | lff | none
      fusion_type: channel_light       # channel_light | windowed_cross_attention
      freq_stages: [1, 3, 4]
      fusion_stages: [2, 5]
      lff:
        grid_size: [4, 8, 8]
        groups: 8
      caf:
        window_size: [2, 7, 7]
        num_heads: 4
        dropout: 0.0
```

The four required configurations:

| Config | `frequency_filter_type` | `fusion_type` |
|---|---|---|
| Adapted PDHD-Net baseline | `wavelet` | `channel_light` |
| PDHD + LFF only | `lff` | `channel_light` |
| PDHD + CAF only | `wavelet` | `windowed_cross_attention` |
| Full GCALF-Net | `lff` | `windowed_cross_attention` |

**Avoiding code duplication:** exactly one `Encoder` class. The registry/factory (`nndet/arch/encoder/gcalf/registry.py`) is the only place that maps a string to a module class. Do not retain duplicate boolean aliases such as `use_lff`; one canonical enum per module prevents contradictory configs.

**Experiment output organization:** one directory per config under `gcalf_experiments/<config_name>_<seed>/`, each containing `checkpoints/`, `logs/metrics.csv`, `predictions/`, and a frozen `config_snapshot.yaml` + `git_commit.txt`. A `gcalf_eval/collect_results.py` reads all `metrics.csv` into one comparison table (rows = configs, cols = metrics). This makes baseline-vs-LFF-vs-CAF-vs-full a one-command diff.

---

## 4. Dataset pipeline plan

**Ground truth (verified from `marksheet.csv`, 1500 rows, cols: `patient_id, study_id, …, lesion_GS, lesion_ISUP, case_ISUP, case_csPCa, center`):**

```
case_ISUP distribution: {0: 847, 1: 228, 2: 234, 3: 99, 4: 40, 5: 52}
```

**This is the single most important data finding and it complicates the "5-class" framing:**
- ISUP grade == Gleason Grade Group (GGG 1 ↔ ISUP 1, … GGG 5 ↔ ISUP 5). Use `case_ISUP` (case-level) or `lesion_ISUP` (per-lesion).
- Only **653 cases are GGG 1–5**; 847 are benign (ISUP 0). **Recommendation: keep the benign cases in training as negative (zero-instance) cases** — they contribute background anchors and keep FROC meaningful — and report GGG 1–5 metrics on the positives. Do **not** model "benign" as a class: nnDetection's classifier is foreground-only (see §0), so a benign case is a case whose `instances` dict is empty, not a case labelled 0.
- **Severe imbalance:** GGG4 = 40, GGG5 = 52 cases. Any 5-class split will have <10 GGG4 cases in a test fold. This drives class-weighting, stratified splits, and metric choice (§10). Flag this as a study limitation.
- **Per-lesion vs per-case ambiguity:** `lesion_ISUP`/`lesion_GS` are per-lesion (a case can have multiple lesions of different grades); `case_ISUP` is the max. nnDetection predicts per-lesion → GGG is a **per-lesion** label. Decide early: the detection model naturally does per-lesion GGG; a classifier fallback does per-case (use `case_ISUP`).

**Input format (T2W/ADC/DWI):** PI-CAI ships DICOM/MHA. Follow `picai_baseline/nndetection_baseline.md`:
1. `picai_prep` `dcm2mha` → per-sequence MHA.
2. `picai_prep` `mha2nnunet` → resampled, co-registered, stacked multi-channel `.nii.gz` in nnUNet layout (`_0000`=T2W, `_0001`=ADC, `_0002`=high-b DWI). **Channel order is fixed by the `_000X` suffix — assert it (§sanity).**
3. `picai_prep` `nnunet2nndet` (`nnunet2nndet.py`) → nnDetection task layout with detection boxes derived from lesion masks.
4. `scripts/convert_seg2det.py` / `convert_cls2fg.py` in PDHD-Net convert segmentation masks → detection targets with class = GGG.

**Labels for 5-class GGG:** `gcalf_data/build_labels.py` joins `marksheet.csv` (`lesion_ISUP`) to the lesion delineations in `picai_labels/csPCa_lesion_delineations/`, assigning each lesion its GGG. nnDetection's label format is a per-case **instance-ID volume** (`case.nii.gz`, voxels 1..N) plus `case.json` = `{"instances": {"1": <class>, …}}` where class is a **0-indexed foreground id**. So map `lesion_ISUP k → class k-1` (GGG1→0 … GGG5→4), and emit `"instances": {}` for benign cases. `dataset.json["labels"] = {"0":"GGG1", …, "4":"GGG5"}` — matching PDHD-Net's README exactly — which makes the planner derive `classifier_classes = 5`.

**Masks available:** csPCa lesion delineations (human + AI-derived — check subfolders, prefer human), whole-gland and PZ/TZ zonal masks (`anatomical_delineations/`). Use lesion masks for detection/segmentation targets; gland masks optionally for cropping/normalization.

**Uncertainties to document:** (a) some cases have AI-derived (not human) lesion masks — mixed provenance; (b) GGG4/5 tiny counts; (c) benign-case handling; (d) `lesion_ISUP` blank for benign rows — handle NaN.

**Preprocessing checklist:**
1. Download PI-CAI images + `picai_labels`.
2. `pip install` picai_prep, run `dcm2mha`.
3. Run `mha2nnunet` (produces resampled 3-channel volumes) — **spacing** typically resampled to something like `3.0 × 0.5 × 0.5` mm (confirm from generated plan, do not assume).
4. `build_labels.py` → GGG-tagged lesion labels.
5. `nnunet2nndet` → nnDetection task.
6. `nndet_prep` (nnDetection planning + fingerprint) generates the `plan` dict incl. `patch_size`, spacing, `in_channels`.
7. Run `sanity_checks.py`.
8. Create fold splits — use the **official PI-CAI cross-validation splits**, available both in `picai_baseline` and in `Z-SSMNet/src/z_ssmnet/splits/picai/` (verified present). Don't reinvent; stratify test reporting by GGG.

**Cross-check with Z-SSMNet (optional but recommended):** `Z-SSMNet/src/z_ssmnet/prepare_data.py` is a *tested* end-to-end PI-CAI prep (an official challenge submission). If `picai_prep` + `picai_baseline` steps 2–5 stall, read Z-SSMNet's prep for the working invocation order and parameters. Z-SSMNet also demonstrates **zonal-aware** preprocessing (`zonal_segmentation/`, using the PZ/TZ masks in `anatomical_delineations/`) — this is an *optional* enhancement (extra input channel or crop guidance), **out of the core thesis scope**; note as future work unless you have spare time.

**Sanity checks (`gcalf_data/sanity_checks.py`, assert-based):**
- spacing identical across T2W/ADC/DWI after resampling (they were co-registered);
- orientation (SITK direction / affine) consistent, RAS or LPS uniformly;
- channel order `_0000/_0001/_0002` == T2W/ADC/DWI (load one case, eyeball intensity ranges: ADC ~ 0–3000, DWI high signal in lesions);
- label map: every value in each `case.json` `instances` dict is in `{0,1,2,3,4}`, the instance-ID volume's non-zero ids match the dict keys exactly, and per-class lesion counts match the marksheet after the `k-1` shift;
- train/val/test disjoint by `patient_id` (no patient leakage across folds).

---

## 5. Baseline implementation plan (get this running FIRST — it is milestone 1)

**Goal:** a 3-channel adapted PDHD-Net (`wavelet` + `channel_light`) trains and evaluates end-to-end on PI-CAI before any LFF/CAF work.

**3-channel change — where:** nnDetection reads `in_channels` from the plan, not hardcoded. Set `plan["architecture"]["in_channels"] = 3` (consumed at `nndet/ptmodule/retinaunet/base.py::_build_encoder`, line ~531 `in_channels=plan_arch["in_channels"]`). The nnDetection fingerprint step derives this from the data automatically once your task has 3 modalities (`_0000/_0001/_0002`). **So mostly: prepare data with 3 modalities and the plan picks up 3 channels.** Verify the Swin branch (`SwinTransformer3D(in_chans=in_channels, …)` in `modular.py` line 97) also receives 3 — it does, it reads the same `in_channels`.

**Confirm the baseline works:**
1. **Forward pass** — instantiate `RetinaUNetV001.from_config_plan` with a toy plan, feed a random `(1, 3, D, H, W)` tensor, check no shape error through encoder → BiFPN → heads.
2. **Training loop** — nnDetection's Lightning `RetinaUNetModule` handles loss (detection + classification focal + segmentation). Run 1 epoch on tiny data.
3. **Loss** — confirm the classification head has `classifier_classes = 5` (GGG1–5, foreground only — background is implicit in the sigmoid focal loss) in the plan.
4. **Eval** — `scripts/predict.py TASK RetinaUNetV001` → detection maps → picai_eval.

**Smoke test (tiny subset):** copy 4–6 cases (mix of GGG grades + 1 benign) into a `Task9xx_PICAI_TINY` task; run prep → 2-epoch train → predict → eval. Purpose is plumbing, not accuracy.

**Expected tensor shapes (representative — confirm exact `patch_size`/`start_channels` from the generated nnDetection plan):**

| Point | Shape (B,C,D,H,W) | Notes |
|---|---|---|
| Input patch | `(2, 3, 20, 320, 320)` | nnDetection patch; B,D,H,W from plan |
| After Swin patch-partition `(2,4,4)` | `(2, C_embed, 10, 80, 80)` | `SwinTransformer3D`, `embed_dim=start_channels` (e.g. 32) |
| CNN stem (stage 0) | `(2, 32, 20, 320, 320)` | `start_channels=32` |
| CNN branch stage i | channels `32·2^i`, spatial ÷ strides | e.g. stage 5 `(2, 320-ish, ...)` capped at `max_channels` |
| Swin branch stage i | `(2, 32·2^i, d_i, h_i, w_i)` | `transformer_out_channels[i]` |
| Freq module (stages 1,3,4) | in == out `(2, C_i, d,h,w)` | shape-preserving |
| Fusion (stages 2,5) | `(2, C_i, d,h,w)` | `cnn` + interp(`swin`) → fused |
| BiFPN outputs | list of `(2, fpn_channels, d,h,w)` | `fpn_channels` from plan |
| Classifier head | per-anchor logits → `(N_anchors, 5)` | detection is per-anchor, not per-volume; 5 = foreground GGG classes |

---

## 6. Learnable Frequency Filter (LFF)

**Purpose:** replace the fixed Haar-wavelet frequency module (`WaveletSpatialFusion`) with a learnable global filter in the 3D Fourier domain (GFNet idea), so the low/high split is learned per-data instead of fixed.

**Source/reference:** `GFNet/gfnet.py::GlobalFilter` (verified):
```python
self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2) * 0.02)  # 2D, last dim = real/imag
x = torch.fft.rfft2(x, dim=(1,2), norm='ortho')
x = x * torch.view_as_complex(self.complex_weight)
x = torch.fft.irfft2(x, s=(a,b), dim=(1,2), norm='ortho')
```

**Integration point:** `nndet/arch/encoder/modular.py`, the `self.wavelet_fusion_modules` slots (stages `[1,3,4]`). LFF must preserve `forward(x)->x` and accept the baseline channel arguments plus optional LFF settings. The registry selects it when `frequency_filter_type == lff`.

**Design - `LearnableFrequencyFilter3D` (new, `nndet/arch/encoder/gcalf/lff.py`):**
- 2D → 3D: `rfft2`→`torch.fft.rfftn(x, dim=(2,3,4))`, `irfft2`→`irfftn(..., s=(D,H,W))`.
- Learnable weight over the rfftn spectrum: for input `(B,C,D,H,W)`, rfftn along the 3 spatial dims gives `(B,C,D,H,W//2+1)` complex. Weight shape `(C, D, H, W//2+1, 2)` as `float32`, `view_as_complex`, multiply elementwise.
- **Variable-size solution:** learn a grouped low-resolution spectral grid and interpolate it to `(D,H,W//2+1)` at runtime. Unlike a per-channel scalar, this is frequency-selective; unlike a full spectrum, it is memory-bounded and shape-tolerant.
- **Define the grid on centered (fftshifted) frequency coordinates.** `rfftn` orders the `D` and `H` axes as `[0, +f, …, Nyquist, −f, …, −1]`. Interpolating a coarse grid directly over that index space puts DC and the lowest negative frequency at opposite ends with Nyquist in the middle, so a low-pass response is not expressible as a smooth low-index blob and the interpolation smooths *across* the Nyquist discontinuity. Build the grid in centered coordinates and `torch.fft.ifftshift` the interpolated result along `D,H` before multiplying (or index the grid by radial `|f|`).
- **Use a real-valued gain, not a complex one.** `x` is real ⟹ its spectrum is Hermitian, so multiplying by a non-Hermitian complex `H` and calling `irfftn` applies the Hermitian projection `(H(f) + conj(H(−f)))/2` — half the learned parameters alias onto the other half and are not identifiable. A real gain per bin is exactly Hermitian by construction, halves the parameter count, and is still frequency-selective (zero-phase filtering). Learning phase is a documented extension, not the default.
- Parameterize the transfer as `H = 1 + delta_H`, initialize `delta_H=0`, and do not add a second spatial residual. This gives an identity initialization while retaining gradients.

**Interface & shapes:**
| Stage | Tensor |
|---|---|
| input | `(B, C, D, H, W)` real |
| after `rfftn(dim=(2,3,4))` | `(B, C, D, H, W//2+1)` complex |
| learnable grid | `(G, Kd, Kh, Kw//2+1)` real gain on centered coords, interpolated → ifftshifted → expanded over channels |
| after multiply | `(B, C, D, H, W//2+1)` complex |
| after `irfftn(s=(D,H,W))` | `(B, C, D, H, W)` real |
| output (optional projection) | `(B, out_channels, D, H, W)` |

**Config:** `frequency_filter_type: lff`; `none` returns `nn.Identity` for an optional diagnostic ablation. See `docs/phases/PHASE_3_lff.md` for the implementation skeleton and exact config path.

**Memory / numerical stability:**
- Cast to `float32` before FFT (mixed-precision: wrap FFT in `torch.cuda.amp.autocast(enabled=False)` — cuFFT half-precision is unreliable). GFNet already forces `x.to(torch.float32)`.
- Zero-initialized `delta_H` makes the transfer exactly identity at startup without suppressing transfer-weight gradients.
- Measure FFT activation memory at every configured stage; grouped grids reduce parameter memory but not FFT activation memory.

**Unit test (`GCALF-Net/tests/gcalf/test_lff.py`):**
- shape in == shape out for several `(D,H,W)`;
- gradient flows to `delta_weight` (`loss.backward()`, assert `.grad is not None` and non-zero);
- with `delta_H=0` and identity projection, output approximates input (`allclose` atol 1e-4);
- synthetic low/high frequency inputs receive different gains after setting distinct grid bins;
- **frequency-axis orientation:** a grid that is 1 at the centre and 0 elsewhere must low-pass (a constant input survives, a Nyquist-checkerboard input is suppressed) — this is the test that catches a missing `ifftshift`;
- runs on CPU and CUDA; finite outputs (no NaN) under autocast.

---

## 7. Cross-Attention Fusion (CAF)

**Purpose:** replace symmetric/implicit channel-light fusion with directed cross-attention so the CNN branch can query the Swin branch and vice-versa — targeting the GGG2/GGG3 boundary the paper flags.

**Definition:** CAF means true bidirectional Q/K/V cross-attention: CNN windows query Swin keys/values and Swin windows query CNN keys/values. TransFuse BiFusion is only a separately named fallback/reference because gating plus a Hadamard interaction is not cross-attention.

**Integration point:** `nndet/arch/encoder/modular.py`, the `self.self_attention_fusion_modules` slots at stages `[2,5]`, currently `MemoryEfficientFusion`. CAF keeps `forward(cnn_feat, trans_feat)->fused`; the Swin feature is already interpolated to the CNN spatial size.

**Design - `WindowedCrossAttentionFusion3D`:** align both branches with `1x1x1` convolutions, pad and partition corresponding 3D windows, perform both directed `nn.MultiheadAttention` calls, reverse/crop windows, and fuse with the aligned CNN residual. `docs/phases/PHASE_4_caf.md` contains the initial module and window-helper snippets.

**Two rules the skeleton must respect:**
- **One residual, counted once.** The aligned CNN feature may enter the output through exactly one path. Adding `cnn_w` inside the attention sum *and* concatenating `cnn` *and* adding `+ cnn` at the end triples it — the same defect as the "all-ones filter plus spatial residual returns ≈2x" trap called out for LFF (§6).
- **Mask the padding.** Window partitioning pads `D/H/W` up to window multiples; with `window_size=(2,7,7)` on feature maps that are not multiples of 7 a large fraction of every window's keys are zeros. `window_partition_3d` must return a `key_padding_mask` and both `MultiheadAttention` calls must use it, or the model learns to attend to padding.

**Memory bound:** global spatial attention is forbidden. Window attention costs `O(n_windows * window_tokens^2)`. Profile stages 2 and 5; fallback in order to smaller windows, activation checkpointing, and stage-5-only true CAF. Never silently relabel BiFusion as CAF.

**Interface & shapes (stage 5 example, `C=out_channels`, `N=d·h·w`):**
| Point | Shape |
|---|---|
| CNN feat | `(B, cnn_channels, d, h, w)` |
| Swin feat (pre-interp) | `(B, trans_channels, d', h', w')` → interp → `(B, trans_channels, d,h,w)` |
| after 1×1×1 align | both `(B, C, d, h, w)` |
| window tokens | `(B*n_windows, wd*wh*ww, C)` |
| per-window attention | `(B*n_windows, heads, Nw, Nw)` |
| attention output | `(B*n_windows, Nw, C)` |
| reshaped fused | `(B, C, d, h, w)` → +residual → `(B, out_channels, d, h, w)` |

**Config:** `fusion_type: windowed_cross_attention` with `caf.window_size`, `caf.num_heads`, and `caf.dropout`. The baseline remains `channel_light`.

**Unit test (`GCALF-Net/tests/gcalf/test_caf.py`):**
- output shape matches CNN spatial dimensions for divisible and padded window shapes;
- **padding invariance:** for a feature map whose dims are not window multiples, the output on the valid region is unchanged when the padded region is filled with a different constant — fails if `key_padding_mask` is missing;
- **residual is counted once:** with both attention outputs forced to zero and an identity `out_proj`, the module returns the aligned CNN feature, not a multiple of it;
- gradients reach both branch projections;
- both directed attention projections receive gradients and branch ablation changes output;
- stage-2 and stage-5 representative tensors pass the declared memory gate;
- output finite under autocast; matches `MemoryEfficientFusion`'s output signature so it's a true drop-in.

---

## 8. Grad-CAM / explainability

**Rev. 2 — use `medcam` (M3d-Cam), don't hand-roll.** M3d-Cam is a PyTorch library that injects Grad-CAM / Grad-CAM++ / Guided-BP into any `nn.Module`, 2D **or 3D**, for classification and segmentation, and writes attention maps as NIfTI. One line:
```python
from medcam import medcam
model = medcam.inject(model, output_dir="attention_maps", backend='gcam',
                      layer='auto', label='best', save_maps=True, data_shape='default')
# then just run inference; maps are written per forward pass
```
`inject(model, output_dir, backend='gcam'|'gcampp'|'ggcam'|'gbp', layer='auto'|'full'|<name>|[names], label=<int>|<lambda>, …)` — verified in `M3d-Cam/medcam/medcam_inject.py`. `layer='auto'` picks the last CAM-able layer; `label` selects the target class channel.

**Where to attach:** the last conv layer feeding the GGG classifier head — the deepest BiFPN feature map used by `_build_head_classifier` (`nndet/ptmodule/retinaunet/base.py` ~line 570) / `nndet/arch/heads/classifier.py`. Pass that module's name as `layer=` to `medcam.inject`. Target the predicted GGG class of the detected lesion via `label=`.
- **nnDetection caveat:** RetinaUNet's head is per-anchor, so `layer='auto'` may latch onto a detection/regression layer, and the "class score" is per-anchor not per-volume. Two options: (a) specify the classifier conv `layer=` explicitly and a `label=` lambda that selects the target lesion's anchor logits; (b) run Grad-CAM on the **classifier-fallback model** (§14) whose single `(B, num_classes)` output is exactly what medcam expects — cleaner, and it's the recommended path for the urologist study.

**Outputs for review (`gcalf_eval/gradcam.py` = thin wrapper around medcam):**
- **NIfTI:** medcam writes CAM volumes directly; re-header them into the input's affine/spacing so radiologists overlay in a NIfTI/DICOM viewer.
- **PNG slices + overlay:** axial slice through the lesion centroid, T2W grayscale + jet CAM at α=0.4; also ADC/DWI overlays.
- **Clinical review packet:** per case, a PNG panel (3 slices × 3 sequences) + predicted GGG + CAM overlay, **blinded** (no GT/prediction label visible, per masterfile §Likert) for the 3 urologists' PI-RADS v2 5-point rating.
- medcam also has a built-in `evaluate=True` mode that scores maps against a ground-truth mask (`metric='wioa'`) — useful sanity metric (CAM-vs-lesion overlap) to report alongside the Likert scores.

**Test plan (`tests/test_gradcam.py`):**
- medcam injects and produces a non-empty CAM of input spatial shape;
- CAM for class c changes when `label` changes (not constant across classes);
- CAM spatial argmax roughly inside the lesion mask for a known strong positive (or `medcam.evaluate` overlap > chance);
- injected model's forward output is unchanged vs the un-injected model (inject is non-destructive).

---

## 9. Training plan

- **Loop:** reuse nnDetection's PyTorch-Lightning `RetinaUNetModule` (`nndet/ptmodule/`). Don't hand-roll — it already wires detection + classification + segmentation losses, sampling, and checkpointing.
- **Losses (multi-task, already in repo):** detection (anchor cls + box regression), **classification focal loss** with dynamic class weights (`nndet/losses/{classification,modern_classification}.py` — relevant for the GGG4/5 imbalance), segmentation (Dice/CE). Class weights should reflect the `{847,228,234,99,40,52}` distribution. For the classifier fallback (§14), `Z-SSMNet/src/z_ssmnet/z_nnmnet/training_docker/focal_loss.py` is a clean, PI-CAI-tuned focal-loss reference to copy rather than re-derive.
- **Classification-only start (if multi-task too hard):** see §14 fallback — a standalone classifier reusing `Encoder` + global-pool head, `case_ISUP` labels, plain cross-entropy/focal. Ships a result even if detection plumbing stalls.
- **Batch / precision / accumulation:** nnDetection auto-plans batch size from GPU mem; on an 11–16 GB GPU expect batch 2–4. Enable AMP (`autocast`) **except around the LFF FFT** (§6). Gradient accumulation 2–4 if batch forced to 1.
- **Schedule (read from `nndet/conf/train/v001.yaml`, not from the README):** `max_num_epochs: 50`, `num_train_batches_per_epoch: 2500`, `swa_epochs: 10`, `initial_lr: 0.01` SGD + poly decay (`poly_gamma: 0.9`), `warm_iterations: 4000`, `precision: 16`. That is ≈150k optimizer steps per run. The README's "1000 epochs" is stale — **the config is authoritative**.
- **Checkpointing / early stop:** Lightning `ModelCheckpoint` on `monitor_key: mAP_IoU_0.14_0.90_0.05_MaxDet_100` (already configured). nnDetection has **no `EarlyStopping`** — it runs the fixed schedule then SWA. Do not add one: it would make the four configs stop at different points and break the comparison. If the schedule must be shortened for budget, shorten it **identically for all 20 runs** (§12).
- **Compute budget:** ≈150k steps × 4 configs × 5 folds. Measure seconds/step in the fold-0 pilot before launching the matrix; at 2–4 it/s a single run is ~11–21 h, so the full matrix is ~10–20 GPU-days. The reduction rule is pre-committed in §12.
- **Resume:** Lightning `--resume`/`ckpt_path`; nnDetection stores under the task's results dir.
- **Logging:** CSV (always, for the results-collector) + TensorBoard. W&B optional but adds a dep and offline-run friction — CSV+TB is enough; add W&B only if you want live dashboards.
- **Reproducibility:** fixed seed (torch/np/random + `torch.use_deterministic_algorithms` where possible — note cuFFT/atomic ops may block full determinism, document it); snapshot `config.yaml`, `git rev-parse HEAD`, and `pip freeze` into each `gcalf_experiments/<exp>/`.

---

## 10. Evaluation plan

- **Detection / lesion-level:** `picai_eval` (`picai_eval/…/eval.py::evaluate`) → **FROC** + lesion **AUROC** (the PI-CAI standard). Feed it detection maps from `scripts/predict.py`. This answers "how well does it find lesions".
- **5-class GGG classification (the thesis question):** per-lesion (or per-case for the classifier fallback):
  - **Confusion matrix — 5×5** over matched lesions (GGG1–5), masterfile Table 3.2. There is no background row/column: the model has no background class (§0). Report the two detection-side error modes **next to** the matrix, not inside it: unmatched ground-truth lesions (misses) and false-positive detections, the latter broken out by benign vs positive case. A 6×6 "with background" matrix would misrepresent the model.
  - **Macro-F1**, **balanced accuracy**, **per-class sensitivity/recall** (esp. **GGG2 vs GGG3** — the headline metric), **per-class precision**;
  - **AUROC** one-vs-rest per class + macro (ordinal, so also report quadratic-weighted Cohen's κ — appropriate for graded classes);
  - given GGG4/5 tiny counts, report **95% CIs** (bootstrap) and lean on macro/balanced metrics over raw accuracy.
- **Segmentation (if masks used):** **Dice (DSC)** vs lesion delineations (`picai_eval` / `nndet` provide this).
- **Ablation comparison:** run all four configs with the **same folds/seeds**; `gcalf_eval/collect_results.py` builds one table: rows = {baseline, LFF-only, CAF-only, GCALF-full}, cols = {macro-F1, balanced-acc, GGG2 sens, GGG3 sens, κ, FROC}. Baseline is the reference column; report Δ.
- **Statistical testing (masterfile §F):** per-fold or per-case bootstrap; **Wilcoxon signed-rank** on paired per-fold metrics (baseline vs full) — masterfile cites Wilcoxon (1945) and Shapiro-Wilk for normality. Report p-values for GCALF-full vs baseline (thesis SOP #3).
- **Result files:** everything under `gcalf_experiments/<exp>/` — `predictions/`, `metrics.csv`, `confusion_matrix.png`, `froc.png`. Comparison table + plots regenerated by one script for reproducibility.

---

## 11. Local testing / development

**Environment (anchored to PDHD-Net's pins):**
- Python 3.8 or 3.9 (matches lightning 1.4.2 / nnunet 1.7.1 era).
- Conda env recommended (CUDA toolkit + old torch is easier in conda): `conda create -n gcalf python=3.9`.
- torch ~1.10 + matching CUDA (11.1/11.3) — must support `torch.fft.rfftn` (≥1.8 ✓) and compile nnDetection's `csrc` extensions.
- `pip install -e GCALF-Net/` (editable — it has `setup.py`), then `pip install -e picai_prep picai_eval` (editable) and `pip install -e M3d-Cam/` (or `pip install medcam`). Install **GCALF-Net**, never `PDHD-Net/` — both provide the `nndet` package and installing both would shadow each other. GFNet/TransFuse/UCTransNet/DCA classes are **copied in**, not installed; Z-SSMNet is read-only reference, **not installed**.
- Install order: torch first → nnDetection deps → picai tools → medcam → verify `nndet` CLI imports (`GCALF-Net/tests/test_imports.py`).

**CPU-only smoke tests:** module unit tests (§6/§7/§8) run on CPU; a `(1,3,16,64,64)` forward pass through `Encoder` on CPU to validate shapes without GPU.

**GPU local steps (if CUDA GPU available):**
- preprocess tiny subset: `python gcalf_data/prepare_picai.py --task Task900_TINY --n 6`
- forward pass: `python -c "from nndet... import RetinaUNetV001; ..."` (shape print)
- **overfit 1–2 samples:** train on 2 cases, expect loss → ~0, classifier memorizes GGG — proves the loss/label wiring.
- small baseline: 20-epoch run on ~50 cases.
- eval one checkpoint: `python scripts/predict.py Task900_TINY RetinaUNetV001 -f 0` → `gcalf_eval/run_eval.py`.
- Grad-CAM one case: `python gcalf_eval/gradcam.py --case <id> --ckpt <path>`.

---

## 12. Cloud testing / deployment

The authoritative runbook is `docs/CLOUD_DEPLOYMENT_PLAN.md`. The fixed contract is a pinned legacy Docker image on a provider-neutral NVIDIA GPU VM, S3-compatible object storage accessed through AWS CLI v2, immutable dataset manifests, preprocessed data staged to local NVMe before training, and checkpoint/log synchronization after every epoch. Raw-to-preprocessed conversion is a separate one-time cloud job; normal training never streams NIfTI files from object storage.

**Pre-committed budget rule (decide before the pilot, not after).** The required matrix is 20 runs × ≈150k steps ≈ 10–20 GPU-days (§9). After the baseline fold-0 and full-GCALF fold-0 pilots report measured seconds/step and cost, apply the **first** option that fits the budget and record which one was used:

1. **Full matrix**, 20 runs at the shipped schedule. Preferred.
2. **Shortened schedule, all 20 runs:** halve `num_train_batches_per_epoch` to 1250 for *every* run. Fairness is preserved because the change is identical across configs; absolute numbers drop and must be reported as a shortened schedule.
3. **Reduced matrix, 14 runs:** all four configs on fold 0 (ablation contrast) plus baseline and `gcalf_full` on all five folds (paired per-fold values for the Wilcoxon test in §10). SOP 2 then rests on a single fold and must say so.

Never reduce by dropping folds for some configs only — that breaks the paired statistics.

---

## 13. Integration strategy

**Non-breaking integration:** the registry/factory (§2, §3) means baseline behavior is the **default** config — LFF/CAF only activate on explicit flags, so the baseline is preserved bit-for-bit for comparison. `modular.py` calls `build_frequency_module(...)` / `build_fusion_module(...)`; with baseline config these return the original `WaveletSpatialFusion` / `MemoryEfficientFusion`.

**Wrappers/adapters:** LFF and CAF implement the **exact `__init__`/`forward` signatures** of the modules they replace (§6, §7) so they slot into the existing `ModuleList`s with zero changes to `Encoder.forward`.

**Branch/commit order (recommended implementation order):**
1. `feat/env-and-data` — environment, PI-CAI prep, sanity checks.
2. `feat/baseline-3ch` — 3-channel adapted PDHD-Net trains + evals (milestone 1). **Tag this.**
3. `feat/registry` — factory + config plumbing, baseline still passes (no behavior change).
4. `feat/lff` — `LearnableFrequencyFilter3D` + unit tests + `lff_only` config.
5. `feat/caf` — `WindowedCrossAttentionFusion3D` + unit tests + `caf_only` config.
6. `feat/gcalf-full` — both on, `gcalf_full` config.
7. `feat/eval` — metrics + comparison collector + stats.
8. `feat/gradcam` — explainability + review packets.

Each ablation config is validated against the tagged baseline before moving on.

---

## 14. Risk assessment & fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| **nnDetection csrc / old-torch env won't build** | High | Use the Docker image with the exact pinned base; if csrc fails, disable the CUDA NMS extension (nnDetection has CPU fallbacks) for smoke tests. Worst case → classifier fallback (below) which doesn't need detection csrc. |
| **PDHD-Net code ≠ paper (already confirmed)** | Certain | Plan already targets the *actual* modules (wavelet, channel-light). Document the discrepancy in the thesis as "we adapt the released implementation." |
| **GPU memory (3D + attention)** | High | Smaller CAF windows, activation checkpointing, stage-5-only CAF, smaller patch, batch 1 plus accumulation; AMP everywhere except FFT. |
| **Windowed CAF still too expensive** | High | Follow the documented fallback ladder and report the resolved architecture. BiFusion is a separately named fallback, never silently relabeled CAF. |
| **GGG labels confusing / GGG4-5 tiny (40,52)** | Certain | Treat as 6-class with background; heavy class weighting (copy Z-SSMNet `focal_loss.py`); report macro/balanced metrics + CIs; consider merging GGG4+5 as a documented sensitivity analysis (note prior work does this). |
| **PI-CAI preprocessing complexity** | Med | Follow `picai_baseline/nndetection_baseline.md` exactly; cross-check against `Z-SSMNet/…/prepare_data.py` if it stalls; use official splits; don't customize prep. |
| **Dependency conflicts** | Med | Single pinned env, everything editable-installed against it; GFNet/TransFuse/UCTransNet/DCA copied not installed; Z-SSMNet reference-only. |
| **Grad-CAM incompatible with detection head** | Med→Low | **medcam mitigates** — `medcam.inject` on an explicit classifier `layer=`; if per-anchor logits are awkward, run it on the classifier-fallback model (clean `(B,num_classes)` output) for the explainability study. |
| **Full multi-task detection too hard to converge** | Med-High | **Reduced-scope fallback model (below).** |

**Reduced-scope fallback model (keeps the thesis intact):** a standalone 3D classifier = `nndet/arch/encoder/modular.py::Encoder` (unchanged, incl. LFF/CAF swaps) + global average pool + linear GGG head, trained on **whole-gland-cropped or lesion-cropped** volumes with `case_ISUP` labels, plain focal loss. This **still tests LFF and CAF** (they live in the encoder) and still answers SOPs 1–3 (baseline vs LFF vs CAF vs full) with confusion matrix / macro-F1 / κ. It drops FROC/Dice/detection but preserves the core contribution. Use this if detection plumbing blocks progress past milestone 2.

**Fallback experiments that still support the thesis:** even classifier-only, the four-way ablation + Grad-CAM urologist study fully answers the research questions. FROC/segmentation become "additional results if detection works."

---

## 15. Milestone timeline

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **M0 — Setup** | Pinned env (Docker), all repos import, unit-test scaffold | `test_imports` green; `Encoder` CPU forward pass runs |
| **M1 — Data** | PI-CAI → nnDetection task, GGG labels built, sanity checks pass | `sanity_checks.py` all-assert green; official folds loaded |
| **M2 — Baseline forward** | 3-channel adapted PDHD-Net instantiates, forward pass, overfits 1–2 samples | loss→~0 on 2 samples; shapes match §5 table |
| **M3 — Baseline tiny train** | 2-epoch run on `Task900_TINY`, predict, eval end-to-end | full pipeline runs, produces `metrics.csv` |
| **M4 — Baseline full train** ⭐ | **End-to-end adapted PDHD-Net trained + evaluated on PI-CAI** (this is the first real milestone; happens *before* LFF/CAF) | baseline numbers reported; tagged commit |
| **M5 — LFF** | `LearnableFrequencyFilter3D` + tests + `lff_only` run | unit tests pass; LFF-only trains, evals |
| **M6 — CAF** | `WindowedCrossAttentionFusion3D` + tests + `caf_only` run | unit tests pass; CAF-only trains, evals |
| **M7 — GCALF full + ablations** | All four configs, same folds/seeds | 4-way comparison table produced |
| **M8 — Evaluation** | picai_eval FROC/AUROC, 5-class metrics, Wilcoxon test | stats + confusion matrices + Δ table |
| **M9 — Grad-CAM** | 3D CAM + blinded urologist review packets | CAMs localize to lesions; packets exported |
| **M10 — Thesis outputs** | Result tables, ablation plots, CAM figures | all SOPs 1–4 answered |

**Ordering rule (per your requirement):** M4 (full baseline) must complete before M5 (LFF). Everything after M2 is config-driven, so LFF/CAF work never blocks the baseline result.

---

### Appendix A — exact files to touch vs reference (quick index)

All paths are relative to the working repo `GCALF-Net/`.

**Edit:** `nndet/arch/encoder/modular.py` (registry calls + `gcalf_cfg` kwarg). `in_channels=3` and `classifier_classes=5` are **derived by the planner from the data** (`nndet/planning/architecture/boxes/base.py:89-95`) — assert them, don't hand-edit `base.py`.
**Create:** `nndet/arch/encoder/gcalf/{lff.py,caf.py,registry.py,classifier_model.py}`, `gcalf_configs/*.yaml`, `gcalf_data/*`, `gcalf_eval/*`, `tests/gcalf/test_{lff,caf,gradcam}.py`.
**Adapt concepts from:** `GFNet/gfnet.py::GlobalFilter` for complex spectral filtering and DCA/UCTransNet for Q/K/V cross-attention. Keep TransFuse BiFusion as a non-cross-attention fallback/reference only.
**Install as tools:** `picai_prep`, `picai_eval`, `medcam` (M3d-Cam). **Reference-only (don't install):** `Z-SSMNet` (prep + focal loss + splits).
**Keep untouched (baseline options):** `WaveletFusion.py`, `channel_lightweight_fusion.py`, `window_attention_fusion.py`, `swimTransformer.py`, `decoder/BiFPN.py`.
**Follow as recipe:** `picai_baseline/nndetection_baseline.md`.

---

### Appendix B — phase execution plans

The 11 milestones (M0–M10) are grouped into 8 executable phases, each with its own comprehensive plan in `docs/phases/`:

| Phase file | Covers | Milestones |
|---|---|---|
| `PHASE_0_environment.md` | Env, installs, csrc build, repo layout, smoke imports | M0 |
| `PHASE_1_data_pipeline.md` | PI-CAI prep, GGG labels, splits, sanity checks | M1 |
| `PHASE_2_baseline.md` | 3-channel adapted PDHD-Net: forward → overfit → tiny → full train/eval | M2–M4 |
| `PHASE_3_lff.md` | **Registry refactor** (baseline-invariant) + `LearnableFrequencyFilter3D`, tests, `lff_only` run | M5 |
| `PHASE_4_caf.md` | Bidirectional windowed Q/K/V cross-attention, tests, profiling, `caf_only` run | M6 |
| `PHASE_5_integration_ablation.md` | Full GCALF-Net (LFF+CAF), 4-way ablation matrix, results collector | M7 |
| `PHASE_6_evaluation.md` | picai_eval, 5-class metrics, stats, comparison tables | M8, M10 |
| `PHASE_7_gradcam.md` | medcam 3D CAM, overlays, blinded urologist packets | M9, M10 |
