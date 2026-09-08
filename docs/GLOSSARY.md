# GCALF-Net Glossary

| Term | Meaning in this repository |
|---|---|
| bpMRI | Biparametric MRI input composed of T2W, ADC, and high-b-value DWI (HBV). |
| csPCa | Clinically significant prostate cancer, PI-CAI's own reference standard: ISUP ≥2. The detector's single foreground class. ISUP ≤1 (benign and GGG1) is background by this definition, not a separate class. |
| GGG | Gleason Grade Group. This project grades **GGG2–5** (`ISUP 2..5`); GGG1 has no spatial mask in PI-CAI and is not a class — see `csPCa` above and ADR 0002 D1. |
| Grade-supervised lesion | A positive lesion whose spatial mask carries a grade-specific voxel value (`human_expert/original`, or recovered by the Phase 1 unifocal linkage audit), as opposed to a binary-only (`Pooch25`/`Bosma22a`) positive mask. Only grade-supervised lesions train the grade head or appear in the grade confusion matrix. |
| Grade head | The separate 4-logit GGG2–5 classifier over matched positive detections (`nndet/arch/encoder/gcalf/grade_head.py`), trained with class-weighted cross-entropy masked to grade-supervised lesions only. Distinct from nnDetection's own single-class `csPCa` anchor classifier, which is unchanged and uses focal loss. |
| Detection denominator / grade-matched denominator | The two counts that must always be reported side by side for any grade metric: 1,499 retained detection cases (one source exclusion) vs. detections matched to a grade-supervised ground-truth lesion. Never collapse one into the other. |
| Instance class | The 0-indexed foreground class id in nnDetection's `labelsTr/<case>.json` `instances` dict. Under this project's contract there is exactly one, `0` = `csPCa`; grade is carried as separate instance metadata (`grade`, `grade_source`, `grade_supervised`), not as the class id. |
| Negative case | A case with no detection-positive lesion: all-zero instance volume, `"instances": {}"`. Benign (ISUP 0) and GGG1 (ISUP 1) PI-CAI cases are both negatives under this contract. |
| FDSF | Frequency-Domain Separation and Shunting: one input-level fixed 3D FFT → `fftshift` → radial low-pass mask at `\|f\|≤0.15` → complementary low/high branches → `ifftshift`/`ifftn`; low feeds Swin and high feeds CNN. The frozen baseline mechanism, built new because it is absent from the released code. |
| FDR | Fixed decomposition representation: the fixed spherical response inside FDSF. Do not use “FDR” as the name of a stage-local module or as a synonym for the complete FDSF routing operation. |
| LFF | Learnable Frequency Filter: replaces FDSF's fixed frequency response at the same input location, returning the same complementary `(x_low, x_high)` pair and initialized as FDSF's own spherical mask plus a zero delta — so at step 0 it *is* FDSF. It is the sole frequency-factor change; the encoder depth and fusion locations remain fixed. |
| WAF | Window Attention Fusion: `nndet/arch/encoder/window_attention_fusion.py::WindowAttentionFusion` — written, imported, but never instantiated in the released code. The frozen baseline fusion, placed at the `fusion_levels` that M2's memory profiling freezes. WAF and CAF always use identical fusion locations. |
| `fusion_levels` | The encoder levels carrying a fusion module. Starts at `[0,1,2,3,4]`, is settled by M2 profiling at the planner's resolved patch size, and is then frozen and shared by every arm. Never re-chosen after fold results exist. |
| CAF | Cross-Attention Fusion: WAF with its self-attention replaced by true bidirectional windowed Q/K/V cross-attention (CNN queries Swin, Swin queries CNN). The single variable that differs from WAF. It does not mean BiFusion. |
| BiFusion | TransFuse gating/Hadamard fusion reference. Not Q/K/V cross-attention; a named fallback only, never relabeled CAF. |
| PDHD-Net released baseline | The code as shipped: `WaveletSpatialFusion` at stages `[1,3,4]` + `ChannelWiseLightFusion` at stages `[2,5]`. This project's baseline is the reconstructed paper-derived five-level FDSF+WAF control, not the released execution path. |
| Prepared dataset | The immutable nnDetection raw/preprocessed task, splits, preprocessing config, and manifest used by training. |
| Dataset manifest | SHA-256 inventory binding data files, modality order, preprocessing config, and split file to one dataset version. |
| Run ID | Unique identity for one config, fold, seed, timestamp, and Git revision. |
| Primary matrix | The required 4 configurations × 5 official folds × 1 primary seed, 20 runs. Reducible only via the pre-committed ladder in `ARCHITECTURE.md §9` / ADR 0002 D10. |
| Durable checkpoint | A locally valid checkpoint and run state successfully synchronized to object storage. |
| S3-compatible | An object store usable through the AWS CLI S3 command contract, optionally with a custom endpoint URL. |
