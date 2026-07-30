# Phase 7 — Grad-CAM Explainability & Clinical Review

**Milestones:** M9, M10 (CAM figures) · **Depends on:** a trained GCALF-Net checkpoint (Phase 5) · **Blocks:** SOP 4 (urologist evaluation)
**Goal:** generate 3D Grad-CAM heatmaps for the trained model using **medcam**, render blinded review packets, and confirm the CAMs are wired to the right class + layer. Answers SOP 4 (three urologists, PI-RADS v2, 5-point Likert).

**Definition of done:**
- [ ] `medcam.inject` produces a 3D CAM of input spatial shape for a test case.
- [ ] CAM changes with target class (not constant); localizes to lesion for known positives.
- [ ] Blinded per-case review packets (PNG panels) exported for the urologists.

---

## 7.1 Use medcam (M3d-Cam) — don't hand-roll

`M3d-Cam/medcam/medcam_inject.py::inject` (verified) works on any `nn.Module`, 2D/3D, and writes NIfTI maps:
```python
from medcam import medcam
model = medcam.inject(
    model, output_dir="attention_maps",
    backend='gcam',        # 'gcam' | 'gcampp' | 'ggcam' | 'gbp'
    layer='auto',          # or explicit classifier-conv layer name
    label='best',          # or int class id / lambda discriminator
    save_maps=True, data_shape='default',
)
model.eval()
_ = model(input_volume)   # CAM written per forward pass — no torch.no_grad()
```

**Do not wrap the forward in `torch.no_grad()`.** Grad-CAM needs the backward pass; medcam's own diagnostic tells you so (`M3d-Cam/medcam/backends/grad_cam.py:165` — "check that no torch.no_grad statements effects medcam"). It re-enables gradients internally via `torch.enable_grad()`, so a stray `no_grad` may appear to work while silently breaking guided backends, which need `requires_grad=True` on the *input*. Leave gradients on; `model.eval()` is what you actually want for deterministic inference.

## 7.2 Where to attach

The last conv feeding the GGG classifier — the deepest BiFPN map used by `_build_head_classifier` (`nndet/ptmodule/retinaunet/base.py` ~line 570) / `nndet/arch/heads/classifier.py`. Pass that module's name as `layer=`.

**nnDetection caveat (important):** RetinaUNet's classification head is **per-anchor**, so:
- `layer='auto'` may latch onto a detection/regression layer, and the "score" is per-anchor, not a clean per-volume class logit.
- **Recommended path:** run Grad-CAM on the **classifier-fallback model** (`nndet/arch/encoder/gcalf/classifier_model.py`, `SPEC.md §14`) whose output is a clean `(B, num_classes)` vector — exactly what medcam expects. This is the cleanest input for the urologist study and sidesteps anchor bookkeeping.
- If you must explain the detection model: specify the classifier conv `layer=` explicitly and a `label=` lambda selecting the target lesion's anchor logits.

## 7.3 Backend choice

- **Grad-CAM (`gcam`)** — primary, localizes the class-discriminative region.
- **Grad-CAM++ (`gcampp`)** — better for multiple/spread lesions; generate both, let urologists compare.
- Guided-BP/Guided-Grad-CAM — optional finer detail; usually noisier for clinical review.

## 7.4 Rendering for clinical review (`gcalf_eval/gradcam.py` = thin medcam wrapper)

- **NIfTI:** re-header medcam's CAM into the input's affine/spacing → overlay in any NIfTI/DICOM viewer.
- **PNG panels:** axial slice through the lesion centroid; T2W grayscale + jet CAM at α=0.4; repeat for ADC and DWI (3 sequences × 2–3 slices).
- **Blinded packet (SOP 4):** per case, one panel image + predicted GGG, **no GT/prediction label shown** (masterfile §Likert: urologists rate blind), named by a random review id. Ship a rating sheet (5-point PI-RADS v2 alignment scale) alongside.
- Optional quantitative sanity: medcam's `evaluate=True` + a lesion mask → CAM-vs-lesion overlap (`metric='wioa'`), reported next to the Likert means.

## 7.5 Test plan (`tests/test_gradcam.py`)

- medcam injects and produces a non-empty CAM of input spatial shape;
- CAM for class c changes when `label` changes (assert not identical across two classes);
- CAM argmax roughly inside the lesion mask for a known strong positive (or `evaluate` overlap > chance);
- **non-destructive:** injected model's forward output equals the un-injected model's output (medcam only hooks).

## 7.6 Risks & fallbacks

| Risk | Fallback |
|---|---|
| medcam picks the wrong layer on the detection head | Specify `layer=` explicitly, or use the classifier-fallback model (recommended). |
| Per-anchor logits make `label` selection awkward | Classifier-fallback model → clean `(B,num_classes)`. |
| CAM too diffuse for clinical usefulness | Try Grad-CAM++; attach to a shallower conv for finer spatial detail; both are `backend=`/`layer=` swaps. |
| medcam version vs old torch | medcam is pure hooks (deps: nibabel, numpy) — installs cleanly into the pinned env. |

## 7.7 Deliverables & commit

- Branch `feat/gradcam` → commit "gradcam: medcam 3D CAM + blinded urologist review packets".
- Files: `gcalf_eval/gradcam.py`, `tests/test_gradcam.py`, review packets under `gcalf_experiments/_gradcam/`.
- Hand the blinded packets + rating sheets to the 3 urologists → collect Likert scores → report mean/SD (threshold ≥4.0 per masterfile) for SOP 4.

**Done.** This closes the milestone chain M0→M10. See `SPEC.md §15` for the full timeline and SOP mapping.
