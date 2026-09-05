# Phase 7 — Grad-CAM Explainability & Clinical Review

**Milestone:** M10 · **Depends on:** a trained GCALF-Net checkpoint (Phase 5) · **Blocks:** SOP 4
**Goal:** generate 3D Grad-CAM heatmaps using **medcam**, targeted at the grade head, render
blinded review packets, confirm CAMs are wired to the right class and layer. Answers SOP 4 (three
urologists, PI-RADS v2, 5-point Likert, 30 cases stratified across GGG2–5).

**Definition of done:**
- [ ] **L:** `medcam.inject` produces a 3D CAM of input spatial shape on a synthetic test case.
- [ ] **L:** CAM changes with target grade (not constant across classes).
- [ ] **V:** CAM localizes to the lesion for known real positives.
- [ ] **L:** saved registration/crop/resampling/padding metadata maps the CAM back to native T2W
      coordinates without using the ground-truth lesion transform.
- [ ] **V:** blinded per-case review packets exported for the urologists.

**Start recruiting the three urologists now, not at this phase.** They are the only dependency
this team does not control and they sit at the end of the chain (`docs/ROADMAP.md` M10 note).

---

## 7.1 Use medcam (M3d-Cam) — don't hand-roll

```python
from medcam import medcam
model = medcam.inject(
    model, output_dir="attention_maps",
    backend='gcam',        # 'gcam' | 'gcampp' | 'ggcam' | 'gbp'
    layer='auto',          # or the grade head's conv/linear layer name, explicit (recommended)
    label='best',          # or int grade id / lambda discriminator
    save_maps=True, data_shape='default',
)
model.eval()
_ = model(input_volume)   # CAM written per forward pass — no torch.no_grad()
```

**Do not wrap the forward in `torch.no_grad()`.** Grad-CAM needs the backward pass; medcam
re-enables gradients internally via `torch.enable_grad()`, so a stray `no_grad` may appear to work
while silently breaking guided backends, which need `requires_grad=True` on the input. Leave
gradients on; `model.eval()` is what you actually want for deterministic inference.

## 7.2 Where to attach — simpler than under the native-class design

Attach to the last conv/linear feeding **`GradeHead`** (`nndet/arch/encoder/gcalf/grade_head.py`,
Phase 2 §2.5), targeting the predicted grade of one matched positive detection via `label=`.

This is materially simpler than the fallback-model detour earlier plans needed: `GradeHead`'s
output is a clean `(N_detections, 4)` tensor per matched positive, not nnDetection's per-anchor
score across the entire dense anchor grid. Select the detection of interest first (the one
lesion being reviewed), then request its 4-way grade logits from `GradeHead` — no separate
classifier-fallback model is needed for this study.

## 7.3 Backend choice

- **Grad-CAM (`gcam`)** — primary, localizes the class-discriminative region.
- **Grad-CAM++ (`gcampp`)** — better for multiple/spread lesions; generate both, let urologists compare.
- Guided-BP/Guided-Grad-CAM — optional finer detail, usually noisier for clinical review.

## 7.4 Rendering for clinical review (`gcalf_eval/gradcam.py` = thin medcam wrapper)

- **NIfTI:** invert the recorded depth padding/crop, z resampling, gland-centred in-plane crop, and
  T2W-grid transform before assigning native T2W geometry. Re-headering alone is insufficient.
- **PNG panels:** axial slice through the lesion centroid; T2W grayscale + jet CAM at α=0.4; repeat
  for ADC and HBV.
- **Blinded packet (SOP 4):** per case, one panel image + predicted grade, **no GT/prediction
  label shown**, named by a random review id. Ship a 5-point PI-RADS v2 alignment rating sheet
  alongside.
- Optional quantitative sanity: medcam's `evaluate=True` + a lesion mask → CAM-vs-lesion overlap
  (`metric='wioa'`), reported next to the Likert means.

## 7.5 Case selection

30 held-out cases, stratified across GGG2–5, drawn only from grade-supervised, correctly-matched
detections (Phase 1's audit-recovered cohort). Oversample GGG4/GGG5 if the stratified draw would
otherwise return fewer than a handful — record the oversampling rule explicitly rather than
silently rebalancing.

## 7.6 Test plan (`tests/test_gradcam.py`)

- medcam injects and produces a non-empty CAM of input spatial shape;
- CAM for grade *g* changes when `label` changes (assert not identical across two grades);
- CAM argmax roughly inside the lesion mask for a known strong positive (or `evaluate` overlap > chance);
- **non-destructive:** injected model's forward output equals the un-injected model's output
  (medcam only hooks — verify this holds with the two-head model, not just the detection path).
- a synthetic landmark survives native→preprocessed→native coordinate round-trip within the
  declared interpolation tolerance.

## 7.7 Risks & fallbacks

| Risk | Fallback |
|---|---|
| medcam picks the wrong layer with `layer='auto'` | Specify `layer=` explicitly as the grade head's conv/linear — should rarely be needed given §7.2's clean attachment point, but confirm rather than assume. |
| CAM too diffuse for clinical usefulness | Try Grad-CAM++; attach to a shallower conv feeding the grade head for finer spatial detail. |
| medcam version vs. old torch | medcam is pure hooks (deps: nibabel, numpy) — installs cleanly into the pinned env. |
| Fewer than 30 grade-supervised, correctly-matched GGG4/5 cases available | Oversample within the available pool and state the deviation explicitly in the review packet's methods note. |

## 7.8 Deliverables & commit

- Branch `feat/gradcam` → commit "gradcam: medcam 3D CAM on grade head + blinded urologist review packets".
- Files: `gcalf_eval/gradcam.py`, `tests/test_gradcam.py`, review packets under
  `gcalf_experiments/_gradcam/`.
- Hand blinded packets + rating sheets to the 3 urologists → collect Likert scores → report
  mean/SD and two-way random-effects absolute-agreement ICC for SOP 4.

**Done.** This closes the milestone chain M0→M11. See `docs/ROADMAP.md` M11 for the final
reproducibility audit and thesis-output checklist.
