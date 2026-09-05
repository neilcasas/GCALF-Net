# Phase 2 — Build FDR + Wire WAF + Masked Grade Head (the baseline)

**Milestones:** M2 (FDR+WAF built) → M3 (forward + grade-head overfit) → M4 (full train/eval ⭐) → M5 (budget ladder decision)
**Depends on:** Phase 1 · **Blocks:** LFF/CAF (they must each beat this on a single changed variable)

**Goal:** unlike the released-code baseline PDHD-Net shipped with (wavelet + channel-light
fusion), the thesis's defended hypotheses name a fixed spherical frequency mask (FDR) and
window-based self-attention (WAF) — **neither exists in the code**, in either `PDHD-Net/` or
`GCALF-Net/` (`ARCHITECTURE.md §0`). This phase builds them. It is the largest new-code item in
the whole roadmap; do not treat it as a config flip.

---

## 2.1 What actually needs to change (read this before touching `modular.py`)

| Item | Current state | This phase |
|---|---|---|
| Frequency module, stages `[1,3,4]` | `WaveletSpatialFusion` (Haar wavelet, `WaveletFusion.py`) | Replace with new `FDR` (fixed FFT low/high split) |
| Fusion module, stages `[2,5]` | `MemoryEfficientFusion`/`ChannelWiseLightFusion` (ECA-style) | Replace with `WindowAttentionFusion` (already written, `window_attention_fusion.py`, imported but dead) |
| Classifier head | Native `classifier_classes` foreground grades (rejected design) | One `csPCa` detection class (nnDetection's existing focal-loss anchor head, unchanged) + a **new** separate 4-logit grade head |
| Dead parameters | `modular.py:158-165` constructs a fusion module for every stage but invokes only `[2,5]` | Fix while wiring WAF — do not carry four unused modules into every checkpoint |

## 2.2 Build FDR (`nndet/arch/encoder/gcalf/fdr.py`)

Per `ARCHITECTURE.md §5`:
```python
class FrequencyDomainRefinement3D(nn.Module):
    """Fixed spherical low/high FFT split. FDR is what the thesis's H1/H2 name as the baseline."""

    def __init__(self, radius=0.15):
        super().__init__()
        self.radius = radius

    def forward(self, x):
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            X = torch.fft.fftshift(torch.fft.fftn(x.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
            mask = self._radial_mask(X.shape[-3:], self.radius, device=X.device)
            X_low, X_high = X * mask, X * (1 - mask)
            low = torch.fft.ifftn(torch.fft.ifftshift(X_low, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
            high = torch.fft.ifftn(torch.fft.ifftshift(X_high, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
        return low.to(input_dtype), high.to(input_dtype)

    @staticmethod
    def _radial_mask(shape, radius, device):
        d, h, w = shape
        dd = torch.linspace(-1, 1, d, device=device).view(-1, 1, 1)
        hh = torch.linspace(-1, 1, h, device=device).view(1, -1, 1)
        ww = torch.linspace(-1, 1, w, device=device).view(1, 1, -1)
        r = torch.sqrt(dd**2 + hh**2 + ww**2)
        return (r <= radius).float()
```
Route `X_low` toward the Swin branch input and `X_high` toward the CNN branch input — **confirm
this exact routing direction against the thesis text before wiring `modular.py`**; if the thesis
specifies the reverse, swap it there, not here.

**Unit tests (`tests/gcalf/test_fdr.py`):** mask is exactly `1` at `r=0` and exactly `0` outside
`radius` (boundary case at `r==radius` inclusive, matching `<=`); `X_low + X_high` reconstructs
`X` losslessly (complementary, no energy loss); shape preserved for odd and even `(D,H,W)`; a
constant volume passes almost entirely into `X_low`; a Nyquist checkerboard `(-1)**(d+h+w)` passes
almost entirely into `X_high` — **this is the orientation gate that catches a missing
`fftshift`/`ifftshift` pair**, the single most valuable test in the file, same defect class as
LFF's orientation gate (`PHASE_3_lff.md §3.4`).

## 2.3 Wire WAF (`nndet/arch/encoder/gcalf/waf.py`)

`WindowAttentionFusion` already exists at `nndet/arch/encoder/window_attention_fusion.py` and is
already imported at `modular.py:11` — this is wiring, not writing:

```python
from nndet.arch.encoder.window_attention_fusion import WindowAttentionFusion

def build_waf(cnn_channels, transformer_channels, out_channels, window_size=(2, 7, 7), num_heads=4):
    return WindowAttentionFusion(cnn_channels, transformer_channels, out_channels,
                                  window_size=window_size, num_heads=num_heads)
```
Read `WindowAttentionFusion.forward`'s existing signature before writing this wrapper — match it
exactly rather than guessing. Replace the `MemoryEfficientFusion` construction at stages `[2,5]`
with this, and **stop constructing fusion modules at stages `0,1,3,4`** (`modular.py:158-165`'s
standing dead-parameter defect) — construct `nn.Identity()` there instead, for every fusion
variant including WAF/CAF, not only the released baseline.

**Unit tests (`tests/gcalf/test_waf.py`):** output shape matches CNN spatial dims; gradients reach
both input projections; a fixed-seed run matches `WindowAttentionFusion`'s own pre-existing
behavior (this module isn't new — confirm the wrapper doesn't change its numerics).

## 2.4 Registry (`nndet/arch/encoder/gcalf/registry.py`)

```python
def build_frequency_module(kind, in_channels, out_channels, options=None):
    options = options or {}
    if kind == "fdr":
        return FrequencyDomainRefinement3D(**options)
    if kind == "lff":
        return LearnableFrequencyFilter3D(in_channels, out_channels, **options)
    raise ValueError(f"Unknown frequency_filter_type: {kind}")

def build_fusion_module(kind, cnn_channels, transformer_channels, out_channels, options=None):
    options = options or {}
    if kind == "waf":
        return build_waf(cnn_channels, transformer_channels, out_channels, **options)
    if kind == "caf":
        return WindowedCrossAttentionFusion3D(cnn_channels, transformer_channels, out_channels, **options)
    raise ValueError(f"Unknown fusion_type: {kind}")
```
`modular.py` calls only these two functions; it never names `FrequencyDomainRefinement3D` or
`WindowAttentionFusion` directly. Baseline defaults: `frequency_filter_type: fdr`,
`fusion_type: waf`, `freq_stages: [1,3,4]`, `fusion_stages: [2,5]`.

## 2.5 Masked grade head (`nndet/arch/encoder/gcalf/grade_head.py`)

Per `ARCHITECTURE.md §8`. A 4-logit head reading each matched positive detection's feature,
parallel to (not replacing) nnDetection's existing single-class `csPCa` anchor classifier:

```python
class GradeHead(nn.Module):
    """4-logit GGG2-5 head over matched positive detections. Loss is masked
    to grade_supervised instances only; unsupervised positives and all
    negatives contribute zero grade loss and zero grade-head gradient."""

    def __init__(self, in_channels, num_grades=4):
        super().__init__()
        self.classifier = nn.Linear(in_channels, num_grades)

    def forward(self, features):
        return self.classifier(features)


def grade_loss(logits, grade_targets, grade_supervised_mask, class_weights):
    if not grade_supervised_mask.any():
        return logits.new_zeros(())          # legitimate: no supervised lesion in this batch
    supervised_logits = logits[grade_supervised_mask]
    supervised_targets = grade_targets[grade_supervised_mask]
    return F.cross_entropy(supervised_logits, supervised_targets, weight=class_weights)
```
Derive `class_weights` (inverse-frequency, normalized to mean 1) from **each fold's training
partition only**, recomputed per fold — never from the global cohort. Log the count of
grade-supervised lesions contributing to every batch; a batch with zero such lesions is expected,
not a bug, and must not raise or silently skip the rest of the loss.

**Unit tests (`tests/gcalf/test_grade_head.py`):** a batch with zero supervised lesions returns a
zero loss with no gradient anywhere in `GradeHead`; a batch mixing supervised and unsupervised
positives updates `GradeHead` parameters only from the supervised subset (verify by comparing
gradients against a hand-computed reference on a tiny synthetic batch); class weights change
per-fold when given different training partitions; the head's output shape is always
`(N_detections, 4)` regardless of how many are supervised.

## 2.6 M2 — forward pass & grade-head overfit

1. **Forward**: `torch.randn(1,3,32,256,256)` (Phase 1's fixed crop shape) through the encoder
   with FDR+WAF; assert no shape error through encoder → BiFPN → detection head → grade head.
2. **Overfit 2 real cases** — one grade-supervised positive, one benign:
   - Train 200 steps; assert final 10-step mean total loss ≤10% of the initial mean.
   - The positive case's matched detection predicts `csPCa` at ≥0.9 and its correct GGG2–5 grade
     at ≥0.9.
   - The benign case's patch stays at ≤0.1 foreground probability, **and** — the test unique to
     this design — assert `GradeHead`'s parameters receive **zero** gradient from the benign-only
     batch (a leak here means the masking in §2.5 is wrong, not that the model needs more data).

This is the single most valuable test in the whole plan — it proves data→label→loss→backprop is
wired correctly end to end, including the mask that the native-4-class design didn't need.

## 2.7 M3 — tiny end-to-end (plumbing, not accuracy)

Reuse `nndet/conf/train/smoke.yaml` (`max_num_epochs: 2`, `num_train_batches_per_epoch: 20`,
`swa_epochs: 2`) and `scripts/run_m3_smoke.sh`, updated for the two-head model:
```bash
scripts/run_m3_smoke.sh "$det_data/Task2201_PICAI_csPCa" "$det_data/Task900_PICAI_TINY"
```
Six deterministic cases: one grade-supervised lesion per grade 2–5, one Pooch25 ungraded positive,
one benign. Assert `classifier_classes == 1` (not 4 or 5 — this changed under the new design) and
that the grade head's output has 4 logits per detection. Exit when `metrics.csv` is populated;
numbers are meaningless on 6 cases.

## 2.8 M4 — full baseline train/eval ⭐

- Train the shipped schedule per fold, unchanged: 50 epochs × 2500 batches + 10 SWA, SGD
  `initial_lr: 0.01` poly decay, `precision: 16`. No `EarlyStopping` — a per-config stopping rule
  would invalidate the four-way comparison later.
- **Record measured seconds/step and total wall time for fold 0** — this is the pilot number M5
  turns into a committed budget rung.
- Class weights for the grade loss derived per-fold from that fold's training partition only
  (§2.5); detection/segmentation losses unchanged from nnDetection's defaults.
- Predict + evaluate per `PHASE_6_evaluation.md`: FROC/case-level AUROC over all 1,500 cases,
  4×4 grade confusion matrix over grade-supervised matched lesions, weighted F1, per-grade
  sensitivity, both denominators (detection, grade-matched) reported side by side.
- **Tag `baseline-v1`.** Every later config (LFF, CAF, full) is compared against this exact commit
  and these numbers. Report plainly in the thesis that this baseline is FDR+WAF as specified by
  the paper's hypotheses, not the released `WaveletSpatialFusion`/`ChannelWiseLightFusion` code
  (ADR 0002 D4) — and if it underperforms the released code, report that rather than switching.

## 2.9 M5 — budget ladder decision

Turn fold 0's measured seconds/step into `projected_hours = seconds_per_step * 150_000 / 3600`
and `projected_cost = projected_hours * 20 * hourly_rate`. Compare against the $150–350 planned
budget (ADR 0002 D10) and select, before any other fold starts:
1. **Full matrix** (planned rung) — 20 runs at the shipped schedule.
2. **Halved schedule, all 20 runs** — `num_train_batches_per_epoch: 1250` for every run, applied
   identically.
3. **14-run reduced matrix** — all four configs on fold 0, plus baseline and full on all five
   folds. Never drop folds for only some configs — it breaks the paired statistics in Phase 6.

Record the chosen rung in every subsequent `run.json`. Choosing after seeing later fold results is
test-set tuning, not budgeting.

## 2.10 Risks & fallbacks

| Risk | Fallback |
|---|---|
| FDR/WAF build slips past its gate | Step down the M5 ladder rather than compressing LFF/CAF phases; do not skip the M2/M3 gates to catch up. |
| Grade head leaks gradient from unsupervised lesions | The M3 overfit test catches this directly — fix the mask in §2.5, do not add more training data to compensate. |
| OOM at the plan's patch size | Reduce patch size in the plan; batch 1 + grad-accum; AMP everywhere except the FDR/LFF FFT block. |
| Baseline numbers implausibly low | Re-run the M3 overfit test; re-check Phase 1's label mapping and channel order before touching the model. |

## 2.11 Deliverables & commit

- Branch `feat/baseline-fdr-waf` → tag `baseline-v1`.
- Files: `nndet/arch/encoder/gcalf/{fdr,waf,grade_head,registry}.py`,
  `tests/gcalf/test_{fdr,waf,grade_head}.py`, `tests/test_overfit.py`, baseline `metrics.csv` +
  confusion matrix under `gcalf_experiments/baseline_seedNN/`, `docs/CLOUD_DEPLOYMENT_PLAN.md`'s
  `run.json` recording the M5 ladder decision.

**Next:** `PHASE_3_lff.md`.
