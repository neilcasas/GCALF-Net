# Phase 2 — Build Input-Level FDSF + Wire Five-Level WAF + Masked Grade Head (the baseline)

**Milestones:** M2 (FDSF+WAF built) → M3 (forward + grade-head overfit) → M4 (full train/eval ⭐) → M5 (budget ladder decision)
**Depends on:** Phase 1 · **Blocks:** LFF/CAF (they must each beat this on a single changed variable)

**Goal:** unlike the released-code baseline PDHD-Net shipped with (wavelet + channel-light
fusion), the thesis's defended hypotheses name fixed 3D frequency-domain separation and shunting
(FDSF) and
window-based self-attention (WAF) — **neither exists in the code**, in either `PDHD-Net/` or
`GCALF-Net/` (`ARCHITECTURE.md §0`). This phase builds them. It is the largest new-code item in
the whole roadmap; do not treat it as a config flip.

---

## 2.1 What actually needs to change (read this before touching `modular.py`)

| Item | Current state | This phase |
|---|---|---|
| Frequency routing | Stage-local `WaveletSpatialFusion` (Haar wavelet) | Remove it from the experimental path; apply new fixed FDSF once to the input before both branches |
| Fusion module | `MemoryEfficientFusion`/`ChannelWiseLightFusion` (ECA-style), currently hard-coded at `[2,5]` in a six-stage path | Replace with WAF at corresponding levels of the fixed five-level encoder |
| Classifier head | Native `classifier_classes` foreground grades (rejected design) | One `csPCa` detection class (nnDetection's existing focal-loss anchor head, unchanged) + a **new** separate 4-logit grade head |
| Level contract | Planner may produce five or six levels (`modular.py:63`; `c002.py:196-204`); `BiFPN.py:224-225` unpacks `p3..p7, _` and drops the sixth. On a six-level plan that discarded input is the stage-5 fusion output, so the wired `[2,5]` fusion contributes through stage 2 only; on a five-level plan stage 5 never runs. | Assert exactly five encoder outputs and require all five to reach BiFPN |
| Dead parameters | Current construction can instantiate unused fusion modules | Construct fusion only at the five resolved locations (or the one predeclared shared subset) |

## 2.2 Build FDSF (`nndet/arch/encoder/gcalf/fdsf.py`)

Per `ARCHITECTURE.md §5`:
```python
class FrequencyDomainSeparationAndShunting3D(nn.Module):
    """Fixed spherical low/high FFT split applied once to the model input."""

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
Route `X_low` toward the Swin branch input and `X_high` toward the CNN branch input. This is fixed
by the PDHD-Net paper and is not an implementation-time choice.

**Unit tests (`tests/gcalf/test_fdsf.py`):** mask is exactly `1` at `r=0` and exactly `0` outside
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
Read `WindowAttentionFusion.forward`'s existing signature before writing this wrapper—match it
exactly rather than guessing. Build a fixed five-level encoder and confirm it emits, and BiFPN
consumes, exactly five features.

`fusion_levels` starts at `[0,1,2,3,4]` but is **settled by measurement, not by the config
default** (`ARCHITECTURE.md §4`, ADR 0002 D5). Level 0 carries no stride, so fusing there means
windowed attention on the full-resolution map plus a trilinear upsample of the Swin feature to
match — the near-full-resolution 3D attention ADR 0001 flagged as the primary OOM risk. Profile
every level at the planner's resolved patch size in §2.6's V gate **before any comparative fold is
trained**, freeze the subset that measurement selects, and use it unchanged for WAF and CAF in all
four arms. Construct fusion modules only at the frozen locations.

**Unit tests (`tests/gcalf/test_waf.py`):** output shape matches CNN spatial dims; gradients reach
both input projections; a fixed-seed run matches `WindowAttentionFusion`'s own pre-existing
behavior (this module isn't new — confirm the wrapper doesn't change its numerics).

## 2.4 Registry (`nndet/arch/encoder/gcalf/registry.py`)

```python
def build_frequency_module(kind, in_channels, options=None):
    """Both branches return (x_low, x_high) and preserve in_channels."""
    options = options or {}
    if kind == "fdsf":
        return FrequencyDomainSeparationAndShunting3D(**options)
    if kind == "lff":
        return LearnableFrequencyFilter3D(in_channels, **options)
    raise ValueError(f"Unknown frequency_filter_type: {kind}")

def build_fusion_module(kind, cnn_channels, transformer_channels, out_channels, options=None):
    options = options or {}
    if kind == "waf":
        return build_waf(cnn_channels, transformer_channels, out_channels, **options)
    if kind == "caf":
        return WindowedCrossAttentionFusion3D(cnn_channels, transformer_channels, out_channels, **options)
    raise ValueError(f"Unknown fusion_type: {kind}")
```
`modular.py` calls only these two functions; it never names the concrete FDSF/LFF or WAF/CAF
classes directly. Baseline defaults: `frequency_filter_type: fdsf`, `fusion_type: waf`,
`num_levels: 5`, `fusion_levels: [0,1,2,3,4]` (starting point — frozen by §2.6's profiling). The
frequency module runs once at input and has no `freq_stages` setting.

**Both frequency branches must share one output contract:** `(x_low, x_high)`, each with the
input's shape and channel count, `x_low + x_high == x`. `Encoder.forward` unpacks the pair and
routes `x_low` to Swin and `x_high` to CNN without knowing which module produced it. A
single-tensor LFF would leave `lff_only` with no shunting and turn a one-variable swap into a
different pipeline — see `ARCHITECTURE.md §6` and ADR 0002 D4. There is no `out_channels`
parameter: the frequency stage sits before the encoder and cannot change channel count.

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

1. **Forward**: `torch.randn(1,3,32,256,256)` through the encoder with FDSF+WAF; assert exactly
   five encoder tensors reach BiFPN with no shape error through the detection and grade heads.
   This is a deliberate **worst-case smoke shape, not the training shape** — the raw task's
   geometry is trimmed by `crop_to_nonzero`, resampled by the planner, and then patched, so real
   batches arrive at the planner's resolved patch size (`PHASE_1 §1.2`).
2. **V: profile fusion levels, then freeze them.** Record CUDA peak memory and step time for WAF at
   every candidate level, at the planner's resolved patch size, batch 1. Declare the acceptance
   criterion first (e.g. "fits the target GPU with ≥10% headroom"). The measurement selects the
   frozen `fusion_levels`; record it in the M2 gate record and in every `config_snapshot.yaml`.
   This happens **before** M4, and the frozen subset is reused verbatim for CAF in Phase 4 — a
   subset chosen after seeing fold results is test-set tuning (§2.9's rule, applied to
   architecture).
3. **Overfit 2 real cases** — one grade-supervised positive, one benign:
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
- Predict + evaluate per `PHASE_6_evaluation.md`: FROC/case-level AUROC over all 1,499 retained cases,
  4×4 grade confusion matrix over grade-supervised matched lesions, weighted F1, per-grade
  sensitivity, both denominators (detection, grade-matched) reported side by side.
- **Tag `baseline-v1`.** Every later config (LFF, CAF, full) is compared against this exact commit
  and these numbers. Report plainly in the thesis that this baseline is FDSF+WAF as specified by
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
| FDSF/WAF build slips past its gate | Step down the M5 ladder rather than compressing LFF/CAF phases; do not skip the M2/M3 gates to catch up. |
| Grade head leaks gradient from unsupervised lesions | The M3 overfit test catches this directly — fix the mask in §2.5, do not add more training data to compensate. |
| OOM at the plan's patch size | Reduce patch size; batch 1 + accumulation; reduce WAF/CAF levels only as one predeclared shared subset; keep AMP off only inside the FDSF/LFF FFT block. |
| Baseline numbers implausibly low | Re-run the M3 overfit test; re-check Phase 1's label mapping and channel order before touching the model. |

## 2.11 Deliverables & commit

- Branch `feat/baseline-fdsf-waf` → tag `baseline-v1`.
- Files: `nndet/arch/encoder/gcalf/{fdsf,waf,grade_head,registry}.py`,
  `tests/gcalf/test_{fdsf,waf,grade_head}.py`, `tests/test_overfit.py`, baseline `metrics.csv` +
  confusion matrix under `gcalf_experiments/baseline_seedNN/`, `docs/CLOUD_DEPLOYMENT_PLAN.md`'s
  `run.json` recording the M5 ladder decision.

**Next:** `PHASE_3_lff.md`.
