# Phase 3 — Learnable Frequency Filter

**Milestone:** M6 | **Depends on:** Phase 2 (FDSF+WAF baseline, tagged `baseline-v1`) | **Blocks:** full GCALF-Net

## Goal

Replace input-level **FDSF**'s fixed radial response with one learned, real-valued response at the
same input location, emitting the same complementary `(x_low, x_high)` pair — the *only* change
from the frozen baseline (`ARCHITECTURE.md §6`), so this ablation is a single-variable comparison.
LFF is a learned *separation*, not a filter; matching FDSF's arity is what keeps `lff_only` a swap
rather than a different pipeline. FDSF stays the registry default so this refactor cannot move the
baseline. The five-level encoder, frozen fusion locations, WAF, BiFPN, heads, preprocessing, and
schedule remain unchanged.

## Exit Criteria

- [ ] **L:** Hydra passes `model_cfg.encoder_kwargs.gcalf_cfg` into `Encoder`, unchanged from Phase 2.
- [ ] **L:** baseline construction (`frequency_filter_type: fdsf`) and a fixed-seed output
      regression test are unchanged after adding `lff` as a registry option.
- [ ] **L:** at `delta_weight=0`, LFF's `(x_low, x_high)` equals FDSF's output — not merely
      `x_low ≈ x` — supports variable spatial shapes, and has nonzero gradients through both outputs.
- [ ] **L:** `x_low + x_high == x` to `atol=1e-5`, the same lossless-split assertion FDSF passes.
- [ ] **L:** the learned grid is oriented in **centered** frequency coordinates (a centre-weighted
      grid low-passes).
- [ ] **V:** `lff_only` completes tiny-task train, resume, predict, and evaluation.
- [ ] **V:** peak memory recorded for the full input-level LFF operation.

## 3.1 Design

Per `ARCHITECTURE.md §6` — a grouped, low-resolution real-valued spectral grid, interpolated to
the runtime `rfftn` shape, defined on centered frequency coordinates:

```python
class LearnableFrequencyFilter3D(nn.Module):
    """Grouped, shape-tolerant, zero-phase, real-valued learnable frequency separation.

    Same (x_low, x_high) arity as FDSF, and the SAME MODULE at delta_weight=0:
    the learned response is parameterized as FDSF's spherical mask plus a
    zero-initialized delta, so `lff_only` starts from the frozen baseline.

    The gain grid is stored on CENTERED frequency coordinates for the two full
    FFT axes (D, H); the trailing rFFT axis already runs DC -> Nyquist.
    """

    def __init__(self, in_channels, radius=0.15, grid_size=(4, 8, 8), groups=1):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        self.in_channels = in_channels
        self.groups = groups
        self.radius = radius                                    # == FDSF's radius
        self.delta_weight = nn.Parameter(torch.zeros(1, groups, *grid_size))  # real, zero-init

    def forward(self, x):
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            spectrum = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
            base = spherical_mask_rfft(x.shape[-3:], self.radius,
                                       device=x.device)         # FDSF's M, centered on D,H
            delta = F.interpolate(self.delta_weight, size=spectrum.shape[-3:],
                                   mode="trilinear", align_corners=True)
            gain = (base + delta).repeat_interleave(self.in_channels // self.groups, dim=1)
            gain = torch.fft.ifftshift(gain, dim=(-3, -2))       # centered grid -> rfftn's ordering
            x_low = torch.fft.irfftn(spectrum * gain, s=x.shape[-3:],
                                      dim=(-3, -2, -1), norm="ortho")
        x_low = x_low.to(input_dtype)
        return x_low, x - x_low                                 # complementary by construction
```

**Two properties, both cheap to fix and both invisible without the orientation-gate test in §3.3:**

1. **Frequency-axis orientation.** `rfftn` orders the two full axes (`D`,`H`) as
   `[0, +f, …, Nyquist, −f, …, −1]`; only the trailing rFFT axis runs monotonically DC→Nyquist.
   Interpolating a coarse grid straight onto that raw index space cannot express a low-pass
   response and smooths across the Nyquist discontinuity. Define the grid in **centered**
   coordinates and `ifftshift` the interpolated result along `D,H` before multiplying — exactly
   the same class of defect FDSF's own fixed mask must avoid (`PHASE_2_baseline.md §2.2`).
2. **A complex gain is not identifiable.** `x` real ⟹ `X` Hermitian; multiplying by a
   non-Hermitian complex `H` and calling `irfftn` applies the Hermitian projection
   `(H(f)+conj(H(−f)))/2` — half the learned parameters alias onto the other half and cannot be
   recovered. A real gain is exactly Hermitian by construction, halves the parameter count, stays
   frequency-selective, and needs no phase-recovery argument. Learning phase is a documented
   extension, never the default.

Notes on the implementation above, to validate rather than assume: `align_corners=True` puts the
extreme grid values on DC and Nyquist — for even `D`/`H` the grid centre lands within half a bin
of the `ifftshift` DC position; confirm on a radial response plot rather than assuming. Complex ×
real broadcasting, `ifftshift`, and `irfftn` all exist in the pinned PyTorch 1.10 image — verify
in the container anyway.

## 3.2 Registry integration

Add to `nndet/arch/encoder/gcalf/registry.py::build_frequency_module` (already stubbed in Phase 2
with the `fdsf` branch):

```python
if kind == "lff":
    return LearnableFrequencyFilter3D(in_channels, **options)
```

No `use_lff` boolean aliases — one canonical `frequency_filter_type` enum. `Encoder.forward`
stays generic: it calls the selected frequency mechanism once before the five-level encoder and
unpacks `(x_low, x_high)` identically for FDSF and LFF, routing `x_low` to Swin and `x_high` to
CNN. Nothing downstream of the unpack knows which module ran.

## 3.3 Correctness tests (`tests/gcalf/test_lff.py`) — L gate

- `delta_weight == 0`: `(x_low, x_high)` equals `FrequencyDomainSeparationAndShunting3D(radius)`'s
  output within `atol=1e-5` — the init-equals-baseline gate. Asserting `x_low ≈ x` instead would
  pass for a filter that does no separation at all.
- `x_low + x_high == x` within `atol=1e-5`, for odd and even `(D,H,W)`, matching FDSF's own
  lossless-split assertion.
- Both outputs preserve shape and channel count for odd and even `(D,H,W)`.
- Backpropagation through **each** output gives finite, nonzero `delta_weight.grad`.
- Deliberately different low/high grid values produce different gains for synthetic low/high
  frequency inputs.
- **Orientation gate** (the single most valuable test in this file): set the grid so the response
  is `1` at its centre and `−1` elsewhere, then feed (i) a constant volume and (ii) a Nyquist
  checkerboard `(-1)**(d+h+w)`. The constant must land in `x_low`; the checkerboard must land in
  `x_high`. With the wrong axis convention this test inverts.
- Float16 CUDA input returns float16 finite output while the FFT itself executes in float32.
- Invalid channel/group combinations fail with a useful error.
- Serialization and reload produce the same output.
- **Fixed-seed regression test**: with `frequency_filter_type: fdsf` (baseline), encoder output is
  byte-identical before and after this phase's registry changes — proves the refactor didn't move
  the baseline.

Do not test an all-ones filter plus a spatial residual — that formulation returns approximately
`2x`, not `x` (the same trap as CAF's residual-counted-twice defect, `PHASE_4_caf.md §4.4`).

## 3.4 V gate — integration sequence

1. Load the `baseline-v1` checkpoint and compare fixed-input outputs before/after this refactor.
2. Add LFF and its unit tests (above).
3. Run one forward/backward pass at the planner's resolved patch size, and once at the worst-case
   `(1,3,32,256,256)` smoke shape; record peak memory for both and confirm exactly five downstream
   encoder outputs.
4. Overfit two cases (reuse Phase 2's cases), then run `Task900_PICAI_TINY` for two epochs.
5. Resume the tiny run from checkpoint; run prediction/evaluation.
6. Launch all five folds only after the tiny gate passes.

## Risks

| Risk | Mitigation |
|---|---|
| FFT activation memory too high | Reduce the common patch size before the matrix; do not move LFF to a different location than FDSF. |
| Grid interpolation creates artifacts | Compare radial response plots and synthetic frequency tests; raise grid resolution only after profiling. |
| Frequency axes silently transposed/unshifted | The orientation gate fails loudly; plot the learned response on centered axes before reporting any "learned low-pass" claim. |
| AMP or cuFFT instability | Keep FFT and the complex multiply inside disabled autocast; assert finite values. |
| LFF does not beat FDSF | Preserve as a valid ablation result; do not tune on the test fold. |

**Deliverables:** `nndet/arch/encoder/gcalf/lff.py`, frequency registry entry, Hydra config, tests,
response visualization, tiny-run metrics, memory report.

**Next:** `PHASE_4_caf.md`.
