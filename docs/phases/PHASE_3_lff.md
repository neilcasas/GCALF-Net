# Phase 3 — Learnable Frequency Filter

**Milestone:** M6 | **Depends on:** Phase 2 (FDR+WAF baseline, tagged `baseline-v1`) | **Blocks:** full GCALF-Net

## Goal

Replace **FDR**'s fixed radial mask with one learned, real-valued gain — the *only* change from
the frozen baseline (`ARCHITECTURE.md §6`), so this ablation is a single-variable comparison.
FDR stays the registry default so this refactor cannot move the baseline.

## Exit Criteria

- [ ] **L:** Hydra passes `model_cfg.encoder_kwargs.gcalf_cfg` into `Encoder`, unchanged from Phase 2.
- [ ] **L:** baseline construction (`frequency_filter_type: fdr`) and a fixed-seed output
      regression test are unchanged after adding `lff` as a registry option.
- [ ] **L:** LFF is identity at initialization, supports variable spatial shapes, has nonzero gradients.
- [ ] **L:** the learned grid is oriented in **centered** frequency coordinates (a centre-weighted
      grid low-passes).
- [ ] **V:** `lff_only` completes tiny-task train, resume, predict, and evaluation.
- [ ] **V:** peak memory recorded for each configured LFF stage.

## 3.1 Design

Per `ARCHITECTURE.md §6` — a grouped, low-resolution real-valued spectral grid, interpolated to
the runtime `rfftn` shape, defined on centered frequency coordinates:

```python
class LearnableFrequencyFilter3D(nn.Module):
    """Grouped, shape-tolerant, zero-phase, real-valued learnable frequency filter.

    The gain grid is stored on CENTERED frequency coordinates for the two full
    FFT axes (D, H); the trailing rFFT axis already runs DC -> Nyquist.
    """

    def __init__(self, in_channels, out_channels, grid_size=(4, 8, 8), groups=8):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        self.in_channels = in_channels
        self.groups = groups
        self.delta_weight = nn.Parameter(torch.zeros(1, groups, *grid_size))  # real, zero-init -> identity
        self.proj = (nn.Identity() if in_channels == out_channels
                     else nn.Conv3d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            spectrum = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
            delta = F.interpolate(self.delta_weight, size=spectrum.shape[-3:],
                                   mode="trilinear", align_corners=True)
            delta = torch.fft.ifftshift(delta, dim=(-3, -2))   # centered grid -> rfftn's ordering
            gain = (1.0 + delta).repeat_interleave(self.in_channels // self.groups, dim=1)
            filtered = torch.fft.irfftn(spectrum * gain, s=x.shape[-3:],
                                         dim=(-3, -2, -1), norm="ortho")
        return self.proj(filtered.to(input_dtype))
```

**Two properties, both cheap to fix and both invisible without the orientation-gate test in §3.3:**

1. **Frequency-axis orientation.** `rfftn` orders the two full axes (`D`,`H`) as
   `[0, +f, …, Nyquist, −f, …, −1]`; only the trailing rFFT axis runs monotonically DC→Nyquist.
   Interpolating a coarse grid straight onto that raw index space cannot express a low-pass
   response and smooths across the Nyquist discontinuity. Define the grid in **centered**
   coordinates and `ifftshift` the interpolated result along `D,H` before multiplying — exactly
   the same class of defect FDR's own fixed mask must avoid (`PHASE_2_baseline.md §2.2`).
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
with the `fdr` branch):

```python
if kind == "lff":
    return LearnableFrequencyFilter3D(in_channels, out_channels, **options)
```

No `use_lff` boolean aliases — one canonical `frequency_filter_type` enum. `Encoder.forward`
stays generic; it calls whatever the registry returns, FDR or LFF, identically.

## 3.3 Correctness tests (`tests/gcalf/test_lff.py`) — L gate

- `delta_weight == 0`, equal channels: output equals input within `atol=1e-4`.
- Odd and even `(D,H,W)` shapes preserve shape.
- Backpropagation gives finite, nonzero `delta_weight.grad`.
- Deliberately different low/high grid values produce different gains for synthetic low/high
  frequency inputs.
- **Orientation gate** (the single most valuable test in this file): set the grid to `1` at its
  centre and `−1` elsewhere, then feed (i) a constant volume and (ii) a Nyquist checkerboard
  `(-1)**(d+h+w)`. The constant must survive; the checkerboard must be suppressed. With the wrong
  axis convention this test inverts.
- Float16 CUDA input returns float16 finite output while the FFT itself executes in float32.
- Invalid channel/group combinations fail with a useful error.
- Serialization and reload produce the same output.
- **Fixed-seed regression test**: with `frequency_filter_type: fdr` (baseline), encoder output is
  byte-identical before and after this phase's registry changes — proves the refactor didn't move
  the baseline.

Do not test an all-ones filter plus a spatial residual — that formulation returns approximately
`2x`, not `x` (the same trap as CAF's residual-counted-twice defect, `PHASE_4_caf.md §4.4`).

## 3.4 V gate — integration sequence

1. Load the `baseline-v1` checkpoint and compare fixed-input outputs before/after this refactor.
2. Add LFF and its unit tests (above).
3. Run one forward/backward pass at each real stage shape (`[1,3,4]`); record peak memory.
4. Overfit two cases (reuse Phase 2's cases), then run `Task900_PICAI_TINY` for two epochs.
5. Resume the tiny run from checkpoint; run prediction/evaluation.
6. Launch all five folds only after the tiny gate passes.

## Risks

| Risk | Mitigation |
|---|---|
| FFT activation memory too high | Reduce patch size, or drop stage 1 only as an explicitly reported architecture change. |
| Grid interpolation creates artifacts | Compare radial response plots and synthetic frequency tests; raise grid resolution only after profiling. |
| Frequency axes silently transposed/unshifted | The orientation gate fails loudly; plot the learned response on centered axes before reporting any "learned low-pass" claim. |
| AMP or cuFFT instability | Keep FFT and the complex multiply inside disabled autocast; assert finite values. |
| LFF does not beat FDR | Preserve as a valid ablation result; do not tune on the test fold. |

**Deliverables:** `nndet/arch/encoder/gcalf/lff.py`, frequency registry entry, Hydra config, tests,
response visualization, tiny-run metrics, memory report.

**Next:** `PHASE_4_caf.md`.
