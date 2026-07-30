# Phase 3 - Learnable Frequency Filter

**Milestone:** M5 | **Depends on:** Phase 2 baseline | **Blocks:** full GCALF-Net

## Goal

Replace `WaveletSpatialFusion` at encoder stages `[1, 3, 4]` with a genuinely frequency-selective, shape-tolerant 3D filter. Keep the released wavelet path as the default so the registry refactor cannot change baseline behavior.

## Exit Criteria

- [ ] Hydra passes `model_cfg.encoder_kwargs.gcalf_cfg` into `Encoder`.
- [ ] Baseline construction and a fixed-seed output regression test are unchanged.
- [ ] LFF is identity at initialization, supports variable spatial shapes, and has nonzero gradients.
- [ ] The learned grid is oriented in **centered** frequency coordinates (a centre-weighted grid low-passes).
- [ ] `lff_only` completes tiny-task train, resume, predict, and evaluation.
- [ ] Peak memory is recorded for each configured LFF stage.

## 3.1 Configuration Path

`RetinaUNetModule._build_encoder` already expands `model_cfg["encoder_kwargs"]` into the encoder constructor. Add only one argument to `Encoder`:

```python
def __init__(self, ..., fft_low_ratio=0.3, gcalf_cfg=None):
    super().__init__()
    self.gcalf_cfg = dict(gcalf_cfg or {})
    self.freq_stages = self.gcalf_cfg.get("freq_stages", [1, 3, 4])
    self.fusion_stages = self.gcalf_cfg.get("fusion_stages", [2, 5])
```

Hydra override shape:

```yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: lff
      fusion_type: channel_light
      freq_stages: [1, 3, 4]
      fusion_stages: [2, 5]
      lff:
        grid_size: [4, 8, 8]
        groups: 8
```

Do not add `use_lff` aliases or write these values into `plan["architecture"]`. The model config is serialized with nnDetection checkpoints and is already restored by `nndet/inference/loading.py`.

## 3.2 Registry

Create `GCALF-Net/nndet/arch/encoder/gcalf/registry.py`:

```python
from torch import nn

from nndet.arch.encoder.WaveletFusion import WaveletSpatialFusion
from nndet.arch.encoder.channel_lightweight_fusion import ChannelWiseLightFusion
from .lff import LearnableFrequencyFilter3D


def build_frequency_module(kind, in_channels, out_channels, options=None):
    options = options or {}
    if kind == "wavelet":
        return WaveletSpatialFusion(in_channels, out_channels, wavelet="haar")
    if kind == "lff":
        return LearnableFrequencyFilter3D(in_channels, out_channels, **options)
    if kind == "none":
        return nn.Identity()
    raise ValueError("Unknown frequency_filter_type: {}".format(kind))
```

Construct one module per encoder stage as today. Use `nn.Identity()` outside `freq_stages`. Keep `Encoder.forward` generic and remove the concrete wavelet type check; invoking an identity is harmless and simpler.

Replace the hard-coded stage checks in `modular.py` rather than adding a second code path:

```python
frequency_kind = self.gcalf_cfg.get("frequency_filter_type", "wavelet")
frequency_options = self.gcalf_cfg.get("lff", {}) if frequency_kind == "lff" else {}

# During stage construction
self.frequency_modules.append(
    build_frequency_module(frequency_kind, in_ch, in_ch, frequency_options)
    if stage_id in self.freq_stages else nn.Identity()
)

# During forward
cnn_x = self.frequency_modules[stage_id](cnn_x)
if self.use_transformer and stage_id in self.fusion_stages:
    ...
```

The baseline defaults must resolve to `[1,3,4]`, `[2,5]`, `wavelet`, and `channel_light`, matching the released code. Rename `wavelet_fusion_modules` to `frequency_modules` only if checkpoint key migration is not required; otherwise keep the old attribute name and document why.

## 3.3 LFF Design

A complex scalar per channel is frequency-invariant and therefore is not sufficient. A full learned tensor at every stage is too rigid and expensive. Use a grouped low-resolution spectral grid and interpolate it to the actual `rfftn` shape.

Two properties the naive version gets wrong, both cheap to fix and both invisible without the tests in 3.4:

**(a) Frequency-axis orientation.** `rfftn` orders the two full axes (`D`, `H`) as `[0, +f, …, Nyquist, −f, …, −1]`; only the last (rFFT) axis runs monotonically DC → Nyquist. Interpolating a coarse grid straight onto that index space puts DC and the lowest negative frequency at opposite ends with Nyquist in the middle, so a low-pass response cannot be expressed as a smooth blob and the interpolation smooths *across* the Nyquist discontinuity. Define the grid in **centered** coordinates (DC at the middle of the `D`/`H` axes) and `torch.fft.ifftshift` the interpolated result along those two axes before multiplying.

**(b) A complex gain is not identifiable.** `x` is real, so `X` is Hermitian. Multiplying by a non-Hermitian complex `H` and calling `irfftn` applies the Hermitian projection `(H(f) + conj(H(−f)))/2` — half the parameters alias onto the other half and cannot be recovered from the trained model. Use a **real-valued gain**: exactly Hermitian by construction, half the parameters, zero-phase, and still fully frequency-selective. Learning phase is a documented extension, not the default.

For `x` shaped `(B,C,D,H,W)`:

1. Convert to float32 and compute `X = rfftn(x)` over the three spatial dimensions.
2. Interpolate the real grid from `(G,Kd,Kh,Kw)` to `(G,D,H,W')`, then `ifftshift` along `D,H`.
3. Form `H = 1 + delta_H` and repeat each group over `C/G` channels.
4. Compute `y = irfftn(X * H, s=(D,H,W))`.
5. Cast back to the input dtype and apply an optional `1x1x1` projection.

Initial implementation:

```python
import torch
from torch import nn
from torch.nn import functional as F


class LearnableFrequencyFilter3D(nn.Module):
    """Grouped, shape-tolerant, zero-phase learnable frequency filter.

    The gain grid is stored on CENTERED frequency coordinates for the two full
    FFT axes (D, H); the trailing rFFT axis already runs DC -> Nyquist.
    """

    def __init__(self, in_channels, out_channels, grid_size=(4, 8, 8), groups=8):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        self.in_channels = in_channels
        self.groups = groups
        # real gain, zero-init -> exactly identity at startup
        self.delta_weight = nn.Parameter(torch.zeros(1, groups, *grid_size))
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            spectrum = torch.fft.rfftn(
                x.float(), dim=(-3, -2, -1), norm="ortho"
            )
            delta = F.interpolate(
                self.delta_weight,
                size=spectrum.shape[-3:],
                mode="trilinear",
                align_corners=True,
            )
            # centered grid -> rfftn's [0, +f, .., Nyq, -f, .., -1] ordering
            delta = torch.fft.ifftshift(delta, dim=(-3, -2))
            gain = (1.0 + delta).repeat_interleave(
                self.in_channels // self.groups, dim=1
            )
            filtered = torch.fft.irfftn(
                spectrum * gain,
                s=x.shape[-3:],
                dim=(-3, -2, -1),
                norm="ortho",
            )
        return self.proj(filtered.to(input_dtype))
```

Notes on this starting point, to validate rather than assume:

- `grid_size` is now the literal grid resolution `(Kd, Kh, Kw)`. The earlier `kw // 2 + 1` conflated grid resolution with the rFFT half-axis; the interpolation target already handles that.
- `align_corners=True` puts the extreme grid values on DC and Nyquist. For even `D`/`H` the grid centre lands within half a bin of the `ifftshift` DC position — acceptable, but confirm on the radial response plot rather than assuming.
- Complex × real broadcasting, `torch.fft.ifftshift`, and `irfftn` all exist in the pinned PyTorch 1.10 image; verify in the container anyway.
- A real gain means there is no Hermitian-symmetry question left to validate — that was the reason for choosing it.

## 3.4 Correctness Tests

Create `GCALF-Net/tests/gcalf/test_lff.py` with these gates:

- `delta_weight == 0`, equal channels: output is input within `atol=1e-4`.
- Odd and even `(D,H,W)` shapes preserve shape.
- Backpropagation gives finite, nonzero `delta_weight.grad`.
- Deliberately different low/high grid values produce different gains for synthetic low/high frequency inputs.
- **Orientation gate (catches a missing `ifftshift`):** set the grid to 1 at its centre and −1 elsewhere, then feed (i) a constant volume and (ii) a Nyquist checkerboard `(-1)**(d+h+w)`. The constant must survive and the checkerboard must be suppressed. With the wrong axis convention this test inverts — it is the single most valuable test in this file.
- Output is real and finite for random input (a real gain guarantees this; the test documents the guarantee).
- Float16 CUDA input returns float16 finite output while FFT executes in float32.
- Invalid channel/group combinations fail with a useful error.
- Serialization and reload produce the same output.

Do not test an all-ones filter plus a spatial residual: that formulation returns approximately `2x`, not `x`. (The same trap appears in the CAF skeleton — see `PHASE_4_caf.md §4.4`.)

## 3.5 Integration Sequence

1. Add `gcalf_cfg=None` to `Encoder` and build every current baseline module through the registry.
2. Load a baseline checkpoint and compare fixed-input outputs before and after the refactor.
3. Add LFF and unit tests.
4. Run one forward/backward pass at each real stage shape and record peak memory.
5. Overfit two cases, then run `Task9xx_PICAI_TINY` for two epochs.
6. Resume the tiny run from checkpoint and run prediction/evaluation.
7. Launch all five folds only after the tiny gate passes.

## Risks

| Risk | Mitigation |
|---|---|
| FFT activation memory is too high | Reduce patch size or remove stage 1 only as an explicitly reported architecture change. |
| Grid interpolation creates artifacts | Compare radial response plots and synthetic frequency tests; increase grid resolution only after profiling. |
| Frequency axes silently transposed/unshifted | The orientation gate in 3.4 fails loudly; plot the learned response on centered axes before reporting any "learned low-pass" claim in the thesis. |
| AMP or cuFFT instability | Keep FFT and complex multiplication inside disabled autocast and assert finite values. |
| LFF does not beat wavelet | Preserve as a valid ablation result; do not tune on the test fold. |

**Deliverables:** `nndet/arch/encoder/gcalf/lff.py`, frequency registry, Hydra configs, tests, response visualization, tiny-run metrics, and memory report.

**Next:** `PHASE_4_caf.md`.
