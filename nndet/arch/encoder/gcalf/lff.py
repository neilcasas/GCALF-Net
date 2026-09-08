"""Learnable Frequency Filter, a learned drop-in replacement for FDSF (ARCHITECTURE.md Sec 6).

Same input-level location and same signature as
``FrequencyDomainSeparationAndShunting3D``: runs once on the three-channel bpMRI input, before
both branches of the five-level encoder, and emits the same complementary ``(x_low, x_high)``
pair. The gain is parameterized as ``M_fixed + delta_H``, where ``M_fixed`` is FDSF's fixed
spherical mask and ``delta_H`` is a zero-initialized learned grid, so at ``delta_weight == 0``
LFF is exactly FDSF -- the baseline cannot move when the registry gains this branch (M6).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nndet.arch.encoder.gcalf.frequency_mask import spherical_mask_rfft


class LearnableFrequencyFilter3D(nn.Module):
    """Grouped, shape-tolerant, zero-phase, real-valued learnable frequency separation.

    Same (x_low, x_high) arity as FDSF, and the SAME MODULE at delta_weight=0: the learned
    response is parameterized as FDSF's spherical mask plus a zero-initialized delta, so
    ``lff_only`` starts from the frozen baseline.

    The gain grid is stored on CENTERED frequency coordinates for the two full FFT axes
    (D, H); the trailing rFFT axis already runs DC -> Nyquist.
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
                                       device=x.device)          # FDSF's M, centered on D,H
            delta = F.interpolate(self.delta_weight, size=spectrum.shape[-3:],
                                  mode="trilinear", align_corners=True)
            gain = (base + delta).repeat_interleave(self.in_channels // self.groups, dim=1)
            gain = torch.fft.ifftshift(gain, dim=(-3, -2))       # centered grid -> rfftn's ordering
            x_low = torch.fft.irfftn(spectrum * gain, s=x.shape[-3:],
                                     dim=(-3, -2, -1), norm="ortho")
        x_low = x_low.to(input_dtype)
        return x_low, x - x_low                                  # complementary by construction
