"""Learnable, complementary 3D frequency separation for GCALF-Net."""
import torch
import torch.nn as nn

from nndet.arch.encoder.gcalf.frequency_mask import radial_frequency_mask


class LearnableFrequencyFilter3D(nn.Module):
    """Refine the fixed radial mask while preserving low-plus-high reconstruction.

    The one-bin learnable offset keeps this module resolution independent. A
    zero ``delta_weight`` gives the exact FDSF split for every input shape.
    """

    def __init__(self, radius=0.15, delta_weight=0.1):
        super().__init__()
        self.radius = radius
        self.delta_weight = nn.Parameter(torch.tensor(float(delta_weight)))
        self.filter_offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            spectrum = torch.fft.fftshift(torch.fft.fftn(x.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
            fixed_mask = radial_frequency_mask(spectrum.shape[-3:], self.radius, spectrum.device)
            mask = (fixed_mask + self.delta_weight * torch.tanh(self.filter_offset)).clamp(0, 1)
            low_spectrum = spectrum * mask
            high_spectrum = spectrum * (1 - mask)
            low = torch.fft.ifftn(torch.fft.ifftshift(low_spectrum, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
            high = torch.fft.ifftn(torch.fft.ifftshift(high_spectrum, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
        return low.to(input_dtype), high.to(input_dtype)
