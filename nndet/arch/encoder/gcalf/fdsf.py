"""Fixed input-level 3D frequency-domain separation and shunting (ARCHITECTURE.md Sec 5).

Runs exactly once on the three-channel bpMRI input, before both branches of the
five-level encoder. Splits the input into a low-frequency and a high-frequency
component via a fixed spherical mask in centered FFT space; the split is
lossless (X_low + X_high == X) and the direction (low -> Swin, high -> CNN) is
fixed by the paper's hypotheses, not a config choice.
"""
import torch
import torch.nn as nn

from nndet.arch.encoder.gcalf.frequency_mask import radial_frequency_mask


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
        return radial_frequency_mask(shape, radius, device)
