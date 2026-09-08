"""Frequency-bin masks shared by GCALF frequency-separation modules."""
import torch


def radial_frequency_mask(shape, radius, device, dtype=torch.float32):
    """Return a centered, Hermitian-symmetric spherical low-pass mask.

    ``fftfreq`` gives the actual FFT-bin frequencies.  Its shifted DC bin is
    zero for both odd and even dimensions, unlike a ``linspace`` grid.
    """
    axes = [torch.fft.fftshift(torch.fft.fftfreq(size, device=device)) * 2 for size in shape]
    depth, height, width = torch.meshgrid(*axes, indexing="ij")
    radius_grid = torch.sqrt(depth.square() + height.square() + width.square())
    return (radius_grid <= radius).to(dtype=dtype)
