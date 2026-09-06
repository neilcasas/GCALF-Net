"""Frequency-bin masks shared by fixed and learnable GCALF filters."""
import torch


def radial_frequency_mask(shape, radius, device, dtype=torch.float32):
    """Return a centered spherical low-frequency mask.

    ``fftfreq`` defines exact FFT bins. Therefore, the shifted DC bin is zero
    for both odd and even shapes. The mask also has Hermitian symmetry.
    """
    axes = [torch.fft.fftshift(torch.fft.fftfreq(size, device=device)) * 2 for size in shape]
    depth, height, width = torch.meshgrid(*axes, indexing="ij")
    return (torch.sqrt(depth.square() + height.square() + width.square()) <= radius).to(dtype=dtype)
