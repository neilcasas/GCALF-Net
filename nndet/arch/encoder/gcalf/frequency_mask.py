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


def spherical_mask_rfft(shape, radius, device, dtype=torch.float32):
    """Return the ``radial_frequency_mask`` equivalent for an ``rfftn`` spectrum.

    ``shape`` is the **spatial** shape ``(D, H, W)``, not the half-spectrum shape
    ``rfftn`` produces -- the trailing axis needs ``rfftfreq(W)`` to derive its own
    length ``W // 2 + 1``, so passing the already-halved spectrum shape both gets
    that length wrong and is not invertible back to ``W``. The two full axes
    (D, H) use the same centered ``fftfreq`` convention as ``radial_frequency_mask``;
    the trailing axis uses ``rfftfreq``, which already runs DC -> Nyquist and is
    left unshifted.
    """
    depth_size, height_size, width_size = shape
    depth_axis = torch.fft.fftshift(torch.fft.fftfreq(depth_size, device=device)) * 2
    height_axis = torch.fft.fftshift(torch.fft.fftfreq(height_size, device=device)) * 2
    width_axis = torch.fft.rfftfreq(width_size, device=device) * 2
    depth, height, width = torch.meshgrid(depth_axis, height_axis, width_axis, indexing="ij")
    radius_grid = torch.sqrt(depth.square() + height.square() + width.square())
    return (radius_grid <= radius).to(dtype=dtype)
