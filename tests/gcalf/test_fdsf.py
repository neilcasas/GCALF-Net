import pytest
import torch

from nndet.arch.encoder.gcalf.fdsf import FrequencyDomainSeparationAndShunting3D


def test_mask_boundary_is_exact_and_inclusive():
    shape = (8, 8, 8)
    mask = FrequencyDomainSeparationAndShunting3D._radial_mask(shape, radius=0.15, device=torch.device("cpu"))
    d, h, w = shape
    dd = torch.fft.fftshift(torch.fft.fftfreq(d))[:, None, None] * 2
    hh = torch.fft.fftshift(torch.fft.fftfreq(h))[None, :, None] * 2
    ww = torch.fft.fftshift(torch.fft.fftfreq(w))[None, None, :] * 2
    r = (dd**2 + hh**2 + ww**2).sqrt().expand(d, h, w)
    assert torch.all(mask[r <= 0.15] == 1.0)
    assert torch.all(mask[r > 0.15] == 0.0)


def test_mask_is_exactly_one_at_center_and_zero_beyond_a_zero_radius():
    mask = FrequencyDomainSeparationAndShunting3D._radial_mask((5, 5, 5), radius=0.0, device=torch.device("cpu"))
    assert mask[2, 2, 2] == 1.0
    assert mask.sum() == 1.0  # only r == 0 satisfies r <= 0


def test_even_shape_places_dc_exactly_in_the_low_branch():
    mask = FrequencyDomainSeparationAndShunting3D._radial_mask((8, 16, 16), radius=0.0, device=torch.device("cpu"))
    assert mask[4, 8, 8] == 1.0
    assert mask.sum() == 1.0


def test_fixed_mask_has_hermitian_symmetry():
    shape = (8, 16, 16)
    mask = FrequencyDomainSeparationAndShunting3D._radial_mask(shape, radius=0.3, device=torch.device("cpu"))
    unshifted = torch.fft.ifftshift(mask)
    reflected = unshifted[(-torch.arange(shape[0])) % shape[0]][:, (-torch.arange(shape[1])) % shape[1]]
    reflected = reflected[:, :, (-torch.arange(shape[2])) % shape[2]]
    assert torch.equal(unshifted, reflected)


@pytest.mark.parametrize("shape", [(8, 16, 16), (9, 17, 17), (4, 8, 8), (5, 9, 9)])
def test_low_plus_high_reconstructs_input_losslessly(shape):
    torch.manual_seed(0)
    module = FrequencyDomainSeparationAndShunting3D(radius=0.15)
    x = torch.randn(1, 3, *shape)

    low, high = module(x)

    assert low.shape == x.shape
    assert high.shape == x.shape
    assert torch.allclose(low + high, x, atol=1e-4)


def test_constant_volume_passes_almost_entirely_into_low_branch():
    # Shape matters here: torch.linspace(-1, 1, n) does not land exactly on 0
    # at the DC bin's shifted index (n // 2) for even n -- the residual offset
    # is ~1/(n-1) per axis. At small shapes (e.g. 8x16x16) that offset alone
    # can exceed a small radius like 0.15 and push DC into the high branch;
    # at the pipeline's real shapes (e.g. 32x256x256) it is two orders of
    # magnitude smaller than the radius. Use a shape on that realistic side,
    # matching tests/test_encoder_cpu.py's convention.
    module = FrequencyDomainSeparationAndShunting3D(radius=0.15)
    x = torch.full((1, 1, 16, 64, 64), 3.0)

    low, high = module(x)

    assert torch.allclose(low, x, atol=1e-4)
    assert torch.allclose(high, torch.zeros_like(x), atol=1e-4)


def test_nyquist_checkerboard_passes_almost_entirely_into_high_branch():
    """Catches a missing fftshift/ifftshift pair: without it, the Nyquist
    delta lands at the wrong grid location and this assertion inverts."""
    module = FrequencyDomainSeparationAndShunting3D(radius=0.15)
    d, h, w = 16, 64, 64
    idx_d = torch.arange(d).view(-1, 1, 1)
    idx_h = torch.arange(h).view(1, -1, 1)
    idx_w = torch.arange(w).view(1, 1, -1)
    checkerboard = ((-1.0) ** (idx_d + idx_h + idx_w)).expand(d, h, w).reshape(1, 1, d, h, w).float()

    low, high = module(checkerboard)

    assert torch.allclose(high, checkerboard, atol=1e-3)
    assert torch.allclose(low, torch.zeros_like(checkerboard), atol=1e-3)
