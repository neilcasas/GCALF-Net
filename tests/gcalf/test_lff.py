import torch

from nndet.arch.encoder.gcalf.fdsf import FrequencyDomainSeparationAndShunting3D
from nndet.arch.encoder.gcalf.lff import LearnableFrequencyFilter3D


def test_zero_delta_weight_matches_fdsf_exactly():
    x = torch.randn(1, 3, 8, 16, 16)
    lff = LearnableFrequencyFilter3D(radius=0.15, delta_weight=0.0)
    fdsf = FrequencyDomainSeparationAndShunting3D(radius=0.15)
    assert all(torch.allclose(a, b) for a, b in zip(lff(x), fdsf(x)))


def test_lff_reconstructs_odd_and_even_shapes():
    module = LearnableFrequencyFilter3D()
    for shape in ((7, 15, 15), (8, 16, 16)):
        x = torch.randn(1, 3, *shape)
        low, high = module(x)
        assert low.shape == x.shape
        assert torch.allclose(low + high, x, atol=1e-4)


def test_low_and_high_losses_produce_finite_filter_gradients():
    for output_index in (0, 1):
        module = LearnableFrequencyFilter3D()
        x = torch.randn(1, 1, 8, 16, 16)
        module(x)[output_index].square().mean().backward()
        gradient = module.filter_offset.grad
        assert gradient is not None
        assert torch.isfinite(gradient)
        assert gradient.abs() > 0


def test_orientation_separates_constant_and_nyquist_inputs():
    module = LearnableFrequencyFilter3D(delta_weight=0.0)
    constant = torch.ones(1, 1, 8, 16, 16)
    indices = sum(torch.arange(size).reshape((1,) * axis + (size,) + (1,) * (2 - axis)) for axis, size in enumerate((8, 16, 16)))
    nyquist = ((-1.0) ** indices).reshape(1, 1, 8, 16, 16)
    constant_low, constant_high = module(constant)
    nyquist_low, nyquist_high = module(nyquist)
    assert torch.allclose(constant_low, constant, atol=1e-4)
    assert torch.allclose(constant_high, torch.zeros_like(constant), atol=1e-4)
    assert torch.allclose(nyquist_low, torch.zeros_like(nyquist), atol=1e-4)
    assert torch.allclose(nyquist_high, nyquist, atol=1e-4)


def test_float16_input_uses_float32_fft_and_serialization_preserves_output():
    module = LearnableFrequencyFilter3D().eval()
    x = torch.randn(1, 1, 8, 16, 16, dtype=torch.float16)
    output = module(x)
    restored = LearnableFrequencyFilter3D().eval()
    restored.load_state_dict(module.state_dict())
    assert all(part.dtype == torch.float16 for part in output)
    assert all(torch.allclose(a, b) for a, b in zip(output, restored(x)))
