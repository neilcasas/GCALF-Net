"""M6 L gate: LearnableFrequencyFilter3D (ARCHITECTURE.md Sec 6, PHASE_3_lff.md Sec 3.3)."""
import pytest
import torch

from nndet.arch.encoder.gcalf.fdsf import FrequencyDomainSeparationAndShunting3D
from nndet.arch.encoder.gcalf.frequency_mask import spherical_mask_rfft
from nndet.arch.encoder.gcalf.lff import LearnableFrequencyFilter3D
from nndet.arch.encoder.gcalf.registry import build_frequency_module
from tests.gcalf.test_encoder_gcalf import BASELINE_GCALF_CFG, _build_encoder

SHAPES = [(8, 16, 16), (9, 17, 17), (4, 8, 8), (5, 9, 9)]


@pytest.mark.parametrize("shape", SHAPES)
def test_zero_delta_weight_equals_fdsf_output_not_merely_the_input(shape):
    """The init-equals-baseline gate: assert equality to FDSF, not `x_low ~= x`."""
    torch.manual_seed(0)
    lff = LearnableFrequencyFilter3D(in_channels=3, radius=0.15)
    fdsf = FrequencyDomainSeparationAndShunting3D(radius=0.15)
    x = torch.randn(1, 3, *shape)

    lff_low, lff_high = lff(x)
    fdsf_low, fdsf_high = fdsf(x)

    assert torch.allclose(lff_low, fdsf_low, atol=1e-5)
    assert torch.allclose(lff_high, fdsf_high, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_low_plus_high_reconstructs_input_losslessly_and_preserves_shape(shape):
    torch.manual_seed(0)
    module = LearnableFrequencyFilter3D(in_channels=3)
    x = torch.randn(1, 3, *shape)

    low, high = module(x)

    assert low.shape == x.shape
    assert high.shape == x.shape
    assert torch.allclose(low + high, x, atol=1e-5)


def test_gradients_reach_delta_weight_through_each_output_independently():
    torch.manual_seed(0)
    module = LearnableFrequencyFilter3D(in_channels=1)
    x = torch.randn(1, 1, 8, 16, 16)

    low, high = module(x)
    low.sum().backward(retain_graph=True)
    assert module.delta_weight.grad is not None
    assert torch.isfinite(module.delta_weight.grad).all()
    assert module.delta_weight.grad.abs().sum() > 0

    module.delta_weight.grad = None
    high.sum().backward()
    assert module.delta_weight.grad is not None
    assert torch.isfinite(module.delta_weight.grad).all()
    assert module.delta_weight.grad.abs().sum() > 0


def test_different_delta_weight_produces_different_output_for_synthetic_inputs():
    torch.manual_seed(0)
    module = LearnableFrequencyFilter3D(in_channels=1, grid_size=(4, 8, 8))
    x = torch.randn(1, 1, 16, 32, 32)

    with torch.no_grad():
        module.delta_weight.fill_(0.05)
    low_a, high_a = module(x)

    with torch.no_grad():
        module.delta_weight.fill_(-0.05)
    low_b, high_b = module(x)

    assert not torch.allclose(low_a, low_b)
    assert not torch.allclose(high_a, high_b)


def test_orientation_gate_routes_low_and_high_frequency_energy_to_the_correct_branch():
    """The single most valuable test in this file (PHASE_3_lff.md Sec 3.3): a grid whose
    response is +1 at DC and -1 elsewhere must send a constant volume into `x_low` and a
    Nyquist checkerboard into `x_high`. A missing `ifftshift` on the D,H axes swaps which
    bin gets which gain, and both directional checks below invert."""
    torch.manual_seed(0)
    d, h, w = 16, 64, 64
    module = LearnableFrequencyFilter3D(in_channels=1, radius=0.0, grid_size=(d, h, w // 2 + 1), groups=1)

    dc_mask = spherical_mask_rfft((d, h, w), radius=0.0, device=torch.device("cpu"))
    assert dc_mask.sum() == 1  # exactly one exact-DC bin
    dc_index = tuple(int(i) for i in dc_mask.nonzero(as_tuple=False)[0])
    with torch.no_grad():
        module.delta_weight.fill_(-1.0)
        module.delta_weight[(0, 0) + dc_index] = 0.0  # gain == +1 at DC (base=1 there), -1 elsewhere

    idx_d = torch.arange(d).view(-1, 1, 1)
    idx_h = torch.arange(h).view(1, -1, 1)
    idx_w = torch.arange(w).view(1, 1, -1)
    checkerboard = ((-1.0) ** (idx_d + idx_h + idx_w)).expand(d, h, w).reshape(1, 1, d, h, w).float()
    constant = torch.full((1, 1, d, h, w), 3.0)

    low_c, high_c = module(constant)
    low_x, high_x = module(checkerboard)

    # gain == base + delta isn't unit magnitude away from the special bins it was pinned
    # at (a constant's spectrum is a delta at DC; a checkerboard's is a delta at Nyquist),
    # so assert which branch carries the energy rather than exact unit values.
    assert low_c.abs().sum() > high_c.abs().sum()
    assert high_x.abs().sum() > low_x.abs().sum()
    assert torch.allclose(high_c, torch.zeros_like(constant), atol=1e-3)
    assert torch.allclose(low_c, constant, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for the fp16 autocast path")
def test_fp16_cuda_input_returns_finite_fp16_output_while_fft_runs_in_fp32():
    module = LearnableFrequencyFilter3D(in_channels=3).cuda()
    x = torch.randn(1, 3, 8, 16, 16, device="cuda", dtype=torch.float16)

    low, high = module(x)

    assert low.dtype == torch.float16
    assert high.dtype == torch.float16
    assert torch.isfinite(low).all()
    assert torch.isfinite(high).all()


def test_invalid_group_channel_combination_raises_a_useful_error():
    with pytest.raises(ValueError, match="divisible"):
        LearnableFrequencyFilter3D(in_channels=3, groups=2)


def test_valid_group_channel_combinations_construct_without_error():
    LearnableFrequencyFilter3D(in_channels=3, groups=1)
    LearnableFrequencyFilter3D(in_channels=3, groups=3)


def test_state_dict_round_trip_reproduces_output():
    torch.manual_seed(0)
    module = LearnableFrequencyFilter3D(in_channels=1)
    with torch.no_grad():
        module.delta_weight.normal_()
    x = torch.randn(1, 1, 8, 16, 16)
    expected_low, expected_high = module(x)

    reloaded = LearnableFrequencyFilter3D(in_channels=1)
    reloaded.load_state_dict(module.state_dict())
    low, high = reloaded(x)

    assert torch.equal(low, expected_low)
    assert torch.equal(high, expected_high)


def test_registry_fdsf_branch_is_unchanged_by_the_lff_addition():
    """Fixed-seed regression: adding the lff branch must not move the fdsf branch."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 16, 16)
    direct = FrequencyDomainSeparationAndShunting3D(radius=0.15)
    via_registry = build_frequency_module("fdsf", 3, {"radius": 0.15})

    assert isinstance(via_registry, FrequencyDomainSeparationAndShunting3D)
    direct_low, direct_high = direct(x)
    registry_low, registry_high = via_registry(x)
    assert torch.equal(direct_low, registry_low)
    assert torch.equal(direct_high, registry_high)


def test_registry_lff_branch_builds_the_learnable_filter():
    module = build_frequency_module("lff", 3, {"radius": 0.15, "grid_size": (4, 8, 8), "groups": 1})

    assert isinstance(module, LearnableFrequencyFilter3D)


def test_lff_runs_through_the_five_level_encoder():
    torch.manual_seed(0)
    cfg = dict(
        BASELINE_GCALF_CFG,
        frequency_filter_type="lff",
        lff={"radius": 0.15, "grid_size": (4, 8, 8), "groups": 1},
    )
    encoder = _build_encoder(cfg)

    outputs = encoder(torch.randn(1, 3, 16, 32, 32))

    assert len(outputs) == 5
    assert all(torch.isfinite(output).all() for output in outputs)
