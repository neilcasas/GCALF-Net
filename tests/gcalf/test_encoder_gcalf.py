"""M2.6 L gate: FDSF+WAF wired through the fixed five-level encoder (PHASE_2_baseline.md Sec 2.6).

Forward `torch.randn(1,3,32,256,256)` -- a deliberate worst-case smoke shape, not the training
shape -- through the encoder with FDSF+WAF; assert exactly five encoder tensors reach the decoder
with no shape error, and that the level/fusion-level contract is enforced at construction time
(ARCHITECTURE.md Sec 4: "Configuration validation rejects any other level count or out-of-range
fusion level").
"""
import pytest
import torch

from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.conv import ConvInstanceRelu, Generator
from nndet.arch.encoder.modular import Encoder

BASELINE_GCALF_CFG = {
    "frequency_filter_type": "fdsf",
    "fusion_type": "waf",
    "num_levels": 5,
    "fusion_levels": [0, 1, 2, 3, 4],
    "fdsf": {"radius": 0.15},
    "waf": {"window_size": (2, 2, 2), "num_heads": 4},
}


def _build_encoder(gcalf_cfg, conv_kernels=None, strides=None):
    return Encoder(
        conv=Generator(ConvInstanceRelu, 3),
        conv_kernels=conv_kernels or [(3, 3, 3)] * 5,
        strides=strides or [(2, 2, 2)] * 4,
        block_cls=StackedConvBlock2,
        in_channels=3,
        start_channels=4,
        max_channels=64,
        gcalf_cfg=gcalf_cfg,
    )


def test_five_level_smoke_shape_reaches_decoder_with_no_shape_error():
    torch.manual_seed(0)
    encoder = _build_encoder(BASELINE_GCALF_CFG).eval()

    with torch.no_grad():
        outputs = encoder(torch.randn(1, 3, 32, 256, 256))

    assert len(outputs) == 5
    assert all(torch.isfinite(output).all() for output in outputs)
    assert encoder.get_channels() == [output.shape[1] for output in outputs]


def test_gradients_flow_end_to_end_through_frequency_module_and_fusion_modules():
    torch.manual_seed(0)
    encoder = _build_encoder(BASELINE_GCALF_CFG)
    x = torch.randn(1, 3, 16, 32, 32, requires_grad=True)

    outputs = encoder(x)
    sum(output.sum() for output in outputs).backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    for fusion in encoder.fusion_modules.values():
        assert fusion.cnn_align.weight.grad is not None
        assert fusion.cnn_align.weight.grad.abs().sum() > 0


def test_wrong_level_count_is_rejected_at_construction():
    with pytest.raises(ValueError, match="requires exactly"):
        _build_encoder(BASELINE_GCALF_CFG, conv_kernels=[(3, 3, 3)] * 6, strides=[(2, 2, 2)] * 5)


def test_out_of_range_fusion_level_is_rejected_at_construction():
    bad_cfg = dict(BASELINE_GCALF_CFG, fusion_levels=[0, 5])
    with pytest.raises(ValueError, match="out of range"):
        _build_encoder(bad_cfg)


def test_fusion_modules_are_constructed_only_at_the_declared_levels():
    """Dead-parameter guard (PHASE_2_baseline.md Sec 2.1): a frozen subset smaller than
    all five levels must not instantiate fusion modules at the dropped levels."""
    torch.manual_seed(0)
    cfg = dict(BASELINE_GCALF_CFG, fusion_levels=[0, 2])
    encoder = _build_encoder(cfg).eval()

    assert set(encoder.fusion_modules.keys()) == {"0", "2"}

    with torch.no_grad():
        outputs = encoder(torch.randn(1, 3, 16, 32, 32))
    assert len(outputs) == 5


def test_no_gcalf_cfg_falls_back_to_the_plain_conv_encoder():
    """Legacy path: without gcalf_cfg the encoder builds no Swin branch, no frequency
    module, and no fusion -- this is what tests/test_encoder_cpu.py already relies on."""
    encoder = _build_encoder(gcalf_cfg=None, conv_kernels=[(3, 3, 3)] * 6, strides=[(2, 2, 2)] * 5)
    assert not hasattr(encoder, "frequency_module")
    assert not hasattr(encoder, "fusion_modules")
    assert not hasattr(encoder, "transformer")
