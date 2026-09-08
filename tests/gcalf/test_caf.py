import torch
import torch.nn as nn

from nndet.arch.encoder.gcalf.caf import (
    WindowedCrossAttentionFusion3D,
    window_partition_3d,
    window_reverse_3d,
)
from nndet.arch.encoder.gcalf.registry import build_fusion_module
from tests.gcalf.test_encoder_gcalf import BASELINE_GCALF_CFG, _build_encoder


def test_window_partition_reverse_is_lossless_for_divisible_and_padded_shapes():
    for shape in ((2, 6, 4, 8, 12), (1, 6, 3, 5, 6)):
        x = torch.randn(shape)
        windows, padding_mask, metadata = window_partition_3d(x, (2, 2, 3))

        assert torch.equal(window_reverse_3d(windows, metadata), x)
        assert padding_mask.dtype == torch.bool
        assert bool(padding_mask.any()) is (shape[2:] != (4, 8, 12))


def test_caf_output_shape_and_gradients_for_padded_windows():
    torch.manual_seed(0)
    module = WindowedCrossAttentionFusion3D(8, 12, 16, window_size=(2, 3, 3), num_heads=4)
    cnn = torch.randn(2, 8, 3, 5, 7, requires_grad=True)
    swin = torch.randn(2, 12, 3, 5, 7, requires_grad=True)

    output = module(cnn, swin)
    output.square().mean().backward()

    assert output.shape == (2, 16, 3, 5, 7)
    for parameter in (
        module.cnn_align.weight,
        module.swin_align.weight,
        module.cnn_queries_swin.in_proj_weight,
        module.swin_queries_cnn.in_proj_weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_caf_masks_padding_and_counts_the_cnn_residual_once():
    torch.manual_seed(0)
    module = WindowedCrossAttentionFusion3D(4, 4, 4, window_size=(2, 3, 3), num_heads=2).eval()
    module.out_proj = nn.Identity()
    with torch.no_grad():
        module.cnn_queries_swin.out_proj.weight.zero_()
        module.cnn_queries_swin.out_proj.bias.zero_()
        module.swin_queries_cnn.out_proj.weight.zero_()
        module.swin_queries_cnn.out_proj.bias.zero_()
    cnn = torch.randn(1, 4, 3, 4, 5)
    swin = torch.randn(1, 4, 3, 4, 5)

    output = module(cnn, swin)
    expected = module.cnn_align(cnn)

    assert torch.equal(output, expected)
    assert torch.isfinite(output).all()


def test_caf_registry_branch_builds_the_cross_attention_module():
    module = build_fusion_module("caf", 8, 12, 16, {"window_size": (2, 2, 2), "num_heads": 4})

    assert isinstance(module, WindowedCrossAttentionFusion3D)


def test_caf_runs_through_all_frozen_encoder_fusion_levels():
    cfg = dict(
        BASELINE_GCALF_CFG,
        fusion_type="caf",
        caf={"window_size": (2, 2, 2), "num_heads": 4, "dropout": 0.0},
    )
    encoder = _build_encoder(cfg)
    outputs = encoder(torch.randn(1, 3, 16, 32, 32))

    assert len(outputs) == 5
    assert all(torch.isfinite(output).all() for output in outputs)
