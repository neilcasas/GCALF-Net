import torch

from nndet.arch.encoder.gcalf.waf import WindowAttentionFusion3D, build_waf
from nndet.arch.encoder.window_attention_fusion import WindowAttentionFusion


def test_build_waf_returns_a_window_attention_fusion3d_wrapping_the_existing_module():
    module = build_waf(cnn_channels=8, transformer_channels=12, out_channels=16,
                        window_size=(2, 2, 2), num_heads=4)
    assert isinstance(module, WindowAttentionFusion3D)
    assert isinstance(module.attn, WindowAttentionFusion)


def test_output_shape_matches_cnn_spatial_dims():
    torch.manual_seed(0)
    module = build_waf(cnn_channels=8, transformer_channels=12, out_channels=16,
                        window_size=(2, 2, 2), num_heads=4)
    cnn_feat = torch.randn(1, 8, 4, 8, 8)
    swin_feat = torch.randn(1, 12, 4, 8, 8)

    out = module(cnn_feat, swin_feat)

    assert out.shape == (1, 16, 4, 8, 8)


def test_gradients_reach_both_input_projections():
    torch.manual_seed(0)
    module = build_waf(cnn_channels=8, transformer_channels=12, out_channels=16,
                        window_size=(2, 2, 2), num_heads=4)
    cnn_feat = torch.randn(1, 8, 4, 8, 8, requires_grad=True)
    swin_feat = torch.randn(1, 12, 4, 8, 8, requires_grad=True)

    module(cnn_feat, swin_feat).sum().backward()

    assert module.cnn_align.weight.grad is not None
    assert module.cnn_align.weight.grad.abs().sum() > 0
    assert module.swin_align.weight.grad is not None
    assert module.swin_align.weight.grad.abs().sum() > 0
    assert cnn_feat.grad is not None and cnn_feat.grad.abs().sum() > 0
    assert swin_feat.grad is not None and swin_feat.grad.abs().sum() > 0


def test_wrapper_does_not_change_window_attention_fusions_own_numerics():
    """Regression guard: the wrapper's forward must stay exactly
    align -> add -> existing self-attention -> +cnn residual, so that
    swapping WAF for CAF later is the only thing that changes."""
    torch.manual_seed(0)
    module = WindowAttentionFusion3D(cnn_channels=6, transformer_channels=6, out_channels=8,
                                      window_size=(2, 2, 2), num_heads=4)
    cnn_feat = torch.randn(1, 6, 4, 6, 6)
    swin_feat = torch.randn(1, 6, 4, 6, 6)

    out = module(cnn_feat, swin_feat)

    with torch.no_grad():
        cnn = module.cnn_align(cnn_feat)
        swin = module.swin_align(swin_feat)
        expected = module.attn(cnn + swin) + cnn

    assert torch.allclose(out, expected)
