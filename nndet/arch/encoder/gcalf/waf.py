"""Wires the existing WindowAttentionFusion into a two-branch fusion module.

WindowAttentionFusion (nndet/arch/encoder/window_attention_fusion.py) is pure
window self-attention over a single tensor: its forward takes one input, not a
CNN feature and a Swin feature. WindowAttentionFusion3D below adds the
alignment/combination step that actually fuses the two branches: both are
projected to a shared channel count with a 1x1 conv, summed into one aligned
feature, and the existing self-attention module (unmodified, not reimplemented)
attends over that. The residual is counted exactly once (concat-free: the
aligned CNN feature is added back after attention), matching the rule CAF's
WindowedCrossAttentionFusion3D uses for its own residual (ARCHITECTURE.md Sec
7), so WAF vs CAF stays a single-variable swap (self-attention vs bidirectional
cross-attention) with otherwise matching parameter shapes.
"""
import torch
import torch.nn as nn

from nndet.arch.encoder.window_attention_fusion import WindowAttentionFusion


class WindowAttentionFusion3D(nn.Module):
    def __init__(self, cnn_channels, transformer_channels, out_channels,
                 window_size=(2, 7, 7), num_heads=4):
        super().__init__()
        self.cnn_align = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        self.swin_align = nn.Conv3d(transformer_channels, out_channels, 1, bias=False)
        self.attn = WindowAttentionFusion(out_channels, window_size, num_heads)

    def forward(self, cnn_feat, swin_feat):
        # This attention is fed by independently-normalized CNN and Swin
        # branches.  Keeping its QK/softmax path in fp32 prevents the first
        # mixed-precision backward pass from overflowing before GradScaler can
        # react; surrounding convolutions still use AMP normally.
        output_dtype = cnn_feat.dtype
        with torch.cuda.amp.autocast(enabled=False):
            cnn = self.cnn_align(cnn_feat.float())
            swin = self.swin_align(swin_feat.float())
            aligned = cnn + swin
            fused = self.attn(aligned) + cnn
        return fused.to(output_dtype)


def build_waf(cnn_channels, transformer_channels, out_channels, window_size=(2, 7, 7), num_heads=4):
    return WindowAttentionFusion3D(cnn_channels, transformer_channels, out_channels,
                                    window_size=window_size, num_heads=num_heads)
