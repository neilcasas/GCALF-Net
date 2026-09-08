"""Bidirectional, masked windowed cross-attention fusion for GCALF-Net."""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WindowMetadata:
    """Information needed to reverse a padded 3D window partition."""

    batch_size: int
    spatial_size: Tuple[int, int, int]
    padded_spatial_size: Tuple[int, int, int]
    window_size: Tuple[int, int, int]


def _validate_window_size(window_size):
    window_size = tuple(int(value) for value in window_size)
    if len(window_size) != 3 or any(value <= 0 for value in window_size):
        raise ValueError("window_size must contain three positive integers")
    return window_size


def window_partition_3d(x, window_size):
    """Partition ``(B, C, D, H, W)`` features and mask right-side padding."""
    if x.ndim != 5:
        raise ValueError(f"Expected a 5D feature tensor, found shape {tuple(x.shape)}")
    window_size = _validate_window_size(window_size)
    batch_size, channels, depth, height, width = x.shape
    if min(depth, height, width) <= 0:
        raise ValueError("Feature-map spatial dimensions must be positive")
    pad = tuple((size - dimension % size) % size for dimension, size in zip((depth, height, width), window_size))
    padded = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
    padded_size = tuple(dimension + amount for dimension, amount in zip((depth, height, width), pad))
    depth_windows, height_windows, width_windows = tuple(
        dimension // size for dimension, size in zip(padded_size, window_size)
    )
    token_count = window_size[0] * window_size[1] * window_size[2]
    windows = padded.permute(0, 2, 3, 4, 1).contiguous().view(
        batch_size, depth_windows, window_size[0], height_windows, window_size[1],
        width_windows, window_size[2], channels,
    )
    windows = windows.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, token_count, channels)

    valid = torch.ones((batch_size, 1, depth, height, width), dtype=torch.bool, device=x.device)
    valid = F.pad(valid, (0, pad[2], 0, pad[1], 0, pad[0]), value=False)
    valid = valid.permute(0, 2, 3, 4, 1).contiguous().view(
        batch_size, depth_windows, window_size[0], height_windows, window_size[1], width_windows, window_size[2], 1
    )
    valid = valid.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, token_count)
    metadata = WindowMetadata(batch_size, (depth, height, width), padded_size, window_size)
    return windows, ~valid, metadata


def window_reverse_3d(windows, metadata):
    """Reverse :func:`window_partition_3d` and crop only recorded padding."""
    if windows.ndim != 3:
        raise ValueError(f"Expected window tokens shaped (N, T, C), found {tuple(windows.shape)}")
    depth, height, width = metadata.spatial_size
    padded_depth, padded_height, padded_width = metadata.padded_spatial_size
    wd, wh, ww = metadata.window_size
    expected_tokens = wd * wh * ww
    expected_windows = metadata.batch_size * (padded_depth // wd) * (padded_height // wh) * (padded_width // ww)
    if windows.shape[0] != expected_windows or windows.shape[1] != expected_tokens:
        raise ValueError("Window tokens do not match partition metadata")
    channels = windows.shape[-1]
    x = windows.view(
        metadata.batch_size, padded_depth // wd, padded_height // wh, padded_width // ww, wd, wh, ww, channels
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(
        metadata.batch_size, padded_depth, padded_height, padded_width, channels
    )
    return x[:, :depth, :height, :width].permute(0, 4, 1, 2, 3).contiguous()


class WindowedCrossAttentionFusion3D(nn.Module):
    """Fuse aligned CNN and Swin features through two directed cross-attention paths."""

    def __init__(self, cnn_channels, transformer_channels, out_channels,
                 window_size=(2, 7, 7), num_heads=4, dropout=0.0):
        super().__init__()
        if out_channels <= 0 or num_heads <= 0 or out_channels % num_heads:
            raise ValueError("out_channels must be positive and divisible by num_heads")
        self.window_size = _validate_window_size(window_size)
        self.cnn_align = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        self.swin_align = nn.Conv3d(transformer_channels, out_channels, 1, bias=False)
        self.cnn_queries_swin = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout, batch_first=True)
        self.swin_queries_cnn = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(out_channels)
        self.out_proj = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat, swin_feat):
        if cnn_feat.shape[0] != swin_feat.shape[0] or cnn_feat.shape[2:] != swin_feat.shape[2:]:
            raise ValueError("CNN and Swin features must have matching batch and spatial dimensions")
        output_dtype = cnn_feat.dtype
        with torch.cuda.amp.autocast(enabled=False):
            cnn = self.cnn_align(cnn_feat.float())
            swin = self.swin_align(swin_feat.float())
            cnn_windows, padding_mask, metadata = window_partition_3d(cnn, self.window_size)
            swin_windows, swin_padding_mask, _ = window_partition_3d(swin, self.window_size)
            if not torch.equal(padding_mask, swin_padding_mask):
                raise RuntimeError("Aligned features produced different window padding masks")
            cnn_cross, _ = self.cnn_queries_swin(
                cnn_windows, swin_windows, swin_windows, key_padding_mask=padding_mask, need_weights=False
            )
            swin_cross, _ = self.swin_queries_cnn(
                swin_windows, cnn_windows, cnn_windows, key_padding_mask=padding_mask, need_weights=False
            )
            fused_windows = self.norm(cnn_cross + swin_cross)
            fused = window_reverse_3d(fused_windows, metadata)
            # The aligned CNN feature has one residual path.  It is not also
            # concatenated or added inside the attention expression.
            output = self.out_proj(fused) + cnn
        return output.to(output_dtype)


def build_caf(cnn_channels, transformer_channels, out_channels, **options):
    return WindowedCrossAttentionFusion3D(cnn_channels, transformer_channels, out_channels, **options)
