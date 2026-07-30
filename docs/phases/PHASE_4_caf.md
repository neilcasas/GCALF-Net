# Phase 4 - Bidirectional Windowed Cross-Attention Fusion

**Milestone:** M6 | **Depends on:** Phase 3 registry | **Blocks:** full GCALF-Net

## Goal

Replace `MemoryEfficientFusion`/`ChannelWiseLightFusion` at encoder stages `[2, 5]` with true Q/K/V cross-attention between aligned CNN and Swin features. "CAF" in the thesis always means this module. TransFuse BiFusion is a non-cross-attention fallback and must be named separately.

## Exit Criteria

- [ ] CNN queries Swin keys/values and Swin queries CNN keys/values.
- [ ] Window partition/reverse is lossless for divisible and padded shapes.
- [ ] Padded window positions are masked out of attention (`key_padding_mask`), not attended as real keys.
- [ ] The aligned CNN feature reaches the output through exactly one residual path.
- [ ] Both branch projections receive gradients.
- [ ] Representative stage-2 and stage-5 tensors pass the declared GPU memory gate.
- [ ] `caf_only` completes tiny-task train, resume, predict, and evaluation.

## 4.1 Integration Contract

The released `Encoder.forward` already resizes each Swin feature to the CNN spatial shape before fusion. Preserve this interface:

```python
WindowedCrossAttentionFusion3D(
    cnn_channels,
    transformer_channels,
    out_channels,
    window_size=(2, 7, 7),
    num_heads=4,
    dropout=0.0,
)

fused = module(cnn_feat, resized_swin_feat)
```

Add this branch to `build_fusion_module` in the registry:

```python
def build_fusion_module(kind, cnn_channels, transformer_channels, out_channels, options=None):
    options = options or {}
    if kind == "channel_light":
        return ChannelWiseLightFusion(cnn_channels, transformer_channels, out_channels)
    if kind == "windowed_cross_attention":
        return WindowedCrossAttentionFusion3D(
            cnn_channels, transformer_channels, out_channels, **options
        )
    raise ValueError("Unknown fusion_type: {}".format(kind))
```

## 4.2 Why Windowed Attention

Global spatial cross-attention over `N=D*H*W` tokens stores an `N x N` matrix per head and is not viable at stage 2. Partitioning corresponding aligned features into windows bounds attention to `Nw=wd*wh*ww` tokens. Complexity becomes `O(number_of_windows * Nw^2)`.

The selected direction is bidirectional:

```text
CNN output  = Attention(Q=cnn,  K=swin, V=swin)
Swin output = Attention(Q=swin, K=cnn,  V=cnn)
Fusion      = projection([cnn residual, CNN output, Swin output])
```

This is materially different from independent SE/spatial gates or a Hadamard product.

## 4.3 Window Helpers

Implement and test helpers before the attention module:

```python
def window_partition_3d(x, window_size):
    # x: (B,C,D,H,W); pad D/H/W to exact window multiples.
    # return: (B*n_windows, wd*wh*ww, C), key_padding_mask, metadata
    #   key_padding_mask: (B*n_windows, wd*wh*ww) bool, True where the token
    #   is padding and must be excluded from attention.
    ...


def window_reverse_3d(windows, metadata):
    # Restore (B,C,D,H,W), then crop only the recorded right-side padding.
    ...
```

Metadata must contain batch size, original spatial size, padded spatial size, and window size. Keep ordering explicit and test `reverse(partition(x)) == x` exactly.

**The mask is not optional.** With `window_size=(2,7,7)` and feature maps whose spatial dims are rarely multiples of 7, padding can be a large fraction of the border windows. Unmasked, those zero tokens are valid keys: attention spends probability mass on them and the model learns a padding-shaped bias that shifts with input size. Both `MultiheadAttention` calls take `key_padding_mask`. A window that is *entirely* padding produces NaN under softmax masking — drop those windows, or keep one unmasked position and discard the result during reverse; whichever you choose, assert no NaN in the test suite.

## 4.4 Initial Module Skeleton

```python
class WindowedCrossAttentionFusion3D(nn.Module):
    def __init__(self, cnn_channels, transformer_channels, out_channels,
                 window_size=(2, 7, 7), num_heads=4, dropout=0.0):
        super().__init__()
        if out_channels % num_heads != 0:
            raise ValueError("out_channels must be divisible by num_heads")
        self.window_size = tuple(window_size)
        self.cnn_align = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        self.swin_align = nn.Conv3d(transformer_channels, out_channels, 1, bias=False)
        self.cnn_queries_swin = nn.MultiheadAttention(
            out_channels, num_heads, dropout=dropout, batch_first=True
        )
        self.swin_queries_cnn = nn.MultiheadAttention(
            out_channels, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(out_channels)
        self.out_proj = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat, swin_feat):
        cnn = self.cnn_align(cnn_feat)
        swin = self.swin_align(swin_feat)
        cnn_w, pad_mask, metadata = window_partition_3d(cnn, self.window_size)
        swin_w, _, _ = window_partition_3d(swin, self.window_size)
        cnn_cross, _ = self.cnn_queries_swin(
            cnn_w, swin_w, swin_w,
            key_padding_mask=pad_mask, need_weights=False,
        )
        swin_cross, _ = self.swin_queries_cnn(
            swin_w, cnn_w, cnn_w,
            key_padding_mask=pad_mask, need_weights=False,
        )
        # attention terms only: `cnn` re-enters once, as the residual below
        fused_w = self.norm(cnn_cross + swin_cross)
        fused = window_reverse_3d(fused_w, metadata)
        return self.out_proj(torch.cat([cnn, fused], dim=1)) + cnn
```

**Count the CNN path exactly once.** The earlier draft of this skeleton had `fused_w = norm(cnn_w + cnn_cross + swin_cross)` *and* the concatenation *and* the trailing `+ cnn`, so the aligned CNN feature entered the output three times — the same defect as the "all-ones filter plus a spatial residual returns ≈2x" trap called out in `PHASE_3_lff.md §3.4`. Keep the attention outputs in `fused_w` and let `cnn` reach the output through the concat and the single residual. If you prefer a pre-norm residual inside the window, remove the trailing `+ cnn` instead — one path, not two.

Confirm that `batch_first=True` and `need_weights=False` behave as expected in the pinned PyTorch 1.10 image. If `batch_first` is unavailable in the exact patch version, transpose to `(tokens,batch,channels)` explicitly rather than modernizing the runtime.

**Known simplification:** there is no relative position bias and no shifted-window pass, so attention is permutation-invariant inside a window and carries no information across window borders. That is a deliberate cost/benefit choice against Swin's SW-MSA; record it as an architecture limitation in the thesis, and treat a learnable relative position bias as the first upgrade if CAF underperforms for reasons other than memory.

## 4.5 Configuration

```yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: wavelet
      fusion_type: windowed_cross_attention
      freq_stages: [1, 3, 4]
      fusion_stages: [2, 5]
      caf:
        window_size: [2, 7, 7]
        num_heads: 4
        dropout: 0.0
```

The registry receives the stage channel count and may derive `num_heads` per stage when one constant is invalid. Store the resolved per-stage values in the experiment snapshot.

## 4.6 Tests and Profiling

Create `GCALF-Net/tests/gcalf/test_caf.py`:

- Window partition/reverse round-trip for exact and padded sizes.
- Output shape `(B,out,D,H,W)` for unequal input channel counts.
- **Padding invariance:** for spatial dims that are not window multiples, filling the padded region with a different constant must not change the output on the valid region. Fails whenever `key_padding_mask` is missing or wrong.
- **No NaN from fully-padded windows** at the smallest realistic stage shape.
- **Residual counted once:** force both attention outputs to zero (e.g. zero the attention output projections) with an identity `out_proj`, and assert the module returns the aligned CNN feature — not `2x` or `3x` it.
- Gradients reach `cnn_align`, `swin_align`, and both attention projections.
- Replacing one branch with zeros changes the output.
- Swapping branch inputs after channel-compatible projection changes the output.
- Invalid head/channel and invalid window settings fail early.
- Serialization round-trip is deterministic in evaluation mode.
- CUDA peak memory and wall time are captured at real stage-2/stage-5 shapes.

Memory acceptance must be declared before profiling, for example: the full model forward/backward fits the target GPU with at least 10% free memory at batch size 1. Do not assert an arbitrary byte count in a CPU unit test.

## 4.7 Fallback Ladder

Apply these in order and record any architecture change:

1. Reduce window from `(2,7,7)` to `(2,4,4)` — this also cuts padding, since stage feature maps are far more often multiples of 4 than of 7.
2. Enable activation checkpointing around CAF.
3. Use true windowed CAF at stage 5 only.
4. Run TransFuse-style BiFusion as a separately named fallback experiment.

Never silently replace CAF with BiFusion while retaining a cross-attention claim.

## 4.8 Integration Sequence

1. Complete window helper tests.
2. Complete module gradient and shape tests.
3. Profile standalone stage shapes.
4. Run an end-to-end encoder forward/backward.
5. Overfit two cases.
6. Run and resume the tiny task, then predict/evaluate.
7. Freeze the CAF config before launching five-fold runs.

**Deliverables:** `nndet/arch/encoder/gcalf/caf.py`, fusion registry, Hydra config, tests, stage profile, tiny metrics, and resolved architecture snapshot.

**Next:** `PHASE_5_integration_ablation.md`.
