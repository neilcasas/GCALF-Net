# Phase 4 — Bidirectional Windowed Cross-Attention Fusion

**Milestone:** M7 | **Depends on:** Phase 2 (WAF wired, tagged `baseline-v1`) | **Blocks:** full GCALF-Net

## Goal

Replace **WAF**'s self-attention one-for-one at the fusion levels frozen by M2's profiling
(`PHASE_2 §2.6`) with true bidirectional Q/K/V cross-attention
between aligned CNN and Swin features — the *only* change from the frozen baseline
(`ARCHITECTURE.md §7`). "CAF" always means this. TransFuse BiFusion is a named, non-cross-attention
fallback and must never be relabeled CAF.

## Exit Criteria

- [ ] **L:** CNN queries Swin keys/values and Swin queries CNN keys/values.
- [ ] **L:** window partition/reverse is lossless for divisible and padded shapes.
- [ ] **L:** padded window positions are masked out of attention (`key_padding_mask`), not attended as real keys.
- [ ] **L:** the aligned CNN feature reaches the output through exactly one residual path.
- [ ] **L:** both branch projections receive gradients.
- [ ] **V:** tensors from every frozen fusion level pass the GPU memory gate at the planner's
      resolved patch size — the same levels and criterion WAF was profiled against at M2.
- [ ] **V:** `caf_only` completes tiny-task train, resume, predict, and evaluation.

## 4.1 Integration contract

`Encoder.forward` already resizes each Swin feature to the CNN spatial shape before fusion —
unchanged from the WAF baseline. Preserve WAF's interface exactly so CAF is a drop-in replacement:

```python
WindowedCrossAttentionFusion3D(
    cnn_channels, transformer_channels, out_channels,
    window_size=(2, 7, 7), num_heads=4, dropout=0.0,
)
fused = module(cnn_feat, resized_swin_feat)
```

Add to `nndet/arch/encoder/gcalf/registry.py::build_fusion_module` (already stubbed in Phase 2
with the `waf` branch):

```python
if kind == "caf":
    return WindowedCrossAttentionFusion3D(cnn_channels, transformer_channels, out_channels, **options)
```

## 4.2 Why windowed, and why bidirectional

Global spatial cross-attention over `N=D*H*W` tokens stores an `N×N` matrix per head and is not
viable at high-resolution encoder levels. Partitioning aligned features into windows bounds attention to
`Nw = wd*wh*ww` tokens; complexity becomes `O(n_windows * Nw²)` — the same bound WAF already
accepts for self-attention.

```text
CNN output  = Attention(Q=cnn,  K=swin, V=swin)
Swin output = Attention(Q=swin, K=cnn,  V=cnn)
Fusion      = projection([cnn residual, CNN output, Swin output])
```
This is materially different from WAF's single self-attention pass over one aligned feature, and
from independent SE/spatial gates or a Hadamard product (TransFuse BiFusion).

## 4.3 Window helpers

Reuse (or verify, if WAF's wiring in Phase 2 didn't already need them) window partition/reverse
helpers before writing the attention module itself:

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
Metadata carries batch size, original spatial size, padded spatial size, window size. Test
`reverse(partition(x)) == x` exactly.

**The mask is not optional.** With `window_size=(2,7,7)` on feature maps rarely a multiple of 7,
padding can be a large fraction of border windows. Unmasked, those zero tokens are valid keys:
attention spends probability mass on them and the model learns an input-size-dependent bias. Both
`MultiheadAttention` calls take `key_padding_mask`. A window that is *entirely* padding produces
NaN under softmax masking — drop those windows, or keep one unmasked position and discard the
result on reverse; assert no NaN either way.

## 4.4 Module

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
            cnn_w, swin_w, swin_w, key_padding_mask=pad_mask, need_weights=False,
        )
        swin_cross, _ = self.swin_queries_cnn(
            swin_w, cnn_w, cnn_w, key_padding_mask=pad_mask, need_weights=False,
        )
        # attention terms only: `cnn` re-enters once, as the residual below
        fused_w = self.norm(cnn_cross + swin_cross)
        fused = window_reverse_3d(fused_w, metadata)
        return self.out_proj(torch.cat([cnn, fused], dim=1)) + cnn
```

**Count the CNN path exactly once.** A defect of this exact shape has already been documented
twice in this codebase's history (`fused_w = norm(cnn_w + cnn_cross + swin_cross)` *and* the
concat *and* the trailing `+ cnn` — tripling the aligned feature — was the earlier draft's bug;
the same class of error is called out for LFF's identity-plus-residual trap,
`PHASE_3_lff.md §3.3`). Keep the attention outputs in `fused_w` and let `cnn` reach the output
through the concat and the single trailing residual. If a pre-norm residual inside the window is
preferred instead, remove the trailing `+ cnn` — one path, not two, either way.

Confirm `batch_first=True` and `need_weights=False` behave as expected in the pinned PyTorch 1.10
image; transpose to `(tokens,batch,channels)` explicitly if `batch_first` is unavailable in that
exact patch version rather than modernizing the runtime.

**Known simplification:** no relative position bias, no shifted-window pass — attention is
permutation-invariant inside a window and carries no cross-window information. This is a
deliberate cost/benefit choice against Swin's SW-MSA; record it as an architecture limitation in
the thesis, and treat a learnable relative position bias as the first upgrade if CAF underperforms
for reasons other than memory.

## 4.5 Configuration

```yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: fdsf
      fusion_type: caf
      num_levels: 5
      fusion_levels: [0, 1, 2, 3, 4]   # starting point; frozen by M2 profiling
      caf:
        window_size: [2, 7, 7]
        num_heads: 4
        dropout: 0.0
```
The registry receives the stage channel count and may derive `num_heads` per stage when one
constant is invalid. Store resolved per-stage values in the experiment snapshot.

## 4.6 Tests and profiling (`tests/gcalf/test_caf.py`)

- Window partition/reverse round-trip for exact and padded sizes.
- Output shape `(B,out,D,H,W)` for unequal input channel counts.
- **Padding invariance:** for spatial dims not a window multiple, filling the padded region with a
  different constant must not change the output on the valid region. Fails whenever
  `key_padding_mask` is missing or wrong.
- **No NaN from fully-padded windows** at the smallest realistic stage shape.
- **Residual counted once:** force both attention outputs to zero (zero the attention output
  projections) with an identity `out_proj`, and assert the module returns the aligned CNN feature
  exactly — not `2×` or `3×` it.
- Gradients reach `cnn_align`, `swin_align`, and both attention projections.
- Replacing one branch with zeros changes the output; swapping branch inputs (after
  channel-compatible projection) changes the output.
- Invalid head/channel and invalid window settings fail early.
- Serialization round-trip is deterministic in evaluation mode.
- CUDA peak memory and wall time captured at every frozen fusion level, at the planner's resolved
  patch size; declare the memory
  acceptance criterion before profiling (e.g. "fits the target GPU with ≥10% free memory at batch
  1") — never assert an arbitrary byte count in a CPU unit test.

## 4.7 Fallback ladder

M2's profiling should already have frozen an affordable `fusion_levels`; this ladder handles a CAF
that still will not fit there. Apply in order, and freeze the outcome before any comparative fold
is trained:
1. Reduce window from `(2,7,7)` to `(2,4,4)` — also cuts padding, since feature maps are far more
   often multiples of 4 than of 7.
2. Enable activation checkpointing around both CAF and WAF.
3. Drop the highest-resolution level(s) from the frozen subset — that is where the cost
   concentrates — and re-run the WAF arm with the same reduced subset.
4. TransFuse-style BiFusion as a separately named fallback experiment.

Never silently replace CAF with BiFusion while retaining a cross-attention claim, and never reduce
only CAF while leaving the WAF control at different fusion locations.

## 4.8 Integration sequence

1. Complete window helper tests (reuse from Phase 2's WAF wiring if already present).
2. Complete module gradient and shape tests.
3. Profile standalone shapes for every frozen fusion level.
4. Run an end-to-end encoder forward/backward.
5. Overfit two cases (reuse Phase 2's cases).
6. Run and resume the tiny task, then predict/evaluate.
7. Freeze the CAF config before launching five-fold runs.

**Deliverables:** `nndet/arch/encoder/gcalf/caf.py`, fusion registry entry, Hydra config, tests,
stage profile, tiny metrics, resolved architecture snapshot.

**Next:** `PHASE_5_integration_ablation.md`.
