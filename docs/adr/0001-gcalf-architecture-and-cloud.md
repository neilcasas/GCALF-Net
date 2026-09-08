# ADR 0001: GCALF Architecture and Cloud Training Contract

**Status:** Accepted | **Date:** 2026-07-26 | **Source:** `/grilling` session

## Context

The existing plan mixed incompatible definitions of GCALF-Net. It called TransFuse BiFusion cross-attention even though BiFusion has no Q/K/V attention; proposed a frequency-invariant per-channel scalar as a learnable frequency filter; described configuration through the nnDetection plan even though the released model passes encoder options through Hydra `model_cfg.encoder_kwargs`; and treated cloud execution as a list of providers rather than a recoverable workflow.

The released PDHD-Net is an nnDetection fork that performs lesion detection, per-lesion classification, and segmentation. Its encoder applies wavelet enhancement at stages `[1,3,4]` and CNN/Swin fusion at stages `[2,5]`. This sentence describes released-code archaeology only; ADR 0002 supersedes it for the study model with a fixed five-level FDSF/WAF contract.

## Decisions

1. Preserve lesion-level multi-task nnDetection as the primary task.
2. Define LFF as a grouped, low-resolution complex spectral grid interpolated to each runtime rFFT shape.
3. Parameterize the spectral transfer as `1 + delta`, initialized with `delta=0`, without an additional spatial residual.
4. Define CAF as true bidirectional windowed Q/K/V cross-attention between aligned CNN and Swin features.
5. Treat TransFuse BiFusion only as a separately named fallback or additional ablation.
6. Pass all GCALF options through `model_cfg.encoder_kwargs.gcalf_cfg` into the existing `Encoder`.
7. Require four models across all five official PI-CAI folds with one fixed primary seed.
8. Freeze the legacy PyTorch/Lightning/CUDA runtime until baseline and GCALF comparisons are complete.
9. Use a provider-neutral GPU VM and pinned Docker image for cloud execution.
10. Use AWS CLI v2 as the S3-compatible transfer contract, stage prepared data to local NVMe, and verify an immutable SHA-256 manifest before training.
11. Synchronize checkpoints and run state after every epoch so preemption loses at most one epoch.
12. Apply explicit least-privilege, encryption, retention, and logging controls to PI-CAI and derived artifacts.

## Consequences

- The thesis can accurately claim learnable frequency-selective filtering and cross-attention.
- Window helpers, interpolation behavior, memory profiling, and recovery tests become required engineering work.
- The required experiment contains 20 primary training runs before optional repeated seeds.
- CAF may require smaller windows, activation checkpointing, or a predeclared reduced fusion-level subset applied identically to WAF and CAF; ADR 0002 prohibits a CAF-only fallback.
- A BiFusion result cannot be reported as GCALF CAF without changing the method name and thesis claim.
- Dependency modernization is deferred and must be evaluated separately after the thesis comparison.

## Rejected Alternatives

| Alternative | Reason rejected |
|---|---|
| BiFusion as primary CAF | It is gating and Hadamard interaction, not Q/K/V cross-attention. |
| Per-channel scalar LFF | It applies one gain to every frequency and is not frequency-selective. |
| Full global 3D attention | Quadratic spatial-token memory is infeasible at the configured fusion stages. |
| Train through S3 mount | Throughput and failure behavior are less predictable than verified local staging. |
| Modernize runtime first | Framework changes would confound baseline-vs-GCALF attribution. |
| Case-level classifier as primary | It diverges from the released PDHD-Net task and discards lesion-level behavior. |

## Amendments — rev. 2 (2026-07-30)

The decisions above stand. These five corrections come from reading the released code rather than the papers, and they change *how* decisions 2, 4, and 7 are implemented, not whether.

| # | Correction | Evidence |
|---|---|---|
| A1 | `classifier_classes = 5`, not 6. nnDetection instance classes are **foreground-only and 0-indexed**, so labels map `lesion_ISUP k → class k-1` and a benign case is a zero-instance case, not a "class 0". Background is implicit in the sigmoid focal loss. The reported confusion matrix is 5×5 with misses and false positives beside it. | `nndet/planning/architecture/boxes/base.py:89` (`len(class_dct)`, `class_dct = dataset.json["labels"]`); PDHD-Net `README.md` already specifies `{"0":"GGG1" … "4":"GGG5"}` |
| A2 | The LFF grid is defined on **centered** frequency coordinates and `ifftshift`ed before use. `rfftn` orders the two full axes `[0,+f,…,Nyq,−f,…,−1]`, so interpolating a coarse grid over the raw index space cannot express a low-pass response and smooths across the Nyquist discontinuity. | `torch.fft.rfftn` output ordering |
| A3 | The LFF gain is **real-valued**, refining decision 2. A complex gain is unidentifiable: `x` real ⟹ `X` Hermitian ⟹ `irfftn` applies `(H(f)+conj(H(−f)))/2`, aliasing half the parameters. A real gain is exactly Hermitian, halves the parameters, and remains frequency-selective. | Hermitian symmetry of the rFFT of a real signal |
| A4 | CAF must apply the aligned CNN residual **exactly once** and must pass a `key_padding_mask` for padded window tokens. The draft skeleton counted the CNN path three times and let zero-padding act as valid attention keys. | `PHASE_4_caf.md §4.3–4.4`; same defect class as the `1+delta`-plus-residual trap in decision 3 |
| A5 | The real schedule is 50 epochs × 2500 batches + 10 SWA epochs ≈150k steps per run (no `EarlyStopping` exists, and none may be added). This makes decision 7's 20-run matrix ~10–20 GPU-days, so a **budget ladder is pre-committed** before the fold-0 pilots: full matrix → halve batches-per-epoch across all runs → 14-run reduced matrix (never drop folds for some configs only). | `nndet/conf/train/v001.yaml`; the README's "1000 epochs" is stale |

Also recorded: the working repository is now `GCALF-Net/`, a copy of `PDHD-Net/` at tag `pdhd-upstream` with the upstream remote removed. `PDHD-Net/` stays pristine as the released reference, and only one of the two may be installed at a time since both provide the `nndet` package.

## Amendments — rev. 3 (2026-09-08)

These three corrections come from implementing LFF (M6) and prototyping the spec's own reference
code in the pinned `gcalf:m0` image; they change how decision 2 is implemented, not whether.

| # | Correction | Evidence |
|---|---|---|
| A6 | `spherical_mask_rfft` now exists (`nndet/arch/encoder/gcalf/frequency_mask.py`) and takes the **spatial** shape `(D, H, W)`, not the half-spectrum shape `rfftn` produces. Both `ARCHITECTURE.md` §6 and `PHASE_3_lff.md` §3.1's reference code called it with `spectrum.shape[-3:]`, which crashes: `W//2+1` does not determine `W`, so the resulting mask shape mismatches the spectrum on the trailing axis. Both documents' reference code blocks are corrected in place. | `RuntimeError: The size of tensor a (17) must match the size of tensor b (33)` reproduced with the as-written call at shape `(16,64,33)` vs `(16,64,17)`; fixed by passing `x.shape[-3:]` |
| A7 | With the corrected call, LFF at `delta_weight=0` matches FDSF's output at `atol=1e-5` with roughly 10x margin at every measured shape, odd and even alike (max abs diff 3.7e-08 to 1.4e-06 across shapes from `(4,8,8)` to `(32,256,256)`). The `atol=1e-5` gate in `test_lff.py` is honest, not loosened for convenience. | Measured directly in `gcalf:m0` (torch 1.10.1) before writing `tests/gcalf/test_lff.py` |
| A8 | The spec default `grid_size=(4,8,8)` is even on all three axes; with `align_corners=True`, no grid sample lands exactly on DC (samples land at centered indices 0,5,10,15 for a size-16 axis; DC sits at index 8). PHASE_3 §3.1's caveat about "within half a bin" holds only for **odd** grid dims. This is not fatal — DC still receives a well-defined interpolated blend (an all-ones grid gives `gain@DC == 1.0` exactly) — but no single learned parameter controls DC directly. Recorded here rather than changed; the deferred V-gate's required radial-response visualization is the actual gate on whether this weakens the learned low-pass response in practice. | Grid-sample index arithmetic for `align_corners=True`, `grid_size=4`, output size 16 |

## Review Triggers

Revisit this ADR only if the released encoder cannot train end-to-end, the pinned runtime cannot run on available GPU infrastructure, or measured windowed WAF/CAF cannot fit at the five-level contract's shared fusion locations. Any revision must update the method claim, configs, experiment matrix, and all affected documentation together.
