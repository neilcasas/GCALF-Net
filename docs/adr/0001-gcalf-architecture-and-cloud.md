# ADR 0001: GCALF Architecture and Cloud Training Contract

**Status:** Accepted | **Date:** 2026-07-26 | **Source:** `/grilling` session

## Context

The existing plan mixed incompatible definitions of GCALF-Net. It called TransFuse BiFusion cross-attention even though BiFusion has no Q/K/V attention; proposed a frequency-invariant per-channel scalar as a learnable frequency filter; described configuration through the nnDetection plan even though the released model passes encoder options through Hydra `model_cfg.encoder_kwargs`; and treated cloud execution as a list of providers rather than a recoverable workflow.

The released PDHD-Net is an nnDetection fork that performs lesion detection, per-lesion classification, and segmentation. Its encoder applies wavelet enhancement at stages `[1,3,4]` and CNN/Swin fusion at stages `[2,5]`.

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
- Stage-2 CAF may require smaller windows, activation checkpointing, or a reported stage-5-only architecture.
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

## Review Triggers

Revisit this ADR only if the released encoder cannot train end-to-end, the pinned runtime cannot run on available GPU infrastructure, or measured windowed CAF cannot fit even at stage 5. Any revision must update the method claim, configs, experiment matrix, and all affected documentation together.
