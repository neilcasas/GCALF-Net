# ADR 0004: CUDA 12.8 / PyTorch 2.7.1 Toolchain for the Pre-Matrix Runs

**Status:** Accepted | **Date:** 2026-09-22 | **Supersedes:** ADR 0001 decision 8 and its
legacy-runtime clauses in the environment and cloud runbooks

## Context

The repository was pinned to Python 3.8, PyTorch 1.10.1/CUDA 11.3, and PyTorch Lightning 1.4.2.
CUDA 11.3 cannot emit code beyond sm_86, while the available RTX 4050 is sm_89 and the target
RTX 50-series is Blackwell sm_120. PyTorch support for sm_120 begins with the CUDA 12.8 wheels in
PyTorch 2.7.0, and the driver requirement is at least 570. The custom extension is only the
244-line nndet._C NMS surface; it has no architecture-specific PTX or third-party CUDA
dependencies. The required ATen port is therefore bounded and reviewable.

Most importantly, the official 4-arm x 5-fold matrix had not started. Only the non-poolable fold-0
pilot existed, so changing the runtime now cannot reorder official results. The pilot's checkpoints
and results are discarded and the pilot is rerun under this single stack; no 1.4-era checkpoint is
resumed or converted.

## Decision

Adopt the minimum stack that reaches Blackwell while preserving the scientific contract:

| Component | Decision |
|---|---|
| Image | pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel |
| Python | 3.11 |
| torch / torchvision / torchaudio | 2.7.1 / 0.22.1 / 2.7.1 +cu128 |
| PyTorch Lightning | >=2.5,<2.6 |
| torchmetrics | >=1.0,<2 |
| NumPy | <2 |
| nnU-Net / SimpleITK | nnunet==1.7.1; SimpleITK unpinned |
| CUDA architectures | 8.0;8.6;8.9;9.0;12.0+PTX |

The Lightning migration replaces removed epoch-end hooks with explicit detached output buffers,
uses accelerator="gpu" plus devices and DDPStrategy, removes terminate_on_nan rather than
enabling the much slower anomaly detector, and passes resume checkpoints through fit(ckpt_path=…).
SWA is ported to 2.x scheduler-config dataclasses and retains the existing post-SWA checkpoint
semantics. torch.load(..., weights_only=False) is explicit for the project's trusted checkpoints.
FFT/attention autocast-disabled regions remain disabled; the historical GradScaler scale workaround
must be revalidated on the new hardware.

## Consequences

Ampere remains supported, Ada and Hopper become eligible, and Blackwell can compile/run the same
extension. The local RTX 4050 can now build and execute tests/test_csrc_cuda.py, but its 6 GB VRAM
does not support a real training run. The planner's fixed ~11.5 GB memory model is unchanged so
patch and batch plans remain comparable across cards.

All fold-0 pilot checkpoints are obsolete. A fresh smoke run must exercise swa_epochs > 0; a short
Blackwell overfit and CUDA NMS run are required before the matrix. The M5 GPU-hour and price budget
must be re-measured on the selected card; the old RTX 3090 contention factor is not evidence for
the new hardware. The 2.x ModelCheckpoint.state_key collision described in ADR 0003 is fixed by
the new monitor/mode-derived key, but any protocol change based on that fix is deferred and must be
proposed separately.

## Verification and review trigger

Record the target/base image digests, nndet_env, torch.cuda.get_arch_list() including sm_120,
non-skipped tests/test_csrc_cuda.py, the AMP overfit, and the SWA smoke run in the M0/M2 evidence
records. If the target image cannot build the extension or the new stack changes a locked metric,
stop before the official matrix and amend this ADR with the measured cause.
