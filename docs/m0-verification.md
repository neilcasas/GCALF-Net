# M0 verification record

**Status:** complete — CUDA runtime verification (the V gate) now passes.

## 2026-09-05 — CUDA V gate closed, gcalf scaffold completed

The image was rebuilt after adding the remaining empty `gcalf/` scaffold modules (`fdr.py`,
`waf.py`, `grade_head.py` — `caf.py`, `lff.py`, `registry.py`, `__init__.py` already existed):

`fdr.py` is the historical scaffold name from that build. Under ADR 0002's aligned terminology,
Phase 2 replaces it with `fdsf.py`; this record otherwise remains a factual image-build log and is
not updated to match later naming.

- Base image: `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel@sha256:913e6689c5958b187e65561e528ec6c3ce8a02deedcdd38cb50c9cab301907bb` (unchanged).
- Final image: `gcalf:m0@sha256:5bbba84ddc12ffa932c280c903e6ceba883c749a05faa3b6b3fe24dd90461bb7`.
- `docker run --rm gcalf:m0 python -m pytest -q tests` (no GPU): **22 passed, 2 skipped**
  (`test_csrc_cuda.py`'s CUDA-only test, and `test_overfit.py`'s M2 gate behind `RUN_GCALF_M2=1`).
- `docker run --rm --gpus all gcalf:m0 python -m pytest -q tests/test_csrc_cuda.py`: **1 passed** —
  the NVIDIA Container Toolkit/CDI failure recorded below on 2026-07-31 no longer reproduces on
  this host (`nvidia-smi` now resolves `NVIDIA GeForce RTX 4050 Laptop GPU` through the CDI path).
- `docker run --rm --gpus all gcalf:m0 python -m pytest -q tests`: **23 passed, 1 skipped** (only
  the M2 gate remains skipped, correctly gated off by `RUN_GCALF_M2`).
- `env.lock.txt` regenerated to exactly match the sorted (`LC_ALL=C sort`)
  `python -m pip list --format=freeze` output of the rebuilt image. Only transitive, unpinned
  packages moved (`GitPython` 3.1.57→3.1.61, `SQLAlchemy` 2.0.51→2.0.52, `charset-normalizer`
  3.4.9→3.5.1, `graphql-core` 3.2.11→3.2.12, `hydra-core` 1.3.4→1.3.6); every package pinned in
  `requirements.txt`/`requirements-tools.txt` (torch, pytorch-lightning, nnunet, SimpleITK,
  picai_prep/picai_eval/medcam revisions) is unchanged.

M0's exit criteria (`PHASE_0_environment.md`) are now all satisfied, including the previously-open
V gate. Phase 1 is unblocked.

## 2026-07-31 — initial build, CUDA V gate blocked (superseded above)

The final image was built successfully on 2026-07-31:

- Base image: `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel@sha256:913e6689c5958b187e65561e528ec6c3ce8a02deedcdd38cb50c9cab301907bb`.
- Final image: `gcalf:m0@sha256:870df813337d1b6f9507048f9a8b5343d2b34d026849b87cfa2a6a099582392e`.
- `python -m pytest -q tests` without GPU access: **8 passed, 1 skipped**. The skipped test is the intentionally GPU-only CUDA NMS check.
- `env.lock.txt` exactly matches the sorted `python -m pip list --format=freeze` output of the final image.

The required GPU command could not start:

```text
docker run --rm --gpus all gcalf:m0 python -m pytest -q tests/test_csrc_cuda.py
docker: Error response from daemon: failed to discover GPU vendor from CDI: no known GPU vendor found
```

This was resolved on the 2026-09-05 host (see above) — the failure was host/CDI configuration, not
a code or image defect.
