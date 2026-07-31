# M0 verification record

**Status:** implementation complete; CUDA runtime verification blocked by the host Docker daemon.

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

Run that command on a Docker host with NVIDIA Container Toolkit/CDI configured before declaring M0 fully verified.
