# Phase 0 — Environment & Repository Setup

**Milestone:** M0 · **Depends on:** nothing · **Blocks:** everything
**Goal:** one authoritative `gcalf:m1` image (`ARCHITECTURE.md §12`) where every reported result
must reproduce. The local RTX 4050 may run the CUDA extension smoke test in that image; its 6 GB
still excludes full training. Any disposable scratch environment remains outside the reported
runtime and is not used to build or install GCALF-Net.

**Working repository:** `GCALF-Net/` — a copy of `PDHD-Net/` at tag `pdhd-upstream` (commit
`e2330cf`), upstream remote removed. `git diff pdhd-upstream` is the thesis's exact changeset.
`PDHD-Net/` stays pristine; install only one of the two (`nndet` package name collides).

**Definition of done (exit criteria):**
- [x] **L:** `import nndet, nndet._C` succeeds and resolves to `GCALF-Net/nndet`, not `PDHD-Net/nndet`.
- [x] **L:** `python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py` passes.
- [x] **L:** `import picai_prep, picai_eval, medcam` succeeds.
- [x] **L:** a random-tensor forward pass through `nndet.arch.encoder.modular.Encoder` runs on CPU.
- [x] **V:** `tests/test_csrc_cuda.py` runs (not skips) in the GPU container; the passing record is
      in `docs/m0-verification.md`.
- [x] `nndet/arch/encoder/gcalf/` scaffolding (`fdsf.py`, `waf.py`, `lff.py`, `caf.py`,
      `grade_head.py`, `registry.py`) committed as empty modules on branch `feat/env-and-data`.
- [x] `environment.yml` + a sorted, non-editable `pip freeze` lock committed;
      `docs/m0-verification.md` updated with the new attempt's outcome.

---

## 0.1 The authoritative environment is fixed before the matrix

`requirements.txt` uses the minimum stack that supports Blackwell: `pytorch_lightning>=2.5,<2.6`,
`nnunet==1.7.1`, `numpy<2`, and an unpinned modern `SimpleITK`. These choices anchor:

| Component | Version |
|---|---|
| Python | 3.11 (authoritative Docker image) |
| PyTorch / torchvision / torchaudio | 2.7.1 / 0.22.1 / 2.7.1 (CUDA 12.8) |
| CUDA | 12.8 (driver >=570) |
| pytorch-lightning | 2.5.x |
| nnunet | 1.7.1 |

GFNet/TransFuse/UCTransNet/DCA are **never installed** — copy classes out of them. Z-SSMNet is
**never installed** — read-only reference.

## 0.2 Why local training is still out of scope, and what the local CUDA gate is for

The local RTX 4050 is `sm_89`. CUDA 12.8 targets `8.0;8.6;8.9;9.0;12.0+PTX`, so the extension
can be compiled and smoke-tested locally. The card still has only 6 GB, so it cannot host a real
training run. Consequences:

- **Every gate whose result must match what gets reported** runs inside `gcalf:m1`:
  unit tests, config/registry assertions, manifest/fold validation, real preprocessing on fold-0
  cases, synthetic CPU forward/backward, the 2-case CPU micro-overfit (Phase 2). These are the
  **L** gates throughout the roadmap.
- **The CUDA extension build and `test_csrc_cuda.py`** can run locally; FDSF/WAF/LFF/CAF memory
  profiling and real training remain **V** gates on Vast.ai.
- A **separate, disposable modern-torch/CUDA conda env** (e.g. `conda create -n gcalf-scratch
  python=3.11` + a current `torch`/`torch.fft` build) may run on the local 4050 to iterate on
  LFF/CAF tensor math — shape, gradient flow, the orientation-gate logic — before porting the
  validated implementation into the pinned stack by hand. Never `pip install -e` `GCALF-Net/`
  into this environment, never commit its lockfile, and never cite a result produced there in the
  thesis. It exists purely to make the edit-test loop faster than round-tripping through Docker
  for pure-math bugs that have nothing to do with the pinned stack.

## 0.3 Step-by-step (authoritative environment)

1. **Create the matching Conda development environment:**
   ```bash
   conda env create -f environment.yml
   conda activate gcalf
   ```
2. **Install the local checkout and pinned sibling tools:**
   ```bash
   pip install -e . -e ../picai_prep -e ../picai_eval -e ../M3d-Cam
   ```
   Install **only** GCALF-Net — `PDHD-Net/` ships the same `nndet` package name.
   Run `git -C ../picai_eval submodule update --init --recursive` before its editable install.
3. **Set nnDetection env vars** (persist via the conda `activate.d` script):
   ```bash
   export det_data=/data/nnDet_raw
   export det_models=/data/nnDet_results
   ```
4. **Verify imports and CPU forward pass:**
   ```bash
   python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py
   ```
5. **Build and verify the authoritative image (locally when a compatible GPU is available, then on Vast.ai):**
   ```bash
   docker build -t gcalf:m1 .
   docker run --rm --gpus all gcalf:m1 python -m pytest -q tests
   ```
   The CUDA test must run, not skip; assert `torch.cuda.is_available()` first inside the
   container. This gate passed on 2026-09-05; retain its log and rerun it after image or extension
   rebuilds.
6. **Scaffold the gcalf code tree** (empty modules + `__init__.py` for `fdsf.py`, `waf.py`,
   `lff.py`, `caf.py`, `grade_head.py`, `registry.py` — see `ARCHITECTURE.md §2`). Commit.

## 0.4 nnDetection csrc build failure — the top risk

| Symptom | Fix |
|---|---|
| `nvcc` not found | Install matching `cudatoolkit-dev` in conda, or use `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`. |
| ABI / torch version mismatch on import | Rebuild after installing the exact pinned torch: `pip install -e GCALF-Net/ --no-build-isolation`. |
| No local CUDA at all | The RTX 4050 can now run the CUDA extension smoke test in the target image; otherwise run CPU import/encoder checks and keep M0 open until a GPU host executes the CUDA gate. |
| Docker CDI/GPU-vendor discovery fails on the build host | Record the exact error in `docs/m0-verification.md` (as already done once); this blocks M0's V gate — escalate rather than declaring M0 done without it. |

## 0.5 Docker (build once, reuse local + cloud)

The committed `Dockerfile` is the M0 runtime:
```dockerfile
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
RUN pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
ENV det_data=/opt/data det_models=/opt/models
```
Generate `env.lock.txt` from the completed image; record base/image digests in
`docs/m0-verification.md`. The migration is permitted by ADR 0004 because no official matrix result
existed when the build changed. The complete provider-neutral VM, S3 storage, NVMe
staging, recovery, and security workflow is in `docs/CLOUD_DEPLOYMENT_PLAN.md`.

## 0.6 Deliverables & commit

- Record the authoritative image digest and generated lockfile after the toolchain migration.
- Files: `environment.yml`, `requirements-tools.txt`, `env.lock.txt`, `Dockerfile`,
  `tests/test_{imports,encoder_cpu,csrc_cuda}.py`, empty
  `nndet/arch/encoder/gcalf/{__init__,fdsf,waf,lff,caf,grade_head,registry}.py`.

**Next:** `PHASE_1_data_pipeline.md`.
