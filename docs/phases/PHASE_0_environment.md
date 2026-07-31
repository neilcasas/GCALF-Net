# Phase 0 — Environment & Repository Setup

**Milestone:** M0 · **Depends on:** nothing · **Blocks:** everything
**Goal:** a single reproducible Python environment in which PDHD-Net (nnDetection), the PI-CAI tools, and medcam all import and run, plus the `gcalf/` code skeleton in place. The CUDA Docker image is the authoritative M0 runtime; the Conda environment mirrors it for development.

**Working repository:** `GCALF-Net/` — a copy of `PDHD-Net/` taken at tag `pdhd-upstream`, with the upstream git remote removed. All code and commits below happen in `GCALF-Net/`. `PDHD-Net/` stays pristine as the released reference; `git diff pdhd-upstream` in `GCALF-Net/` is the thesis's "what we changed" diff.

**Definition of done (exit criteria):**
- [ ] `import nndet, nndet._C` succeeds (nnDetection CUDA `csrc` compiled) **and** `nndet` resolves to `GCALF-Net/nndet`, not `PDHD-Net/nndet`.
- [ ] `python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py tests/test_csrc_cuda.py` passes in the GPU container.
- [ ] `import picai_prep, picai_eval, medcam` succeeds.
- [ ] A random-tensor forward pass through `nndet.arch.encoder.modular.Encoder` runs on **CPU** (proves the model graph builds).
- [ ] `nndet/arch/encoder/gcalf/`, `gcalf_configs/`, `gcalf_data/`, `gcalf_eval/`, `tests/gcalf/` scaffolding committed on branch `feat/env-and-data`.
- [ ] `environment.yml` + a sorted, non-editable `pip freeze` lock committed; `docs/m0-verification.md` records the source revisions and image digest.

---

## 0.1 The environment is dictated by PDHD-Net, not chosen

PDHD-Net's `requirements.txt` pins a 2021 stack: `pytorch_lightning>=1.3.1,<=1.4.2`, `nnunet==1.7.1`, `SimpleITK<2.1.0`, `torchmetrics<=0.7.3`. **Do not fight these.** They imply:

| Component | Version |
|---|---|
| Python | 3.8 (authoritative Docker image; 3.9 is supported for local development) |
| PyTorch / torchvision / torchaudio | 1.10.1 / 0.11.2 / 0.10.1 (CUDA 11.3) |
| CUDA | 11.1 or 11.3 (must match the torch build) |
| pytorch-lightning | 1.4.2 |
| nnunet | 1.7.1 |

Everything else (picai tools, medcam) bends to this. GFNet/TransFuse/UCTransNet/DCA are **never installed** — you copy classes out of them. Z-SSMNet is **never installed** — read-only reference.

## 0.2 Step-by-step

1. **Create the matching Conda development environment:**
   ```bash
   conda env create -f environment.yml
   conda activate gcalf
   ```
2. **Install the local checkout and pinned sibling tools** (after verifying the commits in `requirements-tools.txt`):
   ```bash
   pip install -e . -e ../picai_prep -e ../picai_eval -e ../M3d-Cam
   ```
   - Install **only** GCALF-Net. `PDHD-Net/` ships the same `nndet` package name; installing both makes imports depend on path order.
   - Run `git -C ../picai_eval submodule update --init --recursive` before its editable install.
3. **Set nnDetection env vars** (this fork needs these dirs):
   ```bash
   export det_data=/data/nnDet_raw
   export det_models=/data/nnDet_results
   ```
   Put them in the conda `activate.d` script so they persist.
4. **Verify imports and CPU forward pass:**
   ```bash
   python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py
   ```
5. **Build and verify the authoritative image on a GPU host:**
   ```bash
   docker build -t gcalf:m0 .
   docker run --rm --gpus all gcalf:m0 python -m pytest -q tests
   ```
   The CUDA test must run rather than skip; first assert `torch.cuda.is_available()` in the container.
6. **Scaffold the gcalf code tree** (empty modules + `__init__.py`) as laid out in `SPEC.md §2`. Commit.

## 0.3 nnDetection csrc build failure — the top risk

nnDetection compiles CUDA extensions (NMS, etc.). Against torch 1.10/CUDA 11.3 this usually works; failure modes and fallbacks:

| Symptom | Fix |
|---|---|
| `nvcc` not found | Install matching `cudatoolkit-dev` in conda, or use the Docker base `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`. |
| ABI / torch version mismatch on import | Rebuild extension after installing the exact pinned torch: `pip install -e GCALF-Net/ --no-build-isolation`. |
| Can't build at all (no CUDA locally) | Run only the CPU import/encoder smoke locally and defer the authoritative Docker build to a GPU host. M0 remains open until the CUDA extension test runs there. |

## 0.4 Docker (build once, reuse local + cloud)

The committed `Dockerfile` is the M0 runtime. It uses the CUDA 11.3 development image, installs the exact PyTorch tuple, clones the three tool repositories at the revisions recorded in `requirements-tools.txt`, initializes `picai_eval` submodules, and compiles `nndet._C` with `FORCE_CUDA=1`.
```dockerfile
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
    torchaudio==0.10.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
ENV det_data=/opt/data det_models=/opt/models
```
Build now, so "works on my machine" == "works in the cloud". Generate `env.lock.txt` from the completed image, then record the base and image digests in `docs/m0-verification.md`. Do not modernize PyTorch/Lightning during the GCALF architecture experiment because that would confound the baseline comparison.

The complete provider-neutral VM, S3-compatible storage, local NVMe staging, recovery, and security workflow is in `docs/CLOUD_DEPLOYMENT_PLAN.md`.

## 0.5 Deliverables & commit

- Branch `feat/env-and-data` in `GCALF-Net/`, commit "env: pinned gcalf conda env + gcalf scaffold + CPU encoder smoke".
- Files: `environment.yml`, `requirements-tools.txt`, `env.lock.txt`, `Dockerfile`, `tests/test_{imports,encoder_cpu,csrc_cuda}.py`, empty `nndet/arch/encoder/gcalf/{__init__,lff,caf,registry}.py`.

**Next:** `PHASE_1_data_pipeline.md`.
