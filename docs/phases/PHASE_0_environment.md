# Phase 0 — Environment & Repository Setup

**Milestone:** M0 · **Depends on:** nothing · **Blocks:** everything
**Goal:** a single reproducible Python environment in which PDHD-Net (nnDetection), the PI-CAI tools, and medcam all import and run, plus the `gcalf/` code skeleton in place.

**Working repository:** `GCALF-Net/` — a copy of `PDHD-Net/` taken at tag `pdhd-upstream`, with the upstream git remote removed. All code and commits below happen in `GCALF-Net/`. `PDHD-Net/` stays pristine as the released reference; `git diff pdhd-upstream` in `GCALF-Net/` is the thesis's "what we changed" diff.

**Definition of done (exit criteria):**
- [ ] `python -c "import nndet"` succeeds (nnDetection CUDA `csrc` compiled) **and** resolves to `GCALF-Net/nndet`, not `PDHD-Net/nndet` — check `nndet.__file__`.
- [ ] `python GCALF-Net/tests/test_imports.py` passes.
- [ ] `python -c "import picai_prep, picai_eval, medcam"` succeeds.
- [ ] A random-tensor forward pass through `nndet.arch.encoder.modular.Encoder` runs on **CPU** (proves the model graph builds).
- [ ] `nndet/arch/encoder/gcalf/`, `gcalf_configs/`, `gcalf_data/`, `gcalf_eval/`, `tests/gcalf/` scaffolding committed on branch `feat/env-and-data`.
- [ ] `environment.yml` + `pip freeze > env.lock.txt` committed.

---

## 0.1 The environment is dictated by PDHD-Net, not chosen

PDHD-Net's `requirements.txt` pins a 2021 stack: `pytorch_lightning>=1.3.1,<=1.4.2`, `nnunet==1.7.1`, `SimpleITK<2.1.0`, `torchmetrics<=0.7.3`. **Do not fight these.** They imply:

| Component | Version |
|---|---|
| Python | 3.8 or 3.9 |
| PyTorch | 1.10.x (supports `torch.fft.rfftn`, works with lightning 1.4.2) |
| CUDA | 11.1 or 11.3 (must match the torch build) |
| pytorch-lightning | 1.4.2 |
| nnunet | 1.7.1 |

Everything else (picai tools, medcam) bends to this. GFNet/TransFuse/UCTransNet/DCA are **never installed** — you copy classes out of them. Z-SSMNet is **never installed** — read-only reference.

## 0.2 Step-by-step

1. **Create env:**
   ```bash
   conda create -n gcalf python=3.9 -y && conda activate gcalf
   ```
2. **Install torch first** (pin CUDA to your GPU / cloud):
   ```bash
   pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
     --extra-index-url https://download.pytorch.org/whl/cu113
   ```
3. **Install GCALF-Net editable** (this pulls nnDetection deps and compiles `nndet/csrc/`):
   ```bash
   pip install -e GCALF-Net/
   ```
   - Install **only** GCALF-Net. `PDHD-Net/` ships the same `nndet` package name; installing both makes imports depend on path order.
   - If csrc compilation fails, that is the #1 blocker — see 0.3.
4. **Install PI-CAI tools + medcam:**
   ```bash
   pip install -e picai_prep/ -e picai_eval/ -e M3d-Cam/
   # or: pip install picai_prep picai_eval medcam
   ```
5. **Set nnDetection env vars** (nnDetection needs these dirs):
   ```bash
   export nnDet_raw=/data/nnDet_raw
   export nnDet_prep=/data/nnDet_prep
   export nnDet_results=/data/nnDet_results
   ```
   Put them in the conda `activate.d` script so they persist.
6. **Verify imports:**
   ```bash
   python -c "import nndet, picai_prep, picai_eval, medcam; print(nndet.__file__)"  # must be under GCALF-Net/
   python GCALF-Net/tests/test_imports.py
   ```
7. **CPU forward-pass smoke** (`tests/test_encoder_cpu.py`): build `Encoder` with a tiny config, feed `torch.randn(1,3,16,64,64)`, assert it returns a list of feature maps without error. This proves the Swin+CNN+wavelet+fusion graph is wired before you touch data.
8. **Scaffold the gcalf code tree** (empty modules + `__init__.py`) as laid out in `THESIS_PLAN.md §2`. Commit.

## 0.3 nnDetection csrc build failure — the top risk

nnDetection compiles CUDA extensions (NMS, etc.). Against torch 1.10/CUDA 11.3 this usually works; failure modes and fallbacks:

| Symptom | Fix |
|---|---|
| `nvcc` not found | Install matching `cudatoolkit-dev` in conda, or use the Docker base `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`. |
| ABI / torch version mismatch on import | Rebuild extension after installing the exact pinned torch: `pip install -e GCALF-Net/ --no-build-isolation`. |
| Can't build at all (no CUDA locally) | For **CPU smoke tests only**, nnDetection has CPU paths for most ops; skip csrc, do model-graph/unit tests on CPU, defer csrc to the GPU/cloud box (Phase 12/cloud). |

## 0.4 Docker (build once, reuse local + cloud)

High-level Dockerfile (full version deferred to Phase 6/cloud; skeleton here so the env is captured early):
```dockerfile
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
COPY GCALF-Net picai_prep picai_eval M3d-Cam /workspace/
RUN pip install -e /workspace/GCALF-Net -e /workspace/picai_prep \
                -e /workspace/picai_eval -e /workspace/M3d-Cam
ENV nnDet_raw=/data/nnDet_raw nnDet_prep=/data/nnDet_prep nnDet_results=/data/nnDet_results
```
Build now, so "works on my machine" == "works in the cloud". Pin every installed package and record the built image digest. Do not modernize PyTorch/Lightning during the GCALF architecture experiment because that would confound the baseline comparison.

The complete provider-neutral VM, S3-compatible storage, local NVMe staging, recovery, and security workflow is in `docs/CLOUD_DEPLOYMENT_PLAN.md`.

## 0.5 Deliverables & commit

- Branch `feat/env-and-data` in `GCALF-Net/`, commit "env: pinned gcalf conda env + gcalf scaffold + CPU encoder smoke".
- Files: `environment.yml`, `env.lock.txt`, `Dockerfile`, `tests/test_encoder_cpu.py`, empty `nndet/arch/encoder/gcalf/{__init__,lff,caf,registry}.py`.

**Next:** `PHASE_1_data_pipeline.md`.
