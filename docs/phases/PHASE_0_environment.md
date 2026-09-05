# Phase 0 — Environment & Repository Setup

**Milestone:** M0 · **Depends on:** nothing · **Blocks:** everything
**Goal:** two environments (`ARCHITECTURE.md §12`) — the pinned `gcalf:m0` image where every
reported result must reproduce, and a disposable modern-CUDA scratch environment for LFF/CAF math
prototyping on the local GPU. They never merge: nothing built in the scratch environment is
installed into `GCALF-Net/`'s dependencies or reported as a thesis result.

**Working repository:** `GCALF-Net/` — a copy of `PDHD-Net/` at tag `pdhd-upstream` (commit
`e2330cf`), upstream remote removed. `git diff pdhd-upstream` is the thesis's exact changeset.
`PDHD-Net/` stays pristine; install only one of the two (`nndet` package name collides).

**Definition of done (exit criteria):**
- [ ] **L:** `import nndet, nndet._C` succeeds and resolves to `GCALF-Net/nndet`, not `PDHD-Net/nndet`.
- [ ] **L:** `python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py` passes.
- [ ] **L:** `import picai_prep, picai_eval, medcam` succeeds.
- [ ] **L:** a random-tensor forward pass through `nndet.arch.encoder.modular.Encoder` runs on CPU.
- [ ] **V:** `tests/test_csrc_cuda.py` **runs** (not skip) in the GPU container. This has never
      passed — `docs/m0-verification.md` records the Docker/NVIDIA-CDI failure that blocked it.
      Re-attempt on the actual Vast.ai host before trusting any later CUDA gate.
- [ ] `nndet/arch/encoder/gcalf/` scaffolding (`fdr.py`, `waf.py`, `lff.py`, `caf.py`,
      `grade_head.py`, `registry.py`) committed as empty modules on branch `feat/env-and-data`.
- [ ] `environment.yml` + a sorted, non-editable `pip freeze` lock committed;
      `docs/m0-verification.md` updated with the new attempt's outcome.

---

## 0.1 The pinned environment is dictated by PDHD-Net, not chosen

`requirements.txt` pins a 2021 stack: `pytorch_lightning>=1.3.1,<=1.4.2`, `nnunet==1.7.1`,
`SimpleITK<2.1.0`, `torchmetrics<=0.7.3`. Do not fight these — they anchor:

| Component | Version |
|---|---|
| Python | 3.8 (authoritative Docker image) |
| PyTorch / torchvision / torchaudio | 1.10.1 / 0.11.2 / 0.10.1 (CUDA 11.3) |
| CUDA | 11.1 or 11.3 (must match the torch build) |
| pytorch-lightning | 1.4.2 |
| nnunet | 1.7.1 |

GFNet/TransFuse/UCTransNet/DCA are **never installed** — copy classes out of them. Z-SSMNet is
**never installed** — read-only reference.

## 0.2 Why CPU-only locally, and what the scratch environment is for

The local RTX 4050 is `sm_89`. The pinned CUDA 11.3 toolchain builds through `sm_86`
(`docs/VAST_TESTING.md`), and the card has 6 GB regardless of compute capability. Consequences:

- **Every gate whose result must match what gets reported** runs inside `gcalf:m0`, CPU-only:
  unit tests, config/registry assertions, manifest/fold validation, real preprocessing on fold-0
  cases, synthetic CPU forward/backward, the 2-case CPU micro-overfit (Phase 2). These are the
  **L** gates throughout the roadmap.
- **Everything CUDA-dependent** — the `csrc` extension build, `test_csrc_cuda.py`, FDR/WAF/LFF/CAF
  memory profiling, real training — is a **V** gate, run on Vast.ai. Do not attempt to satisfy a
  V gate locally; it cannot build against this GPU's compute capability.
- A **separate, disposable modern-torch/CUDA conda env** (e.g. `conda create -n gcalf-scratch
  python=3.11` + a current `torch`/`torch.fft` build) may run on the local 4050 to iterate on
  LFF/CAF tensor math — shape, gradient flow, the orientation-gate logic — before porting the
  validated implementation into the pinned stack by hand. Never `pip install -e` `GCALF-Net/`
  into this environment, never commit its lockfile, and never cite a result produced there in the
  thesis. It exists purely to make the edit-test loop faster than round-tripping through Docker
  for pure-math bugs that have nothing to do with the pinned stack.

## 0.3 Step-by-step (pinned environment)

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
5. **Build and verify the authoritative image on a GPU host (Vast.ai):**
   ```bash
   docker build -t gcalf:m0 .
   docker run --rm --gpus all gcalf:m0 python -m pytest -q tests
   ```
   The CUDA test must run, not skip; assert `torch.cuda.is_available()` first inside the
   container. **This step has never succeeded** (`docs/m0-verification.md`) — treat it as an
   open blocker, not a formality, until it does.
6. **Scaffold the gcalf code tree** (empty modules + `__init__.py` for `fdr.py`, `waf.py`,
   `lff.py`, `caf.py`, `grade_head.py`, `registry.py` — see `ARCHITECTURE.md §2`). Commit.

## 0.4 nnDetection csrc build failure — the top risk

| Symptom | Fix |
|---|---|
| `nvcc` not found | Install matching `cudatoolkit-dev` in conda, or use `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`. |
| ABI / torch version mismatch on import | Rebuild after installing the exact pinned torch: `pip install -e GCALF-Net/ --no-build-isolation`. |
| No local CUDA at all | Expected here (§0.2) — run only the CPU import/encoder smoke locally; the authoritative build happens on the GPU host. M0 stays open until the CUDA extension test runs there. |
| Docker CDI/GPU-vendor discovery fails on the build host | Record the exact error in `docs/m0-verification.md` (as already done once); this blocks M0's V gate — escalate rather than declaring M0 done without it. |

## 0.5 Docker (build once, reuse local + cloud)

The committed `Dockerfile` is the M0 runtime:
```dockerfile
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
    torchaudio==0.10.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
ENV det_data=/opt/data det_models=/opt/models
```
Generate `env.lock.txt` from the completed image; record base/image digests in
`docs/m0-verification.md`. Do not modernize PyTorch/Lightning during the GCALF comparison — it
would confound baseline-vs-GCALF attribution. The complete provider-neutral VM, S3 storage, NVMe
staging, recovery, and security workflow is in `docs/CLOUD_DEPLOYMENT_PLAN.md`.

## 0.6 Deliverables & commit

- Branch `feat/env-and-data`, commit "env: pinned gcalf conda env + gcalf scaffold + CPU encoder smoke".
- Files: `environment.yml`, `requirements-tools.txt`, `env.lock.txt`, `Dockerfile`,
  `tests/test_{imports,encoder_cpu,csrc_cuda}.py`, empty
  `nndet/arch/encoder/gcalf/{__init__,fdr,waf,lff,caf,grade_head,registry}.py`.

**Next:** `PHASE_1_data_pipeline.md`.
