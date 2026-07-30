# AGENTS.md

Operating instructions for coding agents in this repository. Read before every task.

**Working code only. Finish the job. Plausibility is not correctness.**

## 0. Non-negotiables

1. No flattery or filler. Start with the answer or action.
2. Disagree with incorrect premises before doing the work.
3. Never fabricate file paths, commands, APIs, or test results. Read files and run commands to verify them.
4. Stop and ask when a task has materially different plausible interpretations.
5. Touch only what the request requires. Do not include drive-by refactors or formatting changes.

## 1. Before Writing Code

- State a one- or two-sentence plan before editing. For non-trivial work, list steps and a verification check for each.
- Read files to be changed and their direct callers before editing.
- Match the existing repository patterns rather than introducing a new style.
- State assumptions that affect the implementation. Present material alternatives and their tradeoffs before choosing one.

## 2. Writing Code

- Prefer the smallest implementation that meets the stated requirement.
- Do not add single-use abstractions, speculative configuration, or future extension points.
- Handle real failure modes, not hypothetical impossible ones.
- Clean up imports or code made unused by your own edit only.

## 3. Surgical Changes

- Every changed line must trace directly to the request.
- Do not modify adjacent comments, imports, formatting, or dead code unless the task requires it.
- Never revert or overwrite pre-existing worktree changes without explicit user approval.
- Preserve the codebase's Python style and its 120-character pycodestyle limit in `setup.cfg`.

## 4. Goal-Driven Execution

1. Turn the request into verifiable success criteria before coding.
2. Add or update a focused test when practical.
3. Run the most targeted available verification, then a broader relevant check before finalizing.
4. Read failures fully and fix root causes rather than suppressing symptoms.

Do not report completion based on a plausible diff. State exactly what verification ran and any verification that could not run.

## 5. Repository Context

### Stack

- Python package: `nndet`, a PyTorch-based fork of nnDetection for 3D prostate lesion detection and Gleason Grade Group classification.
- Packaging: setuptools in `setup.py`; Python `>=3.8` is required by package metadata.
- Native code: `nndet/csrc` is compiled through PyTorch C++/CUDA extensions during installation.
- Runtime: Linux, PyTorch, and CUDA for GPU training. The Docker image is based on `nvcr.io/nvidia/pytorch:21.11-py3`.
- Configuration: Hydra; training and model configuration is under `nndet/conf`.
- Data and model roots: `det_data` and `det_models` environment variables.

### Layout

- Core package: `nndet/`
- CLI entry points and workflows: `scripts/`
- Import smoke tests: `tests/test_imports.py`
- Project configuration: `setup.py`, `setup.cfg`, and `requirements.txt`
- Container build: `Dockerfile`
- Research and planning documents: `docs/`

### Commands

- Install dependencies after installing a compatible PyTorch build: `pip install -r requirements.txt`
- Install the package and build its extension: `pip install -e .`
- Build with CUDA explicitly enabled when the CUDA toolchain is available: `FORCE_CUDA=1 pip install -v -e .`
- Build the container image: `docker build -t nndet .`
- Run the import smoke tests when pytest is available: `python -m pytest tests/test_imports.py`
- Preprocess a task: `nndet_prep Task022_Prostate`
- Train: `nndet_train 022 RetinaUNetV001_D3V001_3d 0 --fold 0`
- Consolidate folds: `nndet_consolidate Task022_Prostate RetinaUNetV001_D3V001_3d --fold 0 1 2 3 4`
- Predict: `nndet_predict Task022_Prostate RetinaUNetV001_D3V001_3d --fold consolidated`
- Evaluate: `nndet_eval Task022_Prostate RetinaUNetV001_D3V001_3d --fold consolidated`

Do not claim CUDA builds, model training, or tests passed unless their commands completed successfully in the current environment. GPU execution requires compatible drivers, CUDA, and PyTorch; do not assume they are available.

### Project Constraints

- This is research software for prostate cancer lesion detection. Do not represent the model as clinically approved or suitable for clinical decision-making.
- Never commit patient data, raw imaging, generated model checkpoints, or experiment artifacts. `det_data/`, `det_models/`, `*.npy`, `*.npz`, and `*.ckpt` are intentionally ignored.
- Preserve the `det_data` and `det_models` configuration contract. Do not hard-code local data or model paths.
- Treat label semantics and grade mappings as scientific behavior. Do not change them without an explicit request and an accompanying validation plan.
- Do not silently drop cases, predictions, labels, or evaluation failures in preprocessing, training, prediction, or evaluation code.
- Keep CPU and CUDA extension behavior aligned when changing code that crosses `nndet/csrc` boundaries.
- Large training and evaluation runs are not routine verification. Use focused unit or import checks unless the request specifically requires a full experiment.

## 6. Tool Use and Verification

- Prefer running targeted checks over guessing.
- Read the complete error output before modifying code to fix it.
- For training or evaluation changes, verify configuration parsing and the smallest relevant execution path before requesting a GPU-scale run.
- For documentation-only changes, inspect the final rendered Markdown and verify referenced paths and commands against the repository.
- Do not use destructive git commands such as `git reset --hard` or `git checkout --` unless explicitly approved.

## 7. Communication

- Be direct and concise. State risks and uncertainty plainly.
- For code review requests, list findings first, ordered by severity, with file and line references. If there are no findings, say so and name any remaining test gap.
- Final responses must summarize changes, verification run, and blockers or unverified work.

## 8. When to Ask

Ask before proceeding when the requested change alters label definitions, model output semantics, data format, evaluation criteria, compatibility with existing checkpoints, or requires unavailable data, secrets, or GPU resources. Proceed after inspecting the code when ambiguity is resolvable from repository context.

## 9. Project Learnings

- Never normalize lesion grade labels, modality order, or case identifiers without explicit approval.
- Do not assume a local CUDA installation is available merely because the project supports GPU builds.
