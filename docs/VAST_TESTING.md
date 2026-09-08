# Four-GPU Vast.ai validation (M0--M6)

This run uses one verified, on-demand Vast.ai instance only. It validates research software and
is not a clinical workflow. Do not commit PI-CAI images, prepared arrays, checkpoints, credentials,
or exports.

## Prerequisites and cost controls

Use four RTX 3090, RTX A5000, RTX A6000, or A100 GPUs with at least 24 GB VRAM each, 32 CPU cores,
128 GB RAM, 500 GB local disk, direct SSH, and reliability at least 0.99. Do not select RTX 4090/Ada: the
pinned CUDA 11.3 extension build targets compute capabilities through 8.6. The required image is
`pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`.

Keep `$VAST_API_KEY` out of shell history and Git. Search is read-only. Creation and destruction
are explicit; destruction additionally requires `--confirm`. Stop the instance when waiting on a
decision or any failed export. A stopped instance can still retain disk-related charges, so resume,
export, and verify promptly.

```bash
vastai set api-key "$VAST_API_KEY"
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/gcalf_vast_ed25519 -C "gcalf-vast"
vastai create ssh-key "$(tr -d '\n' < ~/.ssh/gcalf_vast_ed25519.pub)"
cloud/vast/host.sh search
cloud/vast/host.sh create --offer-id "$OFFER_ID" --disk-gb 500 --label gcalf-m1-m6
cloud/vast/host.sh status --instance-id "$INSTANCE_ID"
vastai ssh-url "$INSTANCE_ID"
```

The pinned Kaggle CLI reads an API token file. In Kaggle account settings, create and download a
new API token (`kaggle.json`), then copy it to the instance without committing or sharing it:

```bash
mkdir -p /root/.config/kaggle
chmod 700 /root/.config/kaggle
# Copy the downloaded kaggle.json into this directory by a secure method.
chmod 600 /root/.config/kaggle/kaggle.json
```

`cloud/vast/host.sh ... --dry-run` prints its command without changing provider state. Record the
instance ID returned by `create`; the helper deliberately does not infer it from another account
operation.

## Bootstrap and milestones

Copy or clone this repository at the intended commit to `/workspace/GCALF-Net`, connect with the
SSH URL, then execute the following inside the requested image:

```bash
cd /workspace/GCALF-Net
GCALF_CONDA_ENV=/workspace/.conda/gcalf bash cloud/vast/bootstrap.sh --repo-dir "$PWD"
export det_data=/workspace/det_data
export det_models=/workspace/det_models
export OMP_NUM_THREADS=8
export det_num_threads=8
GCALF_CONDA_ENV=/workspace/.conda/gcalf bash cloud/vast/run_milestones.sh --repo-dir "$PWD" --workspace /workspace
```

`bootstrap.sh` installs the pinned M0 dependencies, forces the CUDA extension build, and records
the checkout SHA, image reference, pip freeze, CUDA toolchain, driver, GPU model, and hostname in
`/workspace/evidence/m0/`. `run_milestones.sh` runs M0--M3 in sequence and uses `set -euo pipefail`,
so it stops at the first failed command while retaining all preceding logs.

M0 runs `tests/test_imports.py`, `tests/test_encoder_cpu.py`, and `tests/test_csrc_cuda.py`; a
skipped CUDA test is a failure. It also confirms that `nndet` resolves to this checkout and imports
`picai_prep`, `picai_eval`, and `medcam`.

The M1 download helper retrieves the pinned operational mirror
`varshithpsingh/prostate-cancer-pi-cai-dataset/3` from Kaggle. The PI-CAI project remains the
scientific source; this Kaggle dataset is the reproducible transfer source. The helper records the
downloaded archive SHA-256 and the pinned `picai_labels` and `picai_baseline` Git SHAs in
`/workspace/source/SOURCE_REVISIONS.txt`. It uses the all-case official `picai/splits.json` fold
definition, granular expert labels, `marksheet.csv`, and binary Pooch25 masks as detection-only
positives; Pooch25 lesions never supervise the grade head.

M1 writes `/workspace/det_data/Task2201_PICAI_csPCa`, installs official splits, and checks geometry,
modalities, labels, instances, folds, patient separation, plus the three-input/four-foreground plan.
M2 preserves its initial/final loss, duration, loss components, and peak GPU memory in
`evidence/m2/overfit.log`. M3 creates a new `Task900_PICAI_TINY` by default. If it fails, retain the
partial task and model output for diagnosis; retry with `--m3-task Task901_PICAI_TINY` (or another
unused `Task9xx_PICAI_TINY` identifier). No script deletes a partial task.

## Four-GPU training gate and run map

Fold 0 may use all four GPUs only after a recorded DDP gate: compare one fixed FP32 batch against
the single-GPU result (maximum absolute difference <= `1e-5`), then measure 50 warm-up and 200 timed
steps. Require at least `2.5x` throughput and at least 10% free VRAM on every GPU. Store commands,
GPU telemetry, the batch hash, and results under `/workspace/evidence/ddp/`. A failing gate means
all official folds run on one GPU; it is not a license to change the global batch or learning rate.

Use `scripts/run_ddp_gate.py` for this admission gate. It generates a deterministic, non-patient
one-class/grade-masked FP32 batch at the task's planned shape, records only its SHA-256 hash and scalar
loss (not an image array), and times the actual forward/backward/SGD/DDP-all-reduce path. Run the
single-GPU command first, then the four-rank command. Capture `nvidia-smi` before and after each run in
the same evidence directory.

```bash
EVIDENCE=/workspace/evidence/ddp/YYYYMMDD
mkdir -p "$EVIDENCE"
nvidia-smi > "$EVIDENCE/nvidia-smi-before.txt"
CUDA_VISIBLE_DEVICES=0 python scripts/run_ddp_gate.py Task2201_PICAI_csPCa \
  --mode single --evidence "$EVIDENCE" |& tee "$EVIDENCE/single.log"
python -m torch.distributed.run --nproc_per_node=4 scripts/run_ddp_gate.py \
  Task2201_PICAI_csPCa --mode ddp --evidence "$EVIDENCE" |& tee "$EVIDENCE/ddp.log"
nvidia-smi > "$EVIDENCE/nvidia-smi-after.txt"
```

`gate.json` is the recorded decision. Only its all-pass result permits the Fold 0 DDP command below.

The DDP configuration keeps the planner's global batch fixed, splits it evenly across the four
ranks, assigns each rank disjoint train/validation cases and augmenter seeds, gathers evaluator
caches before nonlinear FROC/Dice calculation, and enables unused-parameter detection for the
masked grade head. It refuses a plan whose batch is not divisible by four.

```bash
# Only after the DDP gate passes; replace the task/config with the registered arm.
python scripts/train.py Task2201_PICAI_csPCa \
  -o train=gcalf_ddp exp.fold=0 exp.seed=2026

# Folds 1--4 remain independent one-GPU runs (one process per GPU).
CUDA_VISIBLE_DEVICES=0 python scripts/train.py Task2201_PICAI_csPCa \
  -o train=gcalf_baseline exp.fold=1 exp.seed=2026
```

Freeze this DDP/single-GPU decision and the fold-to-GPU map before beginning official results.

## Export, local verification, and recovery

After all gates pass, package the prepared M1 task, all milestone logs/reports, and the complete M3
model tree (checkpoints, plan, predictions, and metrics):

```bash
bash cloud/vast/export_results.sh \
  --task-dir /workspace/det_data/Task2201_PICAI_csPCa \
  --evidence-dir /workspace/evidence \
  --model-dir /workspace/det_models/Task900_PICAI_TINY/RetinaUNetV001_D3V001_3d \
  --export-dir /workspace/export

vastai copy "$INSTANCE_ID":/workspace/export/ local:./gcalf-vast-export/
cd gcalf-vast-export
sha256sum -c SHA256SUMS
```

The exporter refuses a non-empty destination and creates `SHA256SUMS` only after all three tar
archives exist. Acceptance requires every local checksum to pass; M3 additionally requires
`model_best.ckpt`, `model_last.ckpt`, `plan_inference.pkl`, predictions, and one finite-row
`metrics.csv` (PI-CAI score, AUROC, lesion AP).

If any milestone, export, transfer, or local checksum fails, preserve the evidence and stop rather
than destroy:

```bash
vastai stop instance "$INSTANCE_ID"
```

Only after local `sha256sum -c SHA256SUMS` succeeds may the instance be destroyed:

```bash
cloud/vast/host.sh destroy --instance-id "$INSTANCE_ID" --confirm
```
