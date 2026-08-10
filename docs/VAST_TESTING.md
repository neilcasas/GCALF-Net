# Single-instance Vast.ai validation (M0--M3)

This run uses one verified, on-demand Vast.ai instance only. It validates research software and
is not a clinical workflow. Do not commit PI-CAI images, prepared arrays, checkpoints, credentials,
or exports.

## Prerequisites and cost controls

Use one RTX 3090, RTX A5000, RTX A6000, or A100 with at least 24 GB VRAM, 8 CPU cores, 32 GB RAM,
500 GB local disk, direct SSH, and reliability at least 0.99. Do not select RTX 4090/Ada: the
pinned CUDA 11.3 extension build targets compute capabilities through 8.6. The required image is
`pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`.

Keep `$VAST_API_KEY` out of shell history and Git. Search is read-only. Creation and destruction
are explicit; destruction additionally requires `--confirm`. Stop the instance when waiting on a
decision or any failed export. A stopped instance can still retain disk-related charges, so resume,
export, and verify promptly.

```bash
vastai set api-key "$VAST_API_KEY"
vastai create ssh-key ~/.ssh/id_ed25519.pub
cloud/vast/host.sh search
cloud/vast/host.sh create --offer-id "$OFFER_ID" --disk-gb 500
cloud/vast/host.sh status --instance-id "$INSTANCE_ID"
vastai ssh-url "$INSTANCE_ID"
```

`cloud/vast/host.sh ... --dry-run` prints its command without changing provider state. Record the
instance ID returned by `create`; the helper deliberately does not infer it from another account
operation.

## Bootstrap and milestones

Copy or clone this repository at the intended commit to `/workspace/GCALF-Net`, connect with the
SSH URL, then execute the following inside the requested image:

```bash
cd /workspace/GCALF-Net
bash cloud/vast/bootstrap.sh --repo-dir "$PWD"
export det_data=/workspace/det_data
export det_models=/workspace/det_models
export OMP_NUM_THREADS=8
export det_num_threads=8
bash cloud/vast/run_milestones.sh --repo-dir "$PWD" --workspace /workspace
```

`bootstrap.sh` installs the pinned M0 dependencies, forces the CUDA extension build, and records
the checkout SHA, image reference, pip freeze, CUDA toolchain, driver, GPU model, and hostname in
`/workspace/evidence/m0/`. `run_milestones.sh` runs M0--M3 in sequence and uses `set -euo pipefail`,
so it stops at the first failed command while retaining all preceding logs.

M0 runs `tests/test_imports.py`, `tests/test_encoder_cpu.py`, and `tests/test_csrc_cuda.py`; a
skipped CUDA test is a failure. It also confirms that `nndet` resolves to this checkout and imports
`picai_prep`, `picai_eval`, and `medcam`.

The M1 download helper retrieves exactly the five archives from the original Zenodo record 6517398,
resumes incomplete `curl` downloads, verifies the record's published MD5 checksums before extraction,
and records the actual `picai_labels` and `picai_baseline` Git SHAs in
`/workspace/source/SOURCE_REVISIONS.txt`. It clones and uses only the original granular expert
labels, `marksheet.csv`, and `picai_nnunet/splits.json`; it does not use binary Pooch25 masks.

M1 writes `/workspace/det_data/Task2201_PICAI_GGG`, installs official splits, and checks geometry,
modalities, labels, instances, folds, patient separation, plus the three-input/four-foreground plan.
M2 preserves its initial/final loss, duration, loss components, and peak GPU memory in
`evidence/m2/overfit.log`. M3 creates a new `Task900_PICAI_TINY` by default. If it fails, retain the
partial task and model output for diagnosis; retry with `--m3-task Task901_PICAI_TINY` (or another
unused `Task9xx_PICAI_TINY` identifier). No script deletes a partial task.

## Export, local verification, and recovery

After all gates pass, package the prepared M1 task, all milestone logs/reports, and the complete M3
model tree (checkpoints, plan, predictions, and metrics):

```bash
bash cloud/vast/export_results.sh \
  --task-dir /workspace/det_data/Task2201_PICAI_GGG \
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
