# GCALF-Net Cloud Deployment Plan

**Status:** implementation plan | **Target:** provider-neutral NVIDIA GPU VM | **Storage:** S3-compatible object storage | **Runtime:** pinned Docker image

This document defines a reproducible path from an object-storage dataset to GCALF-Net training, evaluation, and durable artifacts. The normal training path stages a validated, preprocessed nnDetection task onto local NVMe. It does not train through an object-store mount.

## 1. Decisions and Non-Goals

| Decision | Selected approach |
|---|---|
| Compute | Any Linux GPU VM with NVIDIA Container Toolkit and sufficient local NVMe |
| Runtime | Frozen PyTorch 1.10/CUDA 11.3-era container for baseline fidelity |
| Storage API | AWS CLI v2 `s3` commands with optional `--endpoint-url` |
| Training input | Immutable preprocessed nnDetection dataset staged to local NVMe |
| Raw preprocessing | Supported as a separate one-time job |
| Recovery | Durable checkpoint, logs, and run state after every epoch |
| Required matrix | baseline, LFF-only, CAF-only, full GCALF x five official folds |
| Secrets | VM workload identity when available; otherwise short-lived injected credentials |

Non-goals: managed-provider SDK integration, Kubernetes, Terraform, training directly from S3, and dependency modernization.

## 2. Planned Repository Additions

```text
thesis/
|-- Dockerfile.gcalf
|-- .dockerignore
|-- GCALF-Net/
|   |-- cloud/
|   |   |-- env.example
|   |   |-- stage_dataset.sh
|   |   |-- verify_manifest.py
|   |   |-- sync_run.sh
|   |   |-- run_training.sh
|   |   |-- run_preprocessing.sh
|   |   |-- launch_one.sh
|   |   `-- run_matrix.sh
|   |-- gcalf_configs/
|   |   |-- baseline.yaml
|   |   |-- lff_only.yaml
|   |   |-- caf_only.yaml
|   |   `-- gcalf_full.yaml
|   `-- tests/cloud/
|       `-- test_manifest.py
```

Two things deliberately *not* built: `bootstrap_vm.sh` (its contents are six one-line prerequisite checks — they live as a copy-pasteable block in §7, not as a script to maintain) and `test_shell_contracts.py` (tests for shell scripts that a single operator runs by hand; `verify_manifest.py` is the piece that guards real data, so it is the piece that gets tests). `run_matrix.sh` here is the *only* matrix runner — `PHASE_5` refers to this one rather than defining a second.

Shell scripts must use `set -euo pipefail`, quote paths, avoid printing credentials, and support `--dryrun` where a transfer can delete or overwrite data.

## 3. Object-Storage Layout

Use distinct immutable input and append-only output prefixes:

```text
s3://<bucket>/gcalf/
|-- datasets/
|   |-- raw/picai/<raw-version>/...
|   `-- prepared/<dataset-version>/
|       |-- nnDet_raw/Task9xx_PICAI/...
|       |-- nnDet_prep/Task9xx_PICAI/...
|       |-- splits_final.pkl
|       |-- preprocessing-config.json
|       `-- manifest.json
|-- images/<image-version>/image-digest.txt
`-- runs/<run-id>/
    |-- run.json
    |-- config.yaml
    |-- checkpoints/
    |-- logs/
    |-- predictions/
    |-- metrics/
    `-- COMPLETE
```

Rules:

- A prepared dataset prefix is immutable after `manifest.json` is published.
- `dataset-version` identifies content and preprocessing, not a mutable label such as `latest`.
- Every run writes to a unique `run-id`: `<config>-fold<0..4>-seed<seed>-<UTC timestamp>-<git short SHA>`.
- The `COMPLETE` marker is uploaded only after training, prediction, evaluation, and final synchronization succeed.
- Bucket versioning and lifecycle rules are recommended. Do not expire the newest valid checkpoints or thesis-final artifacts.

## 4. Immutable Dataset Manifest

The manifest is the trust boundary between preprocessing and training. Suggested schema:

```json
{
  "schema_version": 1,
  "dataset_version": "picai-nndet-v1",
  "created_at": "2026-07-26T00:00:00Z",
  "task": "Task9xx_PICAI",
  "modalities": {"0000": "T2W", "0001": "ADC", "0002": "DWI"},
  "preprocessing_config_sha256": "<hex>",
  "splits_sha256": "<hex>",
  "case_count": 1500,
  "files": [
    {"path": "nnDet_prep/Task9xx_PICAI/...", "size": 123, "sha256": "<hex>"}
  ]
}
```

`cloud/verify_manifest.py` must:

1. Reject unknown schema versions and absolute/traversal paths.
2. Verify every listed file exists under the staging root.
3. Compare exact file size and SHA-256.
4. Reject unlisted data files unless explicitly allowed.
5. Verify task name, modality mapping, case count, preprocessing config hash, and split hash.
6. Write `verified-manifest.sha256` only after all checks pass.

Initial verification core:

```python
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


for entry in manifest["files"]:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe manifest path: {}".format(relative))
    candidate = staging_root / relative
    if candidate.stat().st_size != entry["size"]:
        raise ValueError("size mismatch: {}".format(relative))
    if sha256_file(candidate) != entry["sha256"]:
        raise ValueError("checksum mismatch: {}".format(relative))
```

The AWS CLI also validates supported S3 checksums on transfer, but the repository manifest remains required because it binds all objects, splits, and preprocessing settings into one dataset version.

## 5. Environment Contract

`cloud/env.example` documents names only and contains no values:

```bash
S3_BUCKET=
S3_ENDPOINT_URL=
S3_REGION=
DATASET_VERSION=
RAW_DATASET_VERSION=
IMAGE_REF=
IMAGE_DIGEST=
TASK_NAME=Task9xx_PICAI
LOCAL_ROOT=/local/gcalf
RUN_CONFIG=
RUN_FOLD=
RUN_SEED=2026
RUN_ID=
```

Credential variables such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` may be injected at runtime but must never appear in `env.example`, image layers, run metadata, shell tracing, or Git.

Use one helper array in scripts so an empty endpoint works with AWS and a populated endpoint works with compatible services:

```bash
AWS_ARGS=(--region "${S3_REGION}")
if [[ -n "${S3_ENDPOINT_URL:-}" ]]; then
  AWS_ARGS+=(--endpoint-url "${S3_ENDPOINT_URL}")
fi
```

## 6. Container Image

Build once in a controlled environment and run the same digest locally and in cloud:

```dockerfile
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

ARG AWS_CLI_VERSION=2.27.54
ARG AWS_CLI_SHA256=<verified-release-sha256>
RUN curl -fsSLo /tmp/awscliv2.zip \
      "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWS_CLI_VERSION}.zip" \
    && echo "${AWS_CLI_SHA256}  /tmp/awscliv2.zip" | sha256sum -c - \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

WORKDIR /workspace
COPY GCALF-Net /workspace/GCALF-Net
COPY picai_prep /workspace/picai_prep
COPY picai_eval /workspace/picai_eval
COPY M3d-Cam /workspace/M3d-Cam

RUN pip install --no-cache-dir \
      -e /workspace/GCALF-Net \
      -e /workspace/picai_prep \
      -e /workspace/picai_eval \
      -e /workspace/M3d-Cam \
    && python -c "import nndet, picai_prep, picai_eval, medcam"

WORKDIR /workspace/GCALF-Net
ENTRYPOINT ["/bin/bash", "cloud/run_training.sh"]
```

Build from the workspace root so sibling reference/tool repositories are available in the context:

```bash
docker build -f Dockerfile.gcalf -t "${IMAGE_REGISTRY}/gcalf:${GIT_SHA}" .
docker push "${IMAGE_REGISTRY}/gcalf:${GIT_SHA}"
docker image inspect "${IMAGE_REGISTRY}/gcalf:${GIT_SHA}" --format '{{json .RepoDigests}}'
```

Implementation requirements:

- Pin the base by digest and pin all Python dependencies in a lock file.
- Copy **GCALF-Net only**. `PDHD-Net/` provides the same `nndet` package name; shipping both makes the effective code depend on import path order.
- Replace the AWS CLI checksum placeholder with the checksum verified from the official release artifact before building.
- Build nnDetection CUDA extensions against the image's exact PyTorch/CUDA versions.
- Run imports and CPU tests during image build; run GPU extension and forward tests on a GPU VM.
- Do not put datasets, credentials, checkpoints, or `.git` histories into the image.
- Push the image to any OCI registry and record the immutable digest in each `run.json`.
- Generate an SBOM and retain the build log for thesis reproducibility.

## 7. GPU VM Requirements and Bootstrap

Minimum target for the first profile:

- Linux x86_64 host compatible with NVIDIA Container Toolkit.
- NVIDIA GPU with at least 16 GB VRAM; 24 GB is preferred until stage-2 CAF profiling is complete.
- Local NVMe capacity of at least `2 x prepared_dataset_size + 100 GB` for staging, temporary files, and outputs.
- System disk separate from dataset scratch where possible.
- Outbound HTTPS to the object store and container registry.
- No public inbound ports except a restricted administrative path; training needs no public listener.

Verify these prerequisites on a fresh VM before downloading anything. Run them by hand or paste them into the top of `run_training.sh`; they do not need a dedicated script. Verify rather than silently install provider-specific drivers:

```bash
nvidia-smi
docker version
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
test -d "${LOCAL_ROOT}"
df -h "${LOCAL_ROOT}"
aws --version
```

Fail before downloading data if the GPU, driver, free disk, credentials, registry access, or storage endpoint is unavailable.

## 8. Stage and Verify Prepared Data

`cloud/stage_dataset.sh` workflow:

```bash
dataset_uri="s3://${S3_BUCKET}/gcalf/datasets/prepared/${DATASET_VERSION}"
staging_dir="${LOCAL_ROOT}/datasets/${DATASET_VERSION}.partial"
final_dir="${LOCAL_ROOT}/datasets/${DATASET_VERSION}"

mkdir -p "${staging_dir}"
if [[ -e "${final_dir}" ]]; then
  echo "Refusing to replace existing dataset version: ${final_dir}" >&2
  exit 1
fi
aws "${AWS_ARGS[@]}" s3 sync \
  "${dataset_uri}/" "${staging_dir}/" \
  --no-progress --only-show-errors

python cloud/verify_manifest.py \
  --root "${staging_dir}" \
  --manifest "${staging_dir}/manifest.json"

mv -T "${staging_dir}" "${final_dir}"
```

Required hardening:

- Download into a `.partial` directory and atomically rename only after verification.
- Refuse to overwrite an existing verified version with different content.
- Check free disk against manifest total bytes before transfer.
- Keep prepared data read-only inside the training container.
- Do not use `aws s3 sync --delete` on dataset inputs.
- Log transfer duration and bytes, but not object contents or credentials.

AWS CLI `s3 sync` supports `--endpoint-url`, include/exclude filters, and checksum options. Use a CA bundle for private endpoints; never normalize certificate problems by adding `--no-verify-ssl`.

## 9. Optional One-Time Cloud Preprocessing

Normal training starts from a prepared version. If only raw PI-CAI data is available:

1. Launch a storage-optimized VM; a GPU is not required for most conversion steps.
2. Stage `datasets/raw/picai/<raw-version>` to local scratch and verify its source manifest.
3. Run `dcm2mha`, `mha2nnunet`, GGG label construction, `nnunet2nndet`, and `nndet_prep` as specified in `phases/PHASE_1_data_pipeline.md`.
4. Run modality, affine, spacing, label, patient-disjointness, and fold checks.
5. Generate the prepared manifest and its SHA-256.
6. Upload to a new `.partial` prefix.
7. Verify a clean re-download sample and then publish to the immutable final version prefix.

Do not preprocess independently for each training VM. Every required run must use the same prepared dataset and split hashes.

## 10. Run Initialization

Before training, create `${LOCAL_ROOT}/runs/${RUN_ID}` and write `run.json`:

```json
{
  "run_id": "gcalf-full-fold0-seed2026-<timestamp>-<sha>",
  "status": "initializing",
  "config": "gcalf_full",
  "fold": 0,
  "seed": 2026,
  "dataset_version": "picai-nndet-v1",
  "dataset_manifest_sha256": "<hex>",
  "splits_sha256": "<hex>",
  "git_commit": "<sha>",
  "image_digest": "sha256:<hex>",
  "command": ["python", "scripts/train.py", "Task9xx_PICAI", "-o", "..."],
  "started_at": "<UTC>"
}
```

Also snapshot the resolved Hydra config, `pip freeze`, GPU model, driver, CUDA version, hostname/provider metadata if available, and exact stage-memory profile. Upload initialization metadata before starting the first epoch.

## 11. Container Launch

Example host command:

```bash
docker run --rm --gpus all --ipc=host \
  --name "gcalf-${RUN_ID}" \
  --env-file "/secure/path/gcalf.env" \
  -v "${LOCAL_ROOT}/datasets/${DATASET_VERSION}/nnDet_raw:/data/nnDet_raw:ro" \
  -v "${LOCAL_ROOT}/datasets/${DATASET_VERSION}/nnDet_prep:/data/nnDet_prep:ro" \
  -v "${LOCAL_ROOT}/runs/${RUN_ID}:/data/nnDet_results:rw" \
  "${IMAGE_REF}@${IMAGE_DIGEST}"
```

Set `det_data=/data/nnDet_raw` and `det_models=/data/nnDet_results` inside the container; this fork does not read `nnDet_raw`, `nnDet_prep`, or `nnDet_results` as environment variables. Avoid mounting the Docker socket. Use `--shm-size` instead of `--ipc=host` if the environment requires stricter isolation and profiling confirms it is sufficient.

## 12. Checkpoint and Artifact Synchronization

The recovery objective is at most one lost epoch. Integrate an epoch-end callback or wrapper that:

1. Saves `last.ckpt` locally using an atomic temporary file and rename.
2. Updates logs and `run.json` with completed epoch/global step.
3. Uploads checkpoints, logs, config, and run state to the unique run prefix.
4. Confirms the AWS CLI process succeeded before the next epoch is considered durable.

Initial sync command:

```bash
run_uri="s3://${S3_BUCKET}/gcalf/runs/${RUN_ID}"
aws "${AWS_ARGS[@]}" s3 sync \
  "${LOCAL_ROOT}/runs/${RUN_ID}/" "${run_uri}/" \
  --exclude "predictions/tmp/*" \
  --no-progress --only-show-errors
```

Do not use `--delete` during incremental synchronization. Upload a final `COMPLETE` marker only after all required artifacts exist. A failed upload must make run status `sync_failed`, not `completed`.

## 13. Resume After Preemption

`cloud/run_training.sh` resume algorithm:

1. Recreate or reuse a VM and verify runtime prerequisites.
2. Stage and verify the exact dataset version.
3. Pull the same image digest.
4. Sync the existing run prefix into a local `.partial` run directory.
5. Validate `run.json`, config hash, dataset hash, split hash, git SHA, and image digest.
6. Find the newest checkpoint that can be deserialized; do not trust filename ordering alone.
7. Invoke nnDetection in resume mode using its expected existing output directory.
8. Append a resume event with old/new host and UTC timestamp.

If any immutable identifier differs, fail and create a new run instead of resuming incompatible state.

## 14. Required 20-Run Matrix

```text
configs = [baseline, lff_only, caf_only, gcalf_full]
folds   = [0, 1, 2, 3, 4]
seed    = 2026
```

**Budget, stated before launch.** The shipped schedule (`nndet/conf/train/v001.yaml`: 50 epochs × 2500 batches + 10 SWA epochs) is ≈150k optimizer steps per run — roughly 11–21 h on a single 16–24 GB GPU depending on measured seconds/step, so ~10–20 GPU-days for the full matrix. The two fold-0 pilots exist to replace that range with a measurement. As soon as they finish, pick one of the three pre-committed options in `SPEC.md §12` (full matrix / halved batches-per-epoch across all runs / 14-run reduced matrix) and record the choice in every subsequent `run.json`. Choosing after seeing fold results is a form of test-set tuning.

Scheduling rules:

- Tiny smoke tests must pass on the final image before any full run.
- Baseline fold 0 and full-GCALF fold 0 run first to validate cost and memory assumptions.
- Independent fold/config jobs may run concurrently, but each has a unique output prefix.
- A scheduler ledger records `pending`, `running`, `sync_failed`, `failed`, or `complete` and prevents duplicate launches.
- Retry preemption with the same run ID; retry an incompatible code/config change with a new run ID.
- Apply budget tags/labels and an automatic VM shutdown after container exit.

Example matrix loop to be implemented by `cloud/run_matrix.sh`:

```bash
for config in baseline lff_only caf_only gcalf_full; do
  for fold in 0 1 2 3 4; do
    cloud/launch_one.sh --config "${config}" --fold "${fold}" --seed 2026
  done
done
```

`launch_one.sh` is an interface boundary: a provider adapter may create a VM, or an operator may invoke it on an existing VM. Dataset, image, run metadata, and training commands remain provider-neutral.

## 15. Evaluation and Collection

After each fold:

1. Run prediction using the fold's best frozen checkpoint.
2. Compute lesion detection, per-lesion GGG classification, and segmentation metrics.
3. Upload predictions and metrics under the run prefix.
4. Mark the run complete.

After all 20 runs, `gcalf_eval/collect_results.py` downloads only manifests and metric files, verifies that all config/fold pairs share dataset/split/image protocol identifiers, and produces fold-level and aggregate tables. Large prediction maps remain in object storage unless needed for audit.

## 16. Security and Data Governance

- Keep buckets private and block anonymous/public access.
- Prefer a VM role/workload identity scoped to the exact dataset-read and run-write prefixes.
- If static-compatible credentials are unavoidable, use short-lived credentials delivered outside the image and repository.
- Separate read-only dataset permissions from write-only/append run permissions where the provider supports it.
- Encrypt data in transit with verified TLS and at rest with provider-managed or institution-managed keys.
- Restrict administrative network access and disable public services on the training container.
- Do not log images, DICOM metadata, credentials, signed URLs, or patient/study identifiers beyond the approved de-identified experiment key.
- Review Grad-CAM exports before sharing; derived images remain governed dataset artifacts.
- Set retention and deletion policies consistent with PI-CAI terms and institutional requirements.
- Record access, object changes, image digests, and run state for audit.

Minimum conceptual permission split:

```text
dataset reader: list prepared-version prefix, get prepared-version objects
run writer:     list one run prefix, get/put objects in one run prefix
image reader:   pull the approved image digest
```

Do not grant bucket-wide deletion to a training job.

## 17. Observability and Cost Controls

Capture per run:

- GPU utilization, allocated/reserved memory, temperature, and OOM events.
- CPU, RAM, disk usage, disk throughput, and free-space alarms.
- Epoch duration, data-loader wait, transfer duration, and bytes transferred.
- Checkpoint-sync success and age of newest durable checkpoint.
- VM type, region, pricing mode, start/end timestamps, and estimated cost.

Cost controls:

- Stage once per VM and reuse the verified dataset for sequential jobs.
- Use preemptible/spot instances only after resume is proven.
- Shut down automatically after success or terminal failure.
- Keep checkpoints needed for resume and final analysis; lifecycle redundant intermediates after approval.
- Profile before parallelizing the 20-run matrix.

## 18. Validation Gates

| Gate | Required evidence |
|---|---|
| Image | Imports, CUDA extension load, unit tests, image digest recorded |
| Storage | Endpoint access works without public credentials; dry-run paths are correct |
| Dataset | Full manifest, modality, split, and free-space checks pass |
| GPU | Encoder forward/backward and stage-2/stage-5 CAF memory profile pass |
| Tiny train | Two epochs, checkpoint upload, forced stop, resume, predict, evaluate |
| Recovery | Replacement VM resumes same run with at most one epoch lost |
| Baseline | Fold-0 output is consistent with local baseline within declared tolerance |
| Budget | Fold-0 pilots report measured seconds/step, wall time, and cost; one of the three `SPEC.md §12` matrix options is selected and recorded |
| Matrix | Every required config/fold run under the selected option (20, or 14 if reduced) has a complete manifest |
| Collection | All metric rows share dataset/split/protocol identifiers |

No full-fold job launches until the forced-preemption recovery test passes.

## 19. Failure Runbook

| Symptom | Action |
|---|---|
| Dataset checksum mismatch | Delete only the local partial staging directory, re-download, and investigate source mutation if it repeats. |
| Local disk fills | Stop before corrupting outputs; provision larger scratch or remove only verified disposable cache. |
| CUDA extension import fails | Confirm exact image digest, host driver compatibility, and build log; do not patch the running container. |
| CAF OOM | Apply the architecture fallback ladder and create a new config/run identity. |
| Checkpoint upload fails | Keep VM alive, mark `sync_failed`, retry with backoff, and never report completion. |
| VM is preempted | Recreate, verify immutable IDs, download newest durable checkpoint, resume same run ID. |
| Credentials leak to logs | Revoke immediately, preserve audit evidence, rotate credentials, and scrub unauthorized artifacts. |

## 20. Implementation Order

1. Freeze dependency lock and build digest-pinned image.
2. Define manifest schema and unit-test verifier.
3. Implement endpoint-aware stage/sync scripts with a local S3-compatible test service.
4. Publish one tiny prepared dataset version.
5. Run cloud GPU import and model smoke tests.
6. Prove epoch sync and forced-preemption recovery.
7. Run baseline and full-GCALF fold 0 cost/memory pilots.
8. Launch the remaining required matrix.
9. Collect metrics and archive final provenance.

## 21. AWS CLI References

- `aws s3 sync`: <https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html>
- CLI endpoint configuration and `--endpoint-url`: <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-endpoints.html>
- S3 checksum behavior: <https://docs.aws.amazon.com/cli/latest/topic/s3-faq.html>

AWS CLI v2 high-level S3 commands support transfer checksum validation when object checksum metadata is available. The GCALF manifest adds repository-controlled SHA-256 verification and binds the complete dataset version, preprocessing configuration, and split file.
