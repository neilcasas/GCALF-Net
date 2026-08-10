#!/usr/bin/env bash
# Run inside the pinned Vast.ai container after the repository has been copied or cloned.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/bootstrap.sh [--repo-dir DIR] [--evidence-dir DIR] [--image-ref REF] [--dry-run]
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
evidence_dir=/workspace/evidence/m0
image_ref=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir) [[ $# -ge 2 ]] || die "--repo-dir requires a value"; repo_dir=$2; shift 2 ;;
        --evidence-dir) [[ $# -ge 2 ]] || die "--evidence-dir requires a value"; evidence_dir=$2; shift 2 ;;
        --image-ref) [[ $# -ge 2 ]] || die "--image-ref requires a value"; image_ref=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -f "$repo_dir/environment.yml" && -f "$repo_dir/requirements-tools.txt" ]] || die "--repo-dir is not this checkout: $repo_dir"

run() {
    if [[ "$dry_run" == true ]]; then
        printf 'dry-run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ "$dry_run" == true ]]; then
    run python -m pip install --upgrade pip
    run python -m pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
        torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113
    run python -m pip install -r "$repo_dir/requirements.txt" -r "$repo_dir/requirements-tools.txt"
    run env FORCE_CUDA=1 python -m pip install -v -e "$repo_dir"
    exit 0
fi

mkdir -p "$evidence_dir"
{
    printf 'repository_sha='; git -C "$repo_dir" rev-parse HEAD
    printf 'base_image=%s\n' "$image_ref"
    printf 'hostname='; hostname
    printf 'python='; python --version
    printf 'pip='; python -m pip --version
    printf 'cuda_toolkit='; nvcc --version || true
    printf 'gpu='; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
} > "$evidence_dir/environment.txt"

python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113
python -m pip install -r "$repo_dir/requirements.txt" -r "$repo_dir/requirements-tools.txt"
FORCE_CUDA=1 python -m pip install -v -e "$repo_dir"
python -m pip freeze | sort > "$evidence_dir/pip-freeze.txt"
python - <<'PY' | tee "$evidence_dir/extension.txt"
import nndet
import nndet._C
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print("nndet=", nndet.__file__)
print("extension=", nndet._C.__file__)
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
PY
