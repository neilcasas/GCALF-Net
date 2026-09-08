#!/usr/bin/env bash
# Download the version-pinned Kaggle PI-CAI image mirror and pin companion repositories.
set -euo pipefail

DATASET_DEFAULT=varshithpsingh/prostate-cancer-pi-cai-dataset/3
LABELS_COMMIT=ce4a4723d7c46d882a6cbaacb40ed6c4be86282f
BASELINE_COMMIT=2f31d8e7fec4c7ec26729edb1cb7057834bac116

usage() {
    cat <<'EOF'
Usage: cloud/vast/download_picai.sh [options]

Options:
  --source-dir DIR      Parent for the Kaggle archive, images, and source repositories
                        (default: /workspace/source)
  --dataset-ref REF     Version-pinned Kaggle dataset reference
                        (default: varshithpsingh/prostate-cancer-pi-cai-dataset/3)
  --verify-only         Verify an existing archive and extracted image layout only
  --dry-run             Print commands without downloading or writing files
EOF
}

die() { echo "error: $*" >&2; exit 2; }

source_dir=/workspace/source
dataset_ref=$DATASET_DEFAULT
verify_only=false
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir) [[ $# -ge 2 ]] || die "--source-dir requires a value"; source_dir=$2; shift 2 ;;
        --dataset-ref) [[ $# -ge 2 ]] || die "--dataset-ref requires a value"; dataset_ref=$2; shift 2 ;;
        --verify-only) verify_only=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$dataset_ref" == "$DATASET_DEFAULT" ]] || die "dataset reference must be pinned to $DATASET_DEFAULT"

run() {
    if [[ "$dry_run" == true ]]; then
        printf 'dry-run:'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

archive_dir=$source_dir/archives
archive=$archive_dir/prostate-cancer-pi-cai-dataset.zip
images_dir=$source_dir/picai_public_images
labels_dir=$source_dir/picai_labels
baseline_dir=$source_dir/picai_baseline

if [[ "$verify_only" == false ]]; then
    if [[ "$dry_run" == false ]]; then
        command -v kaggle >/dev/null || die "Kaggle CLI is required; create a Kaggle API token and install kaggle.json at /root/.config/kaggle/kaggle.json (mode 600)"
        command -v unzip >/dev/null || die "unzip is required to extract the Kaggle archive"
    fi
    for target in "$labels_dir" "$baseline_dir"; do
        [[ ! -e "$target" ]] || die "refusing to overwrite existing output: $target"
    done
    [[ ! -e "$images_dir" || -z "$(find "$images_dir" -mindepth 1 -print -quit)" ]] || die "refusing to overwrite existing output: $images_dir"
    run mkdir -p "$archive_dir" "$images_dir"
    run kaggle datasets files "$dataset_ref" --csv
    if [[ ! -e "$archive" ]]; then
        run kaggle datasets download "$dataset_ref" --path "$archive_dir"
    fi
fi

if [[ "$dry_run" == true ]]; then
    run sha256sum "$archive"
    run unzip -q "$archive" -d "$images_dir"
    exit 0
fi

[[ -f "$archive" ]] || die "Kaggle archive is missing: $archive"
if [[ "$verify_only" == false ]]; then
    unzip -q "$archive" -d "$images_dir"
fi
for fold in 0 1 2 3 4; do
    [[ -d "$images_dir/picai_public_images_fold$fold" ]] || die "missing fold directory: picai_public_images_fold$fold"
done
for suffix in t2w adc hbv; do
    modality_count=$(find "$images_dir" -type f -name "*_${suffix}.mha" | wc -l | tr -d ' ')
    [[ "$modality_count" == 1500 ]] || die "expected 1,500 $suffix cases, found $modality_count"
done

[[ "$verify_only" == false ]] || exit 0
git clone https://github.com/DIAGNijmegen/picai_labels.git "$labels_dir"
git -C "$labels_dir" checkout --detach "$LABELS_COMMIT"
git clone https://github.com/DIAGNijmegen/picai_baseline.git "$baseline_dir"
git -C "$baseline_dir" checkout --detach "$BASELINE_COMMIT"
{
    printf 'kaggle_dataset_ref=%s\n' "$dataset_ref"
    printf 'kaggle_archive_sha256='; sha256sum "$archive" | awk '{print $1}'
    printf 'picai_labels='; git -C "$labels_dir" rev-parse HEAD
    printf 'picai_baseline='; git -C "$baseline_dir" rev-parse HEAD
} > "$source_dir/SOURCE_REVISIONS.txt"
