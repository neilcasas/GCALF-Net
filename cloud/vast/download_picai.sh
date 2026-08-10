#!/usr/bin/env bash
# Download the original PI-CAI release, verify Zenodo's published MD5s, then prepare sources.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/download_picai.sh [options]

Options:
  --source-dir DIR    Parent for archives, picai_public_images, and source repositories (default: /workspace/source)
  --manifest FILE     Three-column TSV: filename, published MD5, URL
  --verify-only       Verify archives only; do not extract or clone
  --dry-run           Print downloads/extraction/clones without changing files
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
source_dir=/workspace/source
manifest=$repo_dir/cloud/vast/picai_zenodo_manifest.tsv
verify_only=false
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir) [[ $# -ge 2 ]] || die "--source-dir requires a value"; source_dir=$2; shift 2 ;;
        --manifest) [[ $# -ge 2 ]] || die "--manifest requires a value"; manifest=$2; shift 2 ;;
        --verify-only) verify_only=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -f "$manifest" ]] || die "manifest does not exist: $manifest"

filenames=()
checksums=()
urls=()
while IFS=$'\t' read -r filename checksum url extra; do
    [[ -z "$filename" || "$filename" == \#* ]] && continue
    [[ -z "$checksum" || -z "$url" || -n "$extra" ]] && die "invalid manifest row for $filename"
    [[ "$filename" =~ ^picai_public_images_fold[0-4]\.zip$ ]] || die "unexpected archive name: $filename"
    [[ "$checksum" =~ ^[a-fA-F0-9]{32}$ ]] || die "invalid MD5 for $filename"
    [[ "$url" =~ ^https://zenodo\.org/ ]] || die "archive URL must be a Zenodo HTTPS URL: $url"
    for seen in "${filenames[@]}"; do [[ "$seen" != "$filename" ]] || die "duplicate archive name: $filename"; done
    filenames+=("$filename")
    checksums+=("$checksum")
    urls+=("$url")
done < "$manifest"
[[ ${#filenames[@]} -eq 5 ]] || die "manifest must contain exactly five PI-CAI archives"

run() {
    if [[ "$dry_run" == true ]]; then
        printf 'dry-run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

archive_dir=$source_dir/archives
if [[ "$dry_run" == false ]]; then mkdir -p "$archive_dir"; fi
for index in "${!filenames[@]}"; do
    archive=$archive_dir/${filenames[$index]}
    if [[ "$verify_only" == false && ! -f "$archive" ]]; then
        run curl --fail --location --continue-at - --retry 5 --retry-all-errors --output "$archive" "${urls[$index]}"
    fi
    if [[ "$dry_run" == true && ! -f "$archive" ]]; then
        run md5sum "$archive"
        continue
    fi
    [[ -f "$archive" ]] || die "archive is missing: $archive"
    actual=$(md5sum "$archive" | awk '{print $1}')
    [[ "$actual" == "${checksums[$index]}" ]] || die "checksum mismatch for ${filenames[$index]}: expected ${checksums[$index]}, got $actual"
done

[[ "$verify_only" == false ]] || exit 0
images_dir=$source_dir/picai_public_images
labels_dir=$source_dir/picai_labels
baseline_dir=$source_dir/picai_baseline
for target in "$images_dir" "$labels_dir" "$baseline_dir"; do
    [[ ! -e "$target" ]] || die "refusing to overwrite existing output: $target"
done

run mkdir -p "$images_dir"
for index in "${!filenames[@]}"; do
    run unzip -q "$archive_dir/${filenames[$index]}" -d "$images_dir"
done
run git clone https://github.com/DIAGNijmegen/picai_labels.git "$labels_dir"
run git clone https://github.com/DIAGNijmegen/picai_baseline.git "$baseline_dir"
if [[ "$dry_run" == false ]]; then
    {
        printf 'zenodo_record=6517398\n'
        printf 'picai_labels='; git -C "$labels_dir" rev-parse HEAD
        printf 'picai_baseline='; git -C "$baseline_dir" rev-parse HEAD
    } > "$source_dir/SOURCE_REVISIONS.txt"
fi
