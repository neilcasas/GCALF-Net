#!/usr/bin/env bash
# Copy the irreplaceable Task2201 raw grade metadata off a Vast instance and verify it.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/pull_grade_metadata.sh --instance-id ID --destination DIR [options]

Options:
  --instance-id ID       Vast instance holding the authoritative task (required)
  --destination DIR      New local destination directory (required)
  --remote-labels DIR    labelsTr directory on the instance
                        (default: /workspace/det_data/Task2201_PICAI_csPCa/raw_splitted/labelsTr)
  --dry-run              Print the copy and verification commands only
EOF
}

die() { echo "error: $*" >&2; exit 2; }

instance_id=
destination=
remote_labels=/workspace/det_data/Task2201_PICAI_csPCa/raw_splitted/labelsTr
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-id) [[ $# -ge 2 ]] || die "--instance-id requires a value"; instance_id=$2; shift 2 ;;
        --destination) [[ $# -ge 2 ]] || die "--destination requires a value"; destination=$2; shift 2 ;;
        --remote-labels) [[ $# -ge 2 ]] || die "--remote-labels requires a value"; remote_labels=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "$instance_id" =~ ^[0-9]+$ ]] || die "--instance-id must be numeric"
[[ -n "$destination" ]] || die "--destination is required"
[[ ! -e "$destination" ]] || die "refusing to overwrite destination: $destination"
repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=${PYTHON_BIN:-python3}
command -v "$python_bin" >/dev/null || die "Python interpreter not found: $python_bin"

if [[ "$dry_run" == true ]]; then
    echo "vastai copy C.${instance_id}:${remote_labels} local:${destination}"
    echo "$python_bin $repo_dir/scripts/audit_grade_metadata.py --labels-dir $destination[/labelsTr] --manifest $destination/grade_metadata_manifest.json"
    exit 0
fi

mkdir -p "$destination"
vastai copy "C.${instance_id}:${remote_labels}" "local:${destination}"
labels_dir=$destination
[[ -d "$destination/labelsTr" ]] && labels_dir="$destination/labelsTr"
# Vast's directory-copy command has no include filter. Retain only the requested
# grade metadata before producing a manifest or leaving a local backup behind.
find "$labels_dir" -type f ! -name '*.json' -delete
"$python_bin" "$repo_dir/scripts/audit_grade_metadata.py" --labels-dir "$labels_dir" \
    --manifest "$destination/grade_metadata_manifest.json"
