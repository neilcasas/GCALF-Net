#!/usr/bin/env bash
# Archive the grade-remediation evidence from a live Vast instance atomically.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/pull_grade_pilot_artifacts.sh --instance-id ID --destination DIR

Copies Fix B and the four stopped grade-pilot arms into a gzip-compressed tarball,
then writes and verifies a SHA-256 manifest. The destination must not exist.
EOF
}

die() { echo "error: $*" >&2; exit 2; }

instance_id=
destination=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-id) [[ $# -ge 2 ]] || die "--instance-id requires a value"; instance_id=$2; shift 2 ;;
        --destination) [[ $# -ge 2 ]] || die "--destination requires a value"; destination=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "$instance_id" =~ ^[0-9]+$ ]] || die "--instance-id must be numeric"
[[ -n "$destination" && ! -e "$destination" ]] || die "destination must not already exist"

ssh_url=$(vastai ssh-url "$instance_id")
[[ "$ssh_url" =~ ^ssh://root@([^:]+):([0-9]+)$ ]] || die "unexpected Vast SSH URL: $ssh_url"
host=${BASH_REMATCH[1]}
port=${BASH_REMATCH[2]}
mkdir -p "$destination"
archive="$destination/grade-pilot-artifacts.tar.gz"

remote_paths=(
    workspace/evidence/stage3-fixb-20260916
    workspace/evidence/stage3-fixb-20260916-v2
)
for arm in baseline lff caf full; do
    root="workspace/det_models/Task2201_PICAI_csPCa/RetinaUNetV001_D3V001_3d_gradepilot_${arm}/fold0"
    remote_paths+=("$root/mlruns" "$root/config_resolved.yaml" "$root/train.log"
                   "$root/pilot_eval/picai/grade_metrics.json" "$root/pilot_eval/val_predictions")
done

ssh -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" \
    "tar -C / -czf - -- ${remote_paths[*]}" >"$archive"
gzip -t "$archive"
(cd "$destination" && sha256sum "$(basename "$archive")" > SHA256SUMS && sha256sum -c SHA256SUMS)
tar -tzf "$archive" >/dev/null
