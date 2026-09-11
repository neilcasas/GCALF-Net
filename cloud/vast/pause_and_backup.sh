#!/usr/bin/env bash
# Preserve explicitly named training trees before and after a graceful interrupt.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/pause_and_backup.sh --backup-dir DIR --source-dir DIR [options]

Options:
  --backup-dir DIR    New, empty destination for immutable tar snapshots (required)
  --source-dir DIR    Training/evidence directory to preserve; may be repeated (required)
  --pid PID           Send this local trainer process SIGINT before the final snapshot; may be repeated
  --wait-seconds N    Maximum seconds to wait for a graceful stop (default: 600)

The command never deletes or overwrites a source. It writes a pre-interrupt
snapshot when --pid is supplied, then a final snapshot and SHA256SUMS after the
trainer exits. Run it on the Vast instance, where det_models is mounted.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

backup_dir=
declare -a pids=()
wait_seconds=600
declare -a source_dirs=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backup-dir) [[ $# -ge 2 ]] || die "--backup-dir requires a value"; backup_dir=$2; shift 2 ;;
        --source-dir) [[ $# -ge 2 ]] || die "--source-dir requires a value"; source_dirs+=("$2"); shift 2 ;;
        --pid) [[ $# -ge 2 ]] || die "--pid requires a value"; pids+=("$2"); shift 2 ;;
        --wait-seconds) [[ $# -ge 2 ]] || die "--wait-seconds requires a value"; wait_seconds=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$backup_dir" ]] || die "--backup-dir is required"
(( ${#source_dirs[@]} > 0 )) || die "at least one --source-dir is required"
[[ "$wait_seconds" =~ ^[0-9]+$ ]] && (( wait_seconds > 0 )) || die "--wait-seconds must be positive"
[[ ! -e "$backup_dir" ]] || die "refusing to overwrite existing backup directory: $backup_dir"
for source_dir in "${source_dirs[@]}"; do
    [[ -d "$source_dir" ]] || die "source directory does not exist: $source_dir"
done
for pid in "${pids[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || die "--pid must be numeric"
    kill -0 "$pid" 2>/dev/null || die "trainer process is not running: $pid"
done

mkdir -p "$backup_dir"

snapshot() {
    local label=$1
    local index=0
    local source_dir
    for source_dir in "${source_dirs[@]}"; do
        index=$((index + 1))
        tar -C "$(dirname "$source_dir")" -cf \
            "$backup_dir/${label}_${index}_$(basename "$source_dir").tar" "$(basename "$source_dir")"
    done
}

if (( ${#pids[@]} > 0 )); then
    snapshot pre_interrupt
    for pid in "${pids[@]}"; do
        kill -INT "$pid"
    done
    for ((elapsed = 0; elapsed < wait_seconds; elapsed += 5)); do
        still_running=false
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still_running=true
            fi
        done
        if [[ "$still_running" == false ]]; then
            break
        fi
        sleep 5
    done
    still_running=()
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && still_running+=("$pid")
    done
    (( ${#still_running[@]} == 0 )) || \
        die "trainers ${still_running[*]} did not exit within ${wait_seconds}s; pre-interrupt snapshot is retained at $backup_dir"
fi

snapshot final
(cd "$backup_dir" && sha256sum -- *.tar > SHA256SUMS)
echo "Backup complete: $backup_dir"
