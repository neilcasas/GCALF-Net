#!/usr/bin/env bash
# Manage Vast.ai validation/training instances. Mutating commands are explicit.
set -euo pipefail

IMAGE_DEFAULT=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

usage() {
    cat <<'EOF'
Usage: cloud/vast/host.sh <command> [options]

Commands:
  search [--gpu-name NAME] [--num-gpus N] [--min-ram GB] [--dry-run]
  create --offer-id ID --disk-gb GB [--image IMAGE] [--label LABEL] [--dry-run]
  status --instance-id ID [--dry-run]
  ssh-url --instance-id ID [--dry-run]
  stop --instance-id ID [--dry-run]
  destroy --instance-id ID --confirm [--dry-run]
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

require_vastai() {
    [[ "$dry_run" == true ]] || command -v vastai >/dev/null || die "vastai CLI is not installed or is not on PATH"
}

run() {
    if [[ "$dry_run" == true ]]; then
        printf 'dry-run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
command_name=$1
shift
dry_run=false
offer_id=
instance_id=
disk_gb=
image=$IMAGE_DEFAULT
label=gcalf-m1-m6
gpu_name=
num_gpus=4
min_ram=24
confirm=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offer-id) [[ $# -ge 2 ]] || die "--offer-id requires a value"; offer_id=$2; shift 2 ;;
        --instance-id) [[ $# -ge 2 ]] || die "--instance-id requires a value"; instance_id=$2; shift 2 ;;
        --disk-gb) [[ $# -ge 2 ]] || die "--disk-gb requires a value"; disk_gb=$2; shift 2 ;;
        --image) [[ $# -ge 2 ]] || die "--image requires a value"; image=$2; shift 2 ;;
        --label) [[ $# -ge 2 ]] || die "--label requires a value"; label=$2; shift 2 ;;
        --gpu-name) [[ $# -ge 2 ]] || die "--gpu-name requires a value"; gpu_name=$2; shift 2 ;;
        --num-gpus) [[ $# -ge 2 ]] || die "--num-gpus requires a value"; num_gpus=$2; shift 2 ;;
        --min-ram) [[ $# -ge 2 ]] || die "--min-ram requires a value"; min_ram=$2; shift 2 ;;
        --confirm) confirm=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$command_name" in
    search)
        [[ -z "$offer_id$instance_id$disk_gb" && "$confirm" == false ]] || die "search does not accept instance options"
        [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || die "--num-gpus must be a positive integer"
        [[ "$min_ram" =~ ^[1-9][0-9]*$ ]] || die "--min-ram must be a positive integer"
        require_vastai
        query="num_gpus=${num_gpus} gpu_ram>=${min_ram} cpu_cores>=32 cpu_ram>=128 disk_space>=500 reliability>=0.99 direct_port_count>=1 verified=true rentable=true compute_cap>=800 compute_cap<=1200"
        if [[ -n "$gpu_name" ]]; then
            query="gpu_name=${gpu_name} ${query}"
        fi
        run vastai search offers \
            "$query" \
            --type on-demand -o 'reliability-' --raw
        ;;
    create)
        [[ -n "$offer_id" ]] || die "create requires --offer-id"
        [[ "$disk_gb" =~ ^[0-9]+$ ]] && (( disk_gb >= 500 )) || die "create requires --disk-gb of at least 500"
        [[ -z "$instance_id" && "$confirm" == false ]] || die "create accepts only offer, disk, image, and dry-run options"
        require_vastai
        run vastai create instance "$offer_id" --image "$image" --disk "$disk_gb" --ssh --direct --label "$label" --raw
        ;;
    status)
        [[ -n "$instance_id" ]] || die "status requires --instance-id"
        [[ -z "$offer_id$disk_gb" && "$confirm" == false ]] || die "status accepts only --instance-id and --dry-run"
        require_vastai
        run vastai show instance "$instance_id" --raw
        ;;
    ssh-url)
        [[ -n "$instance_id" ]] || die "ssh-url requires --instance-id"
        [[ -z "$offer_id$disk_gb" && "$confirm" == false ]] || die "ssh-url accepts only --instance-id and --dry-run"
        require_vastai
        run vastai ssh-url "$instance_id" --raw
        ;;
    stop)
        [[ -n "$instance_id" ]] || die "stop requires --instance-id"
        [[ -z "$offer_id$disk_gb" && "$confirm" == false ]] || die "stop accepts only --instance-id and --dry-run"
        require_vastai
        run vastai stop instance "$instance_id"
        ;;
    destroy)
        [[ -n "$instance_id" ]] || die "destroy requires --instance-id"
        [[ "$confirm" == true ]] || die "destroy requires --confirm after local checksum verification"
        [[ -z "$offer_id$disk_gb" ]] || die "destroy accepts only --instance-id, --confirm, and --dry-run"
        require_vastai
        run vastai destroy instance "$instance_id"
        ;;
    *)
        usage >&2
        die "unknown command: $command_name"
        ;;
esac
