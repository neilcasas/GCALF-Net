#!/usr/bin/env bash
# Package validation outputs without deleting the source evidence or generated model artifacts.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/export_results.sh [options]

Options:
  --task-dir DIR       Validated Task2201_PICAI_GGG directory (required)
  --evidence-dir DIR   M0--M3 evidence directory (required)
  --model-dir DIR      M3 RetinaUNet model directory (required)
  --export-dir DIR     Empty target directory (default: /workspace/export)
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

task_dir=
evidence_dir=
model_dir=
export_dir=/workspace/export
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-dir) [[ $# -ge 2 ]] || die "--task-dir requires a value"; task_dir=$2; shift 2 ;;
        --evidence-dir) [[ $# -ge 2 ]] || die "--evidence-dir requires a value"; evidence_dir=$2; shift 2 ;;
        --model-dir) [[ $# -ge 2 ]] || die "--model-dir requires a value"; model_dir=$2; shift 2 ;;
        --export-dir) [[ $# -ge 2 ]] || die "--export-dir requires a value"; export_dir=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ -n "$task_dir" && -n "$evidence_dir" && -n "$model_dir" ]] || die "task, evidence, and model directories are required"
[[ -d "$task_dir" ]] || die "task directory does not exist: $task_dir"
[[ -d "$evidence_dir" ]] || die "evidence directory does not exist: $evidence_dir"
[[ -d "$model_dir" ]] || die "model directory does not exist: $model_dir"
for artifact in model_best.ckpt model_last.ckpt plan_inference.pkl; do
    [[ -f "$model_dir/$artifact" ]] || die "M3 artifact is missing: $model_dir/$artifact"
done
[[ -d "$model_dir/test_predictions" ]] || die "M3 predictions are missing: $model_dir/test_predictions"
[[ -f "$model_dir/test_results/picai/metrics.csv" ]] || die "M3 metrics are missing: $model_dir/test_results/picai/metrics.csv"
[[ ! -e "$export_dir" || -z "$(find "$export_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || die "refusing to overwrite existing export output: $export_dir"
mkdir -p "$export_dir"

task_name=$(basename "$task_dir")
model_task_name=$(basename "$(dirname "$model_dir")")
tar -C "$(dirname "$task_dir")" -cf "$export_dir/${task_name}.tar" "$task_name"
tar -C "$(dirname "$evidence_dir")" -cf "$export_dir/milestone_evidence.tar" "$(basename "$evidence_dir")"
tar -C "$(dirname "$(dirname "$model_dir")")" -cf "$export_dir/${model_task_name}_m3_model.tar" "$(basename "$(dirname "$model_dir")")"
(cd "$export_dir" && sha256sum -- *.tar > SHA256SUMS)
