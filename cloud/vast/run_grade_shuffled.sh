#!/usr/bin/env bash
# Run the preregistered shuffled-label transfer control on one GPU.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/run_grade_shuffled.sh --task TASK --tag TAG [options]

Runs the fold-0 shuffled-label control from gcalf_final_shuffled on one GPU.
This control is reported separately and is intentionally not part of the
run_matrix.sh detection safety gate.

Options:
  --task TASK       Task2201_PICAI_csPCa (required)
  --tag TAG         New unique run suffix (required)
  --fold N          Fold, default 0
  --seed N          Experiment seed, default 2026
  --gpu N           CUDA device, default 0
  --repo-dir DIR    Checkout, default current directory
  --python PATH     Python executable, default python
  --dry-run         Print the exact command without running it
EOF
}

die() { echo "error: $*" >&2; exit 2; }

task=
tag=
fold=0
seed=2026
gpu=0
repo_dir=$PWD
python_bin=python
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) [[ $# -ge 2 ]] || die "--task requires a value"; task=$2; shift 2 ;;
        --tag) [[ $# -ge 2 ]] || die "--tag requires a value"; tag=$2; shift 2 ;;
        --fold) [[ $# -ge 2 ]] || die "--fold requires a value"; fold=$2; shift 2 ;;
        --seed) [[ $# -ge 2 ]] || die "--seed requires a value"; seed=$2; shift 2 ;;
        --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu=$2; shift 2 ;;
        --repo-dir) [[ $# -ge 2 ]] || die "--repo-dir requires a value"; repo_dir=$2; shift 2 ;;
        --python) [[ $# -ge 2 ]] || die "--python requires a value"; python_bin=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$task" == "Task2201_PICAI_csPCa" ]] || die "--task must be Task2201_PICAI_csPCa"
[[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || die "--tag must contain only letters, numbers, underscore, or dash"
[[ "$fold" =~ ^[0-4]$ ]] || die "--fold must be 0 through 4"
[[ "$gpu" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
[[ -f "$repo_dir/scripts/train.py" ]] || die "--repo-dir is not a GCALF-Net checkout: $repo_dir"
[[ -n "${det_models:-}" ]] || die "det_models must name the model root"

run_tag="_shuffled_${tag}"
output_dir="$det_models/$task/RetinaUNetV001_D3V001_3d${run_tag}/fold${fold}"
[[ ! -e "$output_dir" ]] || die "refusing to overwrite existing run: $output_dir"

export det_num_threads=${det_num_threads:-16}
command=("$python_bin" scripts/train.py "$task" --sweep -o
         "train=gcalf_final_shuffled" "exp.fold=$fold" "exp.seed=$seed" "exp.tag=$run_tag"
         "augment_cfg.multiprocessing=true")

if [[ "$dry_run" == true ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q det_num_threads=%q ' "$gpu" "$det_num_threads"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

cd "$repo_dir"
CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
