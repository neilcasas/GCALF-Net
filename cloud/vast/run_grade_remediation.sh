#!/usr/bin/env bash
# Run one explicitly named Stage 1--3 remediation diagnostic on a single GPU.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/run_grade_remediation.sh --task TASK --stage STAGE --tag TAG [options]

Stages:
  throughput  Two epochs with multiprocessing augmentation; writes timing logs.
  anchor      One epoch that writes grade_anchor_counts.json.
  fix-a       Fresh 13-epoch pilot using --anchor-counts measured by `anchor`.
  fix-b       Historical fresh 13-epoch grade-balanced-sampling pilot.
  fix-bprime  Corrected Fix B′ pilot with sampled-anchor CE weights.

Options:
  --task TASK             Task2201_PICAI_csPCa (required)
  --stage STAGE           One of the stages above (required)
  --tag TAG               New unique run suffix (required)
  --fold N                Fold, default 0
  --anchor-counts LIST    Four comma-separated GGG2--5 counts, required for fix-a
  --repo-dir DIR          Checkout, default current directory
  --python PATH           Python executable, default python
  --dry-run               Print the exact command without running it

The caller must first copy and verify canonical labels off-instance. The script
refuses an existing output directory and always enables multiprocessing with
det_num_threads=16 unless the caller deliberately overrides that environment.
EOF
}

die() { echo "error: $*" >&2; exit 2; }

task=
stage=
tag=
fold=0
anchor_counts=
repo_dir=$PWD
python_bin=python
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) [[ $# -ge 2 ]] || die "--task requires a value"; task=$2; shift 2 ;;
        --stage) [[ $# -ge 2 ]] || die "--stage requires a value"; stage=$2; shift 2 ;;
        --tag) [[ $# -ge 2 ]] || die "--tag requires a value"; tag=$2; shift 2 ;;
        --fold) [[ $# -ge 2 ]] || die "--fold requires a value"; fold=$2; shift 2 ;;
        --anchor-counts) [[ $# -ge 2 ]] || die "--anchor-counts requires a value"; anchor_counts=$2; shift 2 ;;
        --repo-dir) [[ $# -ge 2 ]] || die "--repo-dir requires a value"; repo_dir=$2; shift 2 ;;
        --python) [[ $# -ge 2 ]] || die "--python requires a value"; python_bin=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "$task" == "Task2201_PICAI_csPCa" ]] || die "--task must be Task2201_PICAI_csPCa"
[[ "$stage" =~ ^(throughput|anchor|fix-a|fix-b|fix-bprime)$ ]] || die "invalid --stage: $stage"
[[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || die "--tag must contain only letters, numbers, underscore, or dash"
[[ "$fold" =~ ^[0-4]$ ]] || die "--fold must be 0 through 4"
[[ -f "$repo_dir/scripts/train.py" ]] || die "--repo-dir is not a GCALF-Net checkout: $repo_dir"
[[ -n "${det_models:-}" ]] || die "det_models must name the model root"

run_tag="_remediation_${stage}_${tag}"
output_dir="$det_models/$task/RetinaUNetV001_D3V001_3d${run_tag}/fold${fold}"
[[ ! -e "$output_dir" ]] || die "refusing to overwrite existing remediation run: $output_dir"

overrides=("train=gcalf_grade_pilot" "exp.fold=$fold" "exp.seed=2026" "exp.tag=$run_tag"
           "augment_cfg.multiprocessing=true")
case "$stage" in
    throughput)
        overrides+=("trainer_cfg.max_num_epochs=2" "trainer_cfg.swa_epochs=0")
        ;;
    anchor)
        overrides+=("trainer_cfg.max_num_epochs=1" "trainer_cfg.swa_epochs=0")
        ;;
    fix-a)
        [[ "$anchor_counts" =~ ^[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$ ]] \
            || die "fix-a requires four positive comma-separated --anchor-counts"
        overrides+=("trainer_cfg.grade_class_weight_source=anchor"
                    "trainer_cfg.grade_anchor_class_counts=[$anchor_counts]")
        ;;
    fix-b)
        overrides[0]="train=gcalf_grade_remediation"
        ;;
    fix-bprime)
        overrides[0]="train=gcalf_grade_remediation_bprime"
        ;;
esac

export det_num_threads=${det_num_threads:-16}
command=("$python_bin" scripts/train.py "$task")
if [[ "$stage" == "fix-a" || "$stage" == "fix-b" || "$stage" == "fix-bprime" ]]; then
    command+=(--sweep)
fi
command+=(-o "${overrides[@]}")

if [[ "$dry_run" == true ]]; then
    printf 'det_num_threads=%q ' "$det_num_threads"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

cd "$repo_dir"
"${command[@]}"
