#!/usr/bin/env bash
# Launch the 20-run study matrix (4 arms x 5 folds) fold-by-fold.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/run_matrix.sh --repo-dir DIR [options]

Launches one fold's four arms (baseline, lff, caf, full) concurrently, one
per GPU. After all four complete, runs scripts/check_matrix_fold.py -- an
infrastructure/correctness gate only (checkpoints exist, no real exception in
train.log, config_resolved.yaml matches the locked protocol, detection metric
finite and non-degenerate, wall clock plausible). It never inspects a grade
metric: per ADR 0003's Statistical Consequences, endpoints and configuration
must not change once fold results are known, and using this gate to judge an
arm's science would be exactly that. After fold 0, when both baseline and full
are present, scripts/check_fold0_detection_gate.py inspects only lesion_ap and
picai_score, the detection endpoint ADR 0005:81-83 declares an
infrastructure/safety decision. A failed gate stops the matrix before the next
fold launches; nothing is deleted.

Options:
  --repo-dir DIR      Checkout to run from (required)
  --fold-start N      First fold to launch, default 0
  --fold-end N        Last fold to launch (inclusive), default 4
  --arms LIST         Space-separated arms, default: baseline lff caf full
                      a full run spanning fold 0 into later folds must include baseline
  --seed N            Default 2026
  --python PATH       Python executable, default python
  --m5-record PATH    Default evidence/m5-budget-rung.json relative to --repo-dir;
                       pass an empty string to skip the wall-clock check
  --dry-run           Print the commands for fold-start only, without running them

Requires det_data and det_models to be set. Refuses to overwrite an existing
fold/arm output directory, matching cloud/vast/run_grade_remediation.sh.
EOF
}

die() { echo "error: $*" >&2; exit 2; }

repo_dir=
fold_start=0
fold_end=4
seed=2026
python_bin=python
m5_record=
dry_run=false
m5_record_set=false
arms=(baseline lff caf full)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir) [[ $# -ge 2 ]] || die "--repo-dir requires a value"; repo_dir=$2; shift 2 ;;
        --fold-start) [[ $# -ge 2 ]] || die "--fold-start requires a value"; fold_start=$2; shift 2 ;;
        --fold-end) [[ $# -ge 2 ]] || die "--fold-end requires a value"; fold_end=$2; shift 2 ;;
        --arms)
            shift
            arms=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                arms+=("$1")
                shift
            done
            ((${#arms[@]} > 0)) || die "--arms requires at least one arm"
            ;;
        --seed) [[ $# -ge 2 ]] || die "--seed requires a value"; seed=$2; shift 2 ;;
        --python) [[ $# -ge 2 ]] || die "--python requires a value"; python_bin=$2; shift 2 ;;
        --m5-record) [[ $# -ge 2 ]] || die "--m5-record requires a value"; m5_record=$2; m5_record_set=true; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$repo_dir" ]] || die "--repo-dir is required"
[[ -f "$repo_dir/scripts/train.py" ]] || die "--repo-dir is not a GCALF-Net checkout: $repo_dir"
[[ -f "$repo_dir/scripts/check_matrix_fold.py" ]] || die "--repo-dir is missing scripts/check_matrix_fold.py"
[[ -f "$repo_dir/scripts/check_fold0_detection_gate.py" ]] || die "--repo-dir is missing scripts/check_fold0_detection_gate.py"
[[ "$fold_start" =~ ^[0-4]$ ]] || die "--fold-start must be 0 through 4"
[[ "$fold_end" =~ ^[0-4]$ ]] || die "--fold-end must be 0 through 4"
[[ "$fold_start" -le "$fold_end" ]] || die "--fold-start must be <= --fold-end"
for arm in "${arms[@]}"; do
    [[ "$arm" =~ ^(baseline|lff|caf|full)$ ]] || die "unknown matrix arm: $arm"
done
has_baseline=false
has_full=false
for arm in "${arms[@]}"; do
    [[ "$arm" == baseline ]] && has_baseline=true
    [[ "$arm" == full ]] && has_full=true
done
if [[ "$fold_start" == 0 && "$fold_end" -gt 0 && "$has_full" == true && "$has_baseline" == false ]]; then
    die "--arms includes full but not baseline while spanning fold 0 through later folds; requires baseline so fold 0 can run the detection safety gate"
fi
[[ -n "${det_data:-}" ]] || die "det_data must name the data root"
[[ -n "${det_models:-}" ]] || die "det_models must name the model root"

if [[ "$m5_record_set" == false ]]; then
    m5_record="$repo_dir/evidence/m5-budget-rung.json"
fi
if [[ -n "$m5_record" ]]; then
    [[ -f "$m5_record" ]] || die "--m5-record not found: $m5_record (pass --m5-record '' to skip)"
fi

task=Task2201_PICAI_csPCa
fold0_gate_record="$det_models/$task/RetinaUNetV001_D3V001_3d_full/fold0/fold0_detection_gate.json"
gate_script="$repo_dir/scripts/check_fold0_detection_gate.py"
if [[ "$fold_start" -gt 0 ]]; then
    [[ -f "$fold0_gate_record" ]] || die "--fold-start $fold_start requires a recorded passing fold-0 detection gate: $fold0_gate_record"
    if ! "$python_bin" "$gate_script" "$task" --validate-record "$fold0_gate_record"; then
        die "recorded fold-0 detection gate is not a valid locked pass: $fold0_gate_record"
    fi
fi

export det_num_threads=${det_num_threads:-16}
cd "$repo_dir"

for fold in $(seq "$fold_start" "$fold_end"); do
    echo "=== Fold $fold: launching ${#arms[@]} arms across ${#arms[@]} GPUs ==="
    declare -a pids=()
    for gpu in "${!arms[@]}"; do
        arm=${arms[$gpu]}
        output_dir="$det_models/$task/RetinaUNetV001_D3V001_3d_${arm}/fold${fold}"
        [[ ! -e "$output_dir" ]] || die "refusing to overwrite existing run: $output_dir"
    done

    if [[ "$dry_run" == true ]]; then
        for gpu in "${!arms[@]}"; do
            arm=${arms[$gpu]}
            train_config="gcalf_${arm}"
            [[ "$arm" == full ]] && train_config=gcalf_final
            printf 'CUDA_VISIBLE_DEVICES=%s det_num_threads=%q ' "$gpu" "$det_num_threads"
            printf '%q ' "$python_bin" scripts/train.py "$task" --sweep -o \
                "train=$train_config" "exp.fold=${fold}" "exp.seed=${seed}" "exp.tag=_${arm}" \
                "augment_cfg.multiprocessing=true"
            printf '\n'
        done
        if [[ "$fold" == 0 && " ${arms[*]} " == *" baseline "* && " ${arms[*]} " == *" full "* ]]; then
            printf '%q ' "$python_bin" scripts/check_fold0_detection_gate.py "$task" --fold 0 --models-root "$det_models"
            printf '\n'
        fi
        exit 0
    fi

    for gpu in "${!arms[@]}"; do
        arm=${arms[$gpu]}
        train_config="gcalf_${arm}"
        [[ "$arm" == full ]] && train_config=gcalf_final
        log_dir="$det_models/$task/RetinaUNetV001_D3V001_3d_${arm}"
        mkdir -p "$log_dir"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/train.py "$task" --sweep -o \
            "train=$train_config" "exp.fold=${fold}" "exp.seed=${seed}" "exp.tag=_${arm}" \
            "augment_cfg.multiprocessing=true" \
            > "$log_dir/launch_fold${fold}.log" 2>&1 &
        pids+=("$!")
    done

    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    if (( status != 0 )); then
        echo "FOLD $fold: one or more arms exited non-zero. Stopping before the next fold;" >&2
        echo "evidence preserved under $det_models/$task/*/fold${fold}/." >&2
        exit 1
    fi

    echo "=== Fold $fold: training complete, running infrastructure sanity gate ==="
    gate_args=("$task" --fold "$fold" --arms "${arms[@]}" --models-root "$det_models")
    if [[ -n "$m5_record" ]]; then
        gate_args+=(--m5-record "$m5_record")
    fi
    "$python_bin" scripts/check_matrix_fold.py "${gate_args[@]}"

    if [[ "$fold" == 0 && " ${arms[*]} " == *" baseline "* && " ${arms[*]} " == *" full "* ]]; then
        echo "=== Fold 0: running ADR 0005 detection safety gate ==="
        "$python_bin" scripts/check_fold0_detection_gate.py "$task" --fold 0 --models-root "$det_models"
    fi

    echo "=== Fold $fold passed. ==="
done

echo "Folds ${fold_start}-${fold_end} launched and passed the sanity gate for all arms."
