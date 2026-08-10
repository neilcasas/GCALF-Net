#!/usr/bin/env bash
# Execute the M0--M3 one-instance validation gates in order.  Any failed gate stops the run.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/run_milestones.sh [--repo-dir DIR] [--workspace DIR] [--m3-task NAME]
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
workspace=/workspace
m3_task=Task900_PICAI_TINY
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir) [[ $# -ge 2 ]] || die "--repo-dir requires a value"; repo_dir=$2; shift 2 ;;
        --workspace) [[ $# -ge 2 ]] || die "--workspace requires a value"; workspace=$2; shift 2 ;;
        --m3-task) [[ $# -ge 2 ]] || die "--m3-task requires a value"; m3_task=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$m3_task" =~ ^Task9[0-9][0-9]_PICAI_TINY$ ]] || die "--m3-task must be an unused Task9xx_PICAI_TINY identifier"
[[ -f "$repo_dir/cloud/vast/download_picai.sh" ]] || die "--repo-dir is not this checkout: $repo_dir"

export det_data=${det_data:-$workspace/det_data}
export det_models=${det_models:-$workspace/det_models}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export det_num_threads=${det_num_threads:-8}
evidence_dir=$workspace/evidence
mkdir -p "$evidence_dir/m0" "$evidence_dir/m1" "$evidence_dir/m2" "$evidence_dir/m3" "$det_data" "$det_models"

run_logged() {
    local log_path=$1
    shift
    "$@" 2>&1 | tee "$log_path"
}

cd "$repo_dir"
run_logged "$evidence_dir/m0/pytest.log" python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py tests/test_csrc_cuda.py
if rg -qi 'skipped' "$evidence_dir/m0/pytest.log"; then
    die "M0 CUDA test was skipped; the extension gate requires a CUDA execution"
fi
run_logged "$evidence_dir/m0/nndet-import.log" python -c 'import nndet, nndet._C; print(nndet.__file__)'
expected_checkout=$repo_dir/nndet
grep -F "$expected_checkout" "$evidence_dir/m0/nndet-import.log" >/dev/null || die "nndet did not resolve to this checkout"
run_logged "$evidence_dir/m0/tool-imports.log" python -c 'import picai_prep, picai_eval, medcam'

[[ ! -e "$det_data/Task2201_PICAI_GGG" ]] || die "refusing to reuse existing M1 task: $det_data/Task2201_PICAI_GGG"
[[ ! -e "$workspace/picai_m1_work" ]] || die "refusing to reuse existing M1 workspace: $workspace/picai_m1_work"
run_logged "$evidence_dir/m1/download.log" "$repo_dir/cloud/vast/download_picai.sh" --source-dir "$workspace/source"
m1_task=$det_data/Task2201_PICAI_GGG
run_logged "$evidence_dir/m1/build.log" python -m gcalf_data.prepare_picai build \
    --images-dir "$workspace/source/picai_public_images" \
    --labels-root "$workspace/source/picai_labels" \
    --splits-json "$workspace/source/picai_baseline/src/picai_baseline/splits/picai_nnunet/splits.json" \
    --task-dir "$m1_task" --work-dir "$workspace/picai_m1_work"
run_logged "$evidence_dir/m1/preprocess.log" nndet_prep Task2201_PICAI_GGG
run_logged "$evidence_dir/m1/install-splits.log" python -m gcalf_data.prepare_picai install-splits \
    --task-dir "$m1_task" --preprocessed-dir "$m1_task/preprocessed"
run_logged "$evidence_dir/m1/sanity.log" python -m gcalf_data.sanity_checks \
    --task-dir "$m1_task" --marksheet "$workspace/source/picai_labels/clinical_information/marksheet.csv" \
    --plan-path "$m1_task/preprocessed/D3V001_3d.pkl" --report-path "$evidence_dir/m1/data_report.md"

run_logged "$evidence_dir/m2/overfit.log" env RUN_GCALF_M2=1 GCALF_M2_TASK=Task2201_PICAI_GGG \
    python -m pytest -q -s tests/test_overfit.py
run_logged "$evidence_dir/m3/smoke.log" bash scripts/run_m3_smoke.sh "$m1_task" "$det_data/$m3_task"
