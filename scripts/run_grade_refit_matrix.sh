#!/usr/bin/env bash
# Run the fixed Fold-0 grade-only pilot across the four GCALF architectures.
set -euo pipefail

repo_dir=${1:-/workspace/GCALF-Net-continuation}
task=Task2201_PICAI_csPCa
models_root=/workspace/det_models
data_root=/workspace/det_data
split_manifest="$models_root/$task/grade_overfitting_pilot/split_seed2026.json"
python_bin=/workspace/gcalf-venv/bin/python
models=(
    RetinaUNetV001_D3V001_3d_baseline
    RetinaUNetV001_D3V001_3d_lff
    RetinaUNetV001_D3V001_3d_caf
    RetinaUNetV001_D3V001_3d_full
)

export det_data=$data_root det_models=$models_root OMP_NUM_THREADS=8 det_num_threads=8
cd "$repo_dir"
[[ -f "$split_manifest" ]] || { echo "Missing split manifest: $split_manifest" >&2; exit 2; }

for condition in control regularized; do
    pids=()
    for gpu in "${!models[@]}"; do
        model=${models[$gpu]}
        fold_dir="$models_root/$task/$model/fold0"
        output="$fold_dir/grade_overfitting_pilot/$condition"
        [[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 2; }
        CUDA_VISIBLE_DEVICES=$gpu "$python_bin" -m scripts.grade_refit \
            --config "$fold_dir/config_resolved.yaml" \
            --plan "$fold_dir/plan.pkl" \
            --source-checkpoint "$fold_dir/model_best.ckpt" \
            --split-manifest "$split_manifest" \
            --condition "$condition" --output-dir "$output" >"$fold_dir/grade_overfitting_pilot/${condition}.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    (( status == 0 )) || exit "$status"
done
