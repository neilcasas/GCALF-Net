#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <source-m1-task-dir> <new-tiny-task-dir>" >&2
    exit 2
fi

source_task_dir=$1
tiny_task_dir=$2
task_name=$(basename "$tiny_task_dir")
model_name=RetinaUNetV001_D3V001_3d
repo_dir=$(cd "$(dirname "$0")/.." && pwd)

: "${det_data:?det_data must be set}"
: "${det_models:?det_models must be set}"
: "${OMP_NUM_THREADS:?OMP_NUM_THREADS must be set}"

if [[ "$(dirname "$tiny_task_dir")" != "$det_data" ]]; then
    echo "Tiny task must be a direct child of det_data: $det_data/$task_name" >&2
    exit 1
fi
if [[ ! -d "$source_task_dir" ]]; then
    echo "Source M1 task does not exist: $source_task_dir" >&2
    exit 1
fi
if [[ -e "$tiny_task_dir" ]]; then
    echo "Refusing to reuse an existing tiny task: $tiny_task_dir" >&2
    exit 1
fi
if [[ -e "$det_models/$task_name/$model_name" ]]; then
    echo "Refusing to reuse existing M3 model output: $det_models/$task_name/$model_name" >&2
    exit 1
fi

cd "$repo_dir"
python -c 'import torch; assert torch.cuda.is_available(), "M3 requires a CUDA GPU"'
python -m gcalf_data.prepare_picai tiny --source-task-dir "$source_task_dir" --task-dir "$tiny_task_dir"
python scripts/preprocess.py "$task_name"
python -c '
import pickle
import sys
with open(sys.argv[1], "rb") as file:
    architecture = pickle.load(file)["architecture"]
assert architecture["in_channels"] == 3, architecture
assert architecture["classifier_classes"] == 4, architecture
' "$tiny_task_dir/preprocessed/D3V001_3d.pkl"
python scripts/train.py "$task_name" --sweep -o train=smoke
python scripts/predict.py "$task_name" "$model_name" -f 0
python -m gcalf_eval.run_eval --task "$task_name" --model "$model_name" --fold 0 --split test

training_dir="$det_models/$task_name/$model_name/fold0"
metrics_path="$training_dir/test_results/picai/metrics.csv"
python -c '
import csv
import math
import sys
from pathlib import Path

training_dir = Path(sys.argv[1])
for artifact in ("model_best.ckpt", "model_last.ckpt", "plan_inference.pkl"):
    assert (training_dir / artifact).is_file(), training_dir / artifact

with open(sys.argv[2], newline="") as file:
    rows = list(csv.DictReader(file))
assert len(rows) == 1, rows
assert all(math.isfinite(float(rows[0][field])) for field in ("picai_score", "auroc", "lesion_ap")), rows
' "$training_dir" "$metrics_path"
