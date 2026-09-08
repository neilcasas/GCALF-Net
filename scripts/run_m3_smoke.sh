#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <source-m1-task-dir> <new-tiny-task-dir> [--resume]" >&2
    exit 2
fi

source_task_dir=$1
tiny_task_dir=$2
resume=${3:-}
if [[ -n "$resume" && "$resume" != "--resume" ]]; then
    echo "Unknown option: $resume" >&2
    exit 2
fi
task_name=$(basename "$tiny_task_dir")
model_name=RetinaUNetV001_D3V001_3d
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
train_config=${GCALF_TRAIN_CONFIG:-gcalf_smoke}

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
if [[ -e "$tiny_task_dir" && -z "$resume" ]]; then
    echo "Refusing to reuse an existing tiny task: $tiny_task_dir (pass --resume only for an interrupted M3 task)" >&2
    exit 1
fi
if [[ -n "$resume" && ! -f "$tiny_task_dir/dataset.json" ]]; then
    echo "Cannot resume: tiny task metadata is missing: $tiny_task_dir/dataset.json" >&2
    exit 1
fi
if [[ -e "$det_models/$task_name/$model_name" ]]; then
    echo "Refusing to reuse existing M3 model output: $det_models/$task_name/$model_name" >&2
    exit 1
fi

cd "$repo_dir"
python -c 'import torch; assert torch.cuda.is_available(), "M3 requires a CUDA GPU"'
if [[ -z "$resume" ]]; then
    python -m gcalf_data.prepare_picai tiny --source-task-dir "$source_task_dir" --task-dir "$tiny_task_dir"
fi
# The tiny deterministic gate has too little work to amortize batchgenerator
# workers, and forked augmentation can deadlock during Lightning's validation
# sanity check on this pinned runtime.
overrides=("train=$train_config" augment_cfg.multiprocessing=false)
if [[ -n "${GCALF_FUSION_LEVELS:-}" ]]; then
    [[ "$GCALF_FUSION_LEVELS" =~ ^[0-4](,[0-4])*$ ]] || {
        echo "GCALF_FUSION_LEVELS must be a comma-separated list of levels 0-4" >&2
        exit 2
    }
    overrides+=("model_cfg.encoder_kwargs.gcalf_cfg.fusion_levels=[$GCALF_FUSION_LEVELS]")
fi
python scripts/preprocess.py "$task_name" -o "${overrides[@]}"
python -c '
import pickle
import sys
with open(sys.argv[1], "rb") as file:
    architecture = pickle.load(file)["architecture"]
assert architecture["in_channels"] == 3, architecture
assert architecture["classifier_classes"] == 1, architecture
assert len(architecture["conv_kernels"]) == 5, architecture
assert len(architecture["strides"]) == 4, architecture
' "$tiny_task_dir/preprocessed/D3V001_3d.pkl"
python scripts/unpack_preprocessed.py "$tiny_task_dir/preprocessed/D3V001_3d/imagesTr" \
    --processes "${GCALF_UNPACK_PROCESSES:-8}"
python scripts/train.py "$task_name" --sweep -o "${overrides[@]}"
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

import torch
state = torch.load(training_dir / "model_last.ckpt", map_location="cpu")["state_dict"]
assert any(key.startswith("model.grade_head.") for key in state), "grade head was not checkpointed"
grade_classifier_weights = [value for key, value in state.items() if key.endswith("grade_head.classifier.weight")]
assert len(grade_classifier_weights) == 1 and grade_classifier_weights[0].shape[0] == 4, "grade head must emit GGG2-5 logits"

with open(sys.argv[2], newline="") as file:
    rows = list(csv.DictReader(file))
assert len(rows) == 1, rows
assert all(math.isfinite(float(rows[0][field])) for field in ("picai_score", "auroc", "lesion_ap")), rows
' "$training_dir" "$metrics_path"
