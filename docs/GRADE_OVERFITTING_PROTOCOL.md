# Grade overfitting correction protocol

This protocol separates the grade and detection decisions. It does not alter
GGG2--5 labels, patient splits, or the detection endpoint.

## 1. Preserve and pause the active Fold-1 run

Run this **on the Vast instance** after finding the PID of each trainer. The
source directories must be the exact model/evidence trees to preserve. The
helper takes a pre-interrupt snapshot, sends SIGINT, waits for a graceful exit,
then writes a final snapshot and `SHA256SUMS`. It refuses an existing backup
directory and never deletes a source.

```bash
ps -eo pid,args | rg 'scripts/train.py.*exp.fold=1'
cloud/vast/pause_and_backup.sh \
  --pid "$TRAINER_PID" \
  --source-dir "$det_models/Task2201_PICAI_csPCa/RetinaUNetV001_D3V001_3d" \
  --source-dir /workspace/evidence \
  --backup-dir "/workspace/backups/fold1-$(date +%Y%m%dT%H%M%S)"
```

For four independent arms, pass the four trainer PIDs as repeated `--pid`
arguments to one invocation so they share one pre-interrupt snapshot. Verify
the final snapshot before stopping the Vast instance:

```bash
(cd "$BACKUP_DIR" && sha256sum -c SHA256SUMS)
```

## 2. Evaluate each existing Fold-0 checkpoint at lesion level

`nndet_sweep` now accepts an exact checkpoint identifier and an output
directory. Passing `model_best-v1` resolves exactly
`model_best-v1.ckpt`, rather than ambiguously matching `model_best.ckpt` too.
It writes new predictions below the supplied directory and leaves the original
training run untouched.

```bash
RUN="$det_models/Task2201_PICAI_csPCa/RetinaUNetV001_D3V001_3d/fold0"
export RUN
for checkpoint in model_best model_best-v1 model_last; do
  nndet_sweep Task2201_PICAI_csPCa RetinaUNetV001_D3V001_3d 0 \
    --checkpoint "$checkpoint" \
    --output-dir "$RUN/checkpoint_evaluations/$checkpoint"
done

python - <<'PY'
import os
from pathlib import Path
from nndet.io.load import load_pickle

run = Path(os.environ["RUN"])
case_ids = sorted(load_pickle(run / "splits.pkl")[0]["val"])
(run / "fold0_val_case_ids.txt").write_text("\n".join(case_ids) + "\n")
PY

for checkpoint in model_best model_best-v1 model_last; do
  python -m gcalf_eval.run_eval \
    --task Task2201_PICAI_csPCa --model RetinaUNetV001_D3V001_3d --fold 0 --split val \
    --prediction-dir "$RUN/checkpoint_evaluations/$checkpoint/val_predictions" \
    --ground-truth-dir "$det_data/Task2201_PICAI_csPCa/raw_splitted/labelsTr" \
    --case-ids-file "$RUN/fold0_val_case_ids.txt" \
    --output-dir "$RUN/checkpoint_evaluations/$checkpoint/picai"
done
```

For baseline, `model_best` is the retained early epoch-5 checkpoint,
`model_best-v1` is the detection-best epoch-57 checkpoint, and `model_last`
is the epoch-60 checkpoint. Confirm the epoch in each checkpoint before
labelling results. The grade reports contain matched-lesion accuracy,
macro-F1, balanced accuracy, confusion matrix, per-class sensitivity,
one-vs-rest AUROC, multiclass Brier score, ECE, and confidence split by
correct/incorrect classification. Missed grade-supervised lesions remain a
separate detection failure count.

## 3. Corrected future runs

All GCALF arm configurations now save both:

- `model_best*.ckpt`: detection mAP selection (the existing behavior);
- `model_best_grade*.ckpt`: lowest masked validation grade cross-entropy.

Run the short baseline pilot before choosing further regularization, sampling,
or staged/frozen-head changes:

```bash
CUDA_VISIBLE_DEVICES=0 nndet_train Task2201_PICAI_csPCa \
  -o train=gcalf_grade_pilot exp.fold=0 exp.seed=2026
```

The pilot lasts 12 main epochs with no SWA and uses the same split and data
sampling as the prior run. It is not a replacement for a finalized fold. Use
its saved grade checkpoint and lesion-level report to choose the final,
predeclared regularization/staging protocol, then rerun all five folds under
that one protocol. Do not pool the old Fold-0 result with corrected folds.
