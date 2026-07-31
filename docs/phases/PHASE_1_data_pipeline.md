# Phase 1 — Data Pipeline (PI-CAI → nnDetection, GGG2--5 labels, splits, sanity)

**Milestone:** M1 · **Depends on:** Phase 0 · **Blocks:** all training

## Goal

Create a validated nnDetection task from PI-CAI's 1,295-case supervised granular-annotation
cohort. It contains co-registered T2W/ADC/HBV volumes and per-lesion GGG2--5 labels.

PI-CAI does not supply spatial GGG1 masks: all ISUP 0/1 tissue is encoded as background. Therefore
M1 uses four foreground classes (`GGG2`--`GGG5`) and keeps ISUP 0/1 cases as zero-instance negatives.
Do not invent a benign or GGG1 foreground class.

## Pipeline

1. Pass the PI-CAI public **MHA** archive and `picai_labels` checkout to
   `python -m gcalf_data.prepare_picai build` with the official `picai_nnunet/splits.json`.
2. `picai_prep` creates the three-channel nnU-Net data; `build_labels.py` remaps source values
   `2..5` to contiguous semantic labels `1..4` before `nnunet2nndet` writes nnDetection instances
   with class IDs `0..3`.
3. Run `nndet_prep Task2201_PICAI_GGG`; the planner must derive `in_channels=3` and
   `classifier_classes=4`.
4. Run `prepare_picai install-splits` to write the official folds to
   `preprocessed/splits_final.pkl`.
5. Run `python -m gcalf_data.sanity_checks` with `--report-path docs/data_report.md`.

## Definition of done

- `Task2201_PICAI_GGG/raw_splitted/` has `_0000` T2W, `_0001` ADC, `_0002` HBV images and paired
  instance volumes/JSON files.
- `dataset.json` contains exactly four labels: `GGG2`, `GGG3`, `GGG4`, and `GGG5`.
- Every instance JSON class is in `{0,1,2,3}` and exactly matches the IDs in its volume.
- Official folds cover the same 1,295 cases as the task and have no patient-level train/validation
  leakage.
- The sanity command and planner assertions pass, and `docs/data_report.md` is regenerated from
  that successful run.

## Cohort limitation

The newer Pooch25 masks cover the remaining 205 positive cases but are binary, so they cannot be
joined to lesion-level GGG classes without an additional, explicitly approved grade-assignment
policy. They are out of scope for this supervised granular M1 task. Record this limitation in the
generated data report; do not silently mix binary and granular masks.
