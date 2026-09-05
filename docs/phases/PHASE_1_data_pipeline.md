# Phase 1 — Data Pipeline (PI-CAI → nnDetection, csPCa detection + masked GGG2–5 grade metadata)

**Milestone:** M1 · **Depends on:** Phase 0 · **Blocks:** all training

## Goal

Build one nnDetection task where **all 1,500 cases** train csPCa detection, and **grade-resolved
lesions only** carry GGG2–5 metadata for the separate grade head (`ARCHITECTURE.md §3, §8`;
ADR 0002 D2). This replaces the native-4-class design of commits `cf3390b`/`1b18cab` — that design
required dropping every ungraded positive lesion (all 205 Pooch25 cases) to have a class to assign
it to. Nothing here invents a GGG1 or benign foreground class: PI-CAI's own reference standard
encodes ISUP ≤1 as background, and this task follows that standard.

## Known implementation gap

`gcalf_data/build_labels.py`, `gcalf_data/prepare_picai.py`, and `gcalf_data/sanity_checks.py`
currently implement the rejected native-4-class contract: `remap_source_label` maps mask values
`{2,3,4,5}` to detection classes `{0,1,2,3}` and rejects anything else, `dataset_json` declares
four foreground labels, and the sanity checks assert `classifier_classes == 4`. `tests/gcalf/
test_data_preparation.py` and `test_data_sanity_checks.py` test exactly that contract. **All of
this needs real rework**, not a patch:
- `remap_source_label` becomes a **grade lookup**, not a detection-class remap: mask value → GGG
  metadata field, independent of the (now-single) detection class `0`.
- `dataset_json` declares one label, `{"0": "csPCa"}`.
- The label writer must accept `Pooch25`'s binary masks (value `1` = positive, ungraded) as valid
  detection-positive input, which the current `SOURCE_LABELS = {0,2,3,4,5}` check rejects outright.
- Sanity checks assert `classifier_classes == 1`, plus the new grade-metadata invariants below.

This phase closes when the reworked pipeline passes its gates — not when the existing (wrong-
contract) code happens to run.

## 1.1 Unifocal linkage recovery audit (run first — it decides the endpoint)

Before building the full task, run a connected-component audit over all 425 positive masks:

1. For every case with a `Pooch25` (binary) mask: count connected components in the mask, and
   count comma-separated entries in that case's `marksheet.csv` `lesion_ISUP` field.
2. **If both counts are exactly 1**, the single component's grade is that lesion's `lesion_ISUP`
   value, unambiguously — record it as `grade_supervised: true`, `grade_source: "audit_unifocal"`.
3. **Any other case (multiple components, or multiple marksheet lesions) is left untouched** —
   `grade_supervised: false`. Do not infer, pool, or split grades across components.
4. Report the recovered count per grade (starting floor: GGG2 135, GGG3 52, GGG4 20, GGG5 18 from
   `human_expert/original` alone) in `docs/data_report.md`.

This audit requires SimpleITK/numpy connected-component labeling (`scipy.ndimage.label` or
`SimpleITK.ConnectedComponent`), unavailable in a bare-Python read-only pass — run it inside
`gcalf:m0`, not as a doc exercise.

**Only after this audit's numbers are in** does Phase 2's evaluation design freeze between plain
4-class reporting and a pre-registered GGG4+5 merged secondary analysis (ADR 0002 D8).

## 1.2 Pipeline

1. Pass the PI-CAI public MHA archive and `picai_labels` checkout to a rebuilt
   `gcalf_data.prepare_picai build`, with the official `picai_nnunet/splits.json`.
2. `picai_prep` creates the three-channel nnU-Net data (`_0000`=T2W, `_0001`=ADC, `_0002`=HBV —
   assert this order). Apply the preprocessing contract from `ARCHITECTURE.md §3`:
   - N4 bias correction on **T2W only**.
   - Resample T2W/ADC/HBV onto a common reference grid.
   - In-plane center-crop 640→256 using the whole-gland mask
     (`picai_labels/anatomical_delineations/whole_gland/`, present for all 1,500 cases) to center it.
   - Slice axis resampled to a **fixed 3.0 mm spacing**, then pad/crop to 32 slices.
   - Per-case, per-modality z-score normalization.
   - **Masks: nearest-neighbour interpolation only, never linear** — verify this explicitly for
     every mask-resampling call; a linear-interpolated `{0,2,3,4,5}`-valued mask silently
     fabricates nonexistent grade values.
3. `build_labels.py` (reworked): every positive lesion gets detection class `0`. Grade-resolved
   lesions (220 `human_expert` + audit-recovered `Pooch25`, §1.1) additionally get `grade ∈
   {2,3,4,5}`, `grade_source`, `grade_supervised: true`. Every other positive lesion gets
   `grade_supervised: false` and no grade field. Benign/GGG1 cases get `"instances": {}`.
4. `nnunet2nndet` → nnDetection task layout. **The task's own geometry (step 2) is the raw input
   — do not hand-plan spacing/patch size a second time.** Run `nndet_prep Task2201_PICAI_csPCa`;
   let its planner derive spacing and patch size, and record the resolved plan values (not your
   input geometry) in the dataset manifest.
5. `prepare_picai install-splits` writes the official folds to `preprocessed/splits_final.pkl`.
6. `gcalf_data.sanity_checks --report-path docs/data_report.md`.

## 1.3 Definition of done

- **L:** `Task2201_PICAI_csPCa/raw_splitted/` has `_0000/_0001/_0002` images and paired instance
  volumes/JSON files for all 1,500 cases.
- **L:** `dataset.json["labels"] == {"0": "csPCa"}`; the planner derives `classifier_classes == 1`
  (nnDetection's own anchor head) and `in_channels == 3`.
- **L:** every instance's detection class is `0`; every instance with `grade_supervised: true` has
  `grade ∈ {2,3,4,5}` matching its mask provenance; instance-volume IDs exactly match `case.json` keys.
- **L:** official folds cover all 1,500 cases, no patient-level leakage, and every held-out fold
  contains at least one grade-supervised lesion of every grade 2–5.
- **L:** `docs/data_report.md` is regenerated from a real run and states the final
  grade-supervised lesion count per grade (§1.1's audit result), the case-level `case_ISUP`
  distribution, and this phase's cohort limitations (below).
- Sanity checks and planner assertions all pass.

## 1.4 Cohort limitations (record in the generated data report, not silently)

- Grade supervision covers 220 baseline + audit-recovered lesions out of 425 positive cases —
  never the full detection-training cohort. State this explicitly wherever weighted F1 or the
  confusion matrix is reported; do not let a reader infer the grade head trained on 425 or 1,500.
- GGG4 (20) and GGG5 (18) are small even before folding — roughly 4 held-out cases per fold each
  at the baseline floor. Report per-grade counts and bootstrap CIs everywhere (Phase 6).
- The multi-component, multi-lesion Pooch25 cases that the audit could not resolve remain
  detection-positive but grade-unsupervised. Do not revisit this rule under schedule pressure —
  it is the boundary between recovered-and-defensible and inferred-and-not.
