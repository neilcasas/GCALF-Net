# Phase 1 — Data Pipeline (PI-CAI → nnDetection, GGG labels, splits, sanity)

**Milestone:** M1 · **Depends on:** Phase 0 · **Blocks:** all training
**Goal:** a validated nnDetection task folder with 3-channel bpMRI volumes (T2W/ADC/DWI), per-lesion GGG labels, the official cross-validation splits, and passing sanity checks.

**Definition of done:**
- [ ] `nnDet_raw/Task2xx_PICAI/` populated: `imagesTr/*_0000/_0001/_0002.nii.gz`, `labelsTr/*.nii.gz` + `labelsTr/*.json`, `dataset.json`.
- [ ] `nndet_prep` planning ran → `plan` dict with `in_channels=3`, `classifier_classes=5`, `patch_size`, spacing.
- [ ] `gcalf_data/build_labels.py` maps every lesion to a **0-indexed foreground class** ∈ {0..4} (GGG1→0 … GGG5→4); benign cases get `"instances": {}`.
- [ ] Official 5-fold splits loaded (no patient leakage).
- [ ] `gcalf_data/sanity_checks.py` — **all asserts green**.
- [ ] A one-page `docs/data_report.md` with class counts, spacing, split sizes, and the label decisions below.

---

## 1.1 Ground-truth reality (verified from `picai_labels/clinical_information/marksheet.csv`)

1500 rows. Columns: `patient_id, study_id, mri_date, patient_age, psa, psad, prostate_volume, histopath_type, lesion_GS, lesion_ISUP, case_ISUP, case_csPCa, center`.

```
case_ISUP distribution: {0: 847, 1: 228, 2: 234, 3: 99, 4: 40, 5: 52}
```

**Decisions you must make and record in `data_report.md`:**
1. **ISUP == GGG.** GGG k ↔ ISUP k. Use `lesion_ISUP` for per-lesion GGG (detection model) or `case_ISUP` for per-case (classifier fallback). **nnDetection instance classes are 0-indexed foreground ids**, so the stored class is `lesion_ISUP - 1`. Keep the marksheet value and the stored class distinct in the code and in `data_report.md`; conflating them is the easiest way to shift every grade by one.
2. **Benign handling (847 cases, ISUP 0).** Recommended: keep them as **negative cases with zero instances** (`"instances": {}`, all-zero instance volume) — they supply background anchors and keep FROC meaningful. There is **no background class**: `classifier_classes = 5` and RetinaUNet's sigmoid focal loss treats background implicitly (see `THESIS_PLAN.md §0`). Alternative (document if used): drop benigns entirely (loses 56% of data). **Do not silently drop, and do not invent a class 0 = benign.**
3. **Severe imbalance (GGG4=40, GGG5=52).** Any test fold has <10 GGG4/5 cases → drives class weighting (Phase 2/6) and metric choice (macro/balanced, CIs). Flag as a study limitation.
4. **Per-lesion vs per-case.** A case can have multiple lesions at different grades; `case_ISUP` = max. nnDetection predicts per-lesion → GGG is per-lesion. `lesion_ISUP` is blank for benign rows → handle NaN in `build_labels.py`.
5. **Mask provenance.** `csPCa_lesion_delineations/` mixes human and AI-derived masks — prefer human subfolders; record which cases use which.

## 1.2 Preprocessing steps (follow `picai_baseline/nndetection_baseline.md`; cross-check `Z-SSMNet/…/prepare_data.py`)

Wrapper: `gcalf_data/prepare_picai.py` orchestrates these; don't run ad-hoc.

1. **Download** PI-CAI images (public training set, ~100+ GB) + this `picai_labels` repo.
2. **`dcm2mha`** (`picai_prep`): DICOM → per-sequence MHA.
3. **`mha2nnunet`** (`picai_prep`): resample + co-register + stack into nnUNet layout. Produces 3-channel volumes with **fixed channel order by suffix**: `_0000`=T2W, `_0001`=ADC, `_0002`=high-b DWI. Spacing is resampled (commonly ≈ `3.0 × 0.5 × 0.5` mm — **read the actual value from the generated plan, don't hard-code**).
4. **`build_labels.py`**: join `marksheet.csv` `lesion_ISUP` to lesion delineations, write per-voxel label maps with GGG class ids.
5. **`nnunet2nndet`** (`picai_prep`): nnUNet task → nnDetection task (derives per-instance boxes from lesion masks). Also see `scripts/convert_seg2det.py`, `convert_cls2fg.py`.
6. **`nndet_prep`** (nnDetection): fingerprint + planning → `plan` dict (`patch_size`, spacing, anchors, `in_channels=3` from the 3 modalities, `classifier_classes=5` from `dataset.json["labels"]` — both auto-derived at `nndet/planning/architecture/boxes/base.py:89-95`; assert, don't hand-edit).
7. **Splits**: use official PI-CAI 5-fold CV, present in `Z-SSMNet/src/z_ssmnet/splits/picai/` and `picai_baseline`. Copy into the nnDetection task; verify split membership by `patient_id`.

## 1.3 `build_labels.py` — label construction

nnDetection's per-case label is **two files**: `labelsTr/<case>.nii.gz`, an **instance-ID** volume with voxels `1..N` (not class ids), and `labelsTr/<case>.json` = `{"instances": {"1": <class>, "2": <class>, …}}` where `<class>` is a 0-indexed foreground id. `picai_prep`'s `nnunet2nndet` emits this structure; `build_labels.py` sets the class values.

- Read `marksheet.csv`; index by `(patient_id, study_id)`.
- For each case's lesion mask, take connected components as instances, number them `1..N` in the volume, and set `instances[str(i)] = lesion_ISUP_i - 1` (fall back to `case_ISUP - 1` if only case-level grading is available).
- Benign (ISUP 0, empty mask) → all-zero instance volume + `"instances": {}`.
- Emit `dataset.json` `"labels" = {"0":"GGG1","1":"GGG2","2":"GGG3","3":"GGG4","4":"GGG5"}` (exactly PDHD-Net's documented task) → the planner derives `classifier_classes = 5`.
- **Assert:** every instance class ∈ {0..4}; the set of non-zero voxel ids equals the set of `instances` keys; per-class lesion counts match the marksheet after the `-1` shift; no case has an instance id present in one file but not the other.

## 1.4 Sanity checks (`gcalf_data/sanity_checks.py`, assert-based, runnable)

Load ~10 random cases + the tiny-subset cases and assert:
1. **Spacing** identical across `_0000/_0001/_0002` per case (co-registered).
2. **Orientation**: consistent SITK direction / affine sign across all cases (uniform RAS or LPS).
3. **Channel order**: `_0000` intensity profile looks like T2W (anatomy), `_0001` like ADC (values ~0–3000, low in tumor), `_0002` like high-b DWI (bright in tumor). Print per-channel min/mean/max for eyeballing.
4. **Label domain**: instance-ID volume contains only `0..N`; every `instances` value ∈ {0..4}; ids in the volume and the JSON agree; per-class lesion counts printed and reconciled with the marksheet.
5. **Split integrity**: train/val/test folds are disjoint by `patient_id` (no patient in two folds); print fold sizes and per-fold GGG distribution.
6. **Image/label shape match** per case.

Ship this as the phase's gate — it must be green before Phase 2.

## 1.5 Risks & fallbacks (this phase)

| Risk | Fallback |
|---|---|
| `picai_prep` step fails / opaque error | Read `Z-SSMNet/src/z_ssmnet/prepare_data.py` for the working call sequence + params; it's a tested PI-CAI submission. |
| Disk blows up (raw DICOM huge) | Preprocess once; keep only the compact resampled nnDetection task (`.nii.gz`); delete intermediate MHA. |
| Lesion mask provenance unclear | Prefer human delineations; log provenance per case; treat AI-derived as a documented caveat. |
| GGG4/5 too few for a fold to contain any | Use stratified fold assignment for reporting; consider grouped GGG4+5 sensitivity analysis (document). |

## 1.6 Deliverables & commit

- Branch `feat/env-and-data` (same as Phase 0) → commit "data: PI-CAI nnDetection task + GGG labels + splits + sanity checks".
- Files: `gcalf_data/{prepare_picai,build_labels,sanity_checks}.py`, `docs/data_report.md`.
- **Also build `Task9xx_PICAI_TINY`** here (4–6 cases spanning GGG grades + 1 benign) — Phase 2 needs it.

**Next:** `PHASE_2_baseline.md`.
