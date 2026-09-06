# Phase 1 — Data Pipeline (PI-CAI → nnDetection, csPCa detection + masked GGG2–5 grade metadata)

**Milestone:** M1 · **Depends on:** Phase 0 · **Blocks:** all training

## Goal

Build one nnDetection task from the **1,500-case source cohort**. One declared source exclusion leaves 1,499 retained cases, where every retained case trains
csPCa detection and **grade-resolved lesions only** carry GGG2–5 metadata for the separate grade
head (`ARCHITECTURE.md §3, §8`;
ADR 0002 D2). This replaces the native-4-class design of commits `cf3390b`/`1b18cab` — that design
required dropping every ungraded positive lesion (all 205 Pooch25 cases) to have a class to assign
it to. Nothing here invents a GGG1 or benign foreground class: PI-CAI's own reference standard
encodes ISUP ≤1 as background, and this task follows that standard.

## Current implementation status

The single-class detection and masked GGG2–5 metadata contract is implemented. The remaining M1
gate is a clean rebuild with the new post-crop audit: `gcalf_data.audit_crop_retention.py` captures
source/resampled/in-plane/final voxel and component counts, and `prepare_picai build` excludes a
source-positive case whose target-independent crop retains no lesion voxels. Such a case must never
remain as an empty-label benign case in any fold.

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
2. Build the three-channel nnU-Net data (`_0000`=T2W, `_0001`=ADC, `_0002`=HBV — assert this
   order). Apply the preprocessing contract from `ARCHITECTURE.md §3` identically for training,
   validation, test, and deployment:
   - N4 bias correction on **T2W only**.
   - Build one reference grid from the corrected T2W: native in-plane geometry at **3.0 mm slice
     spacing**. Resample ADC/HBV onto it with **linear** interpolation (not B-spline — it overshoots
     and can emit negative ADC) and every mask with nearest-neighbour, in a single pass.
   - Validate the whole-gland mask, then in-plane centre-crop to a **fixed 128 mm field of view**
     using only its centroid. Specify it in millimetres, not voxels: T2W in-plane spacing runs
     0.234–0.625 mm across the archive, so a 256-voxel window spans 60–160 mm depending on scanner
     (`ARCHITECTURE.md §3`). Voxel extent therefore varies per case; `nndet_prep` resamples in-plane
     anyway. An empty/implausible gland uses the predeclared T2W geometric-centre fallback and is
     logged.
   - Do **not** pass the lesion mask into the crop-centre resolver. No union or lesion-only fallback
     is allowed; lesion annotations are unavailable at inference.
   - Geometrically pad/crop the slice axis to 32 slices, padding with zeros. Preserve the current
     geometric depth rule while its full-cohort audit shows no added clipping; any future gland-z
     rule requires re-auditing all positives.
   - Do not z-score here. Let nnDetection apply its `nonCT` per-case/per-modality normalization once
     after planned resampling.
   - **Masks: nearest-neighbour interpolation only, never linear** — verify this explicitly for
     every mask-resampling call; a linear-interpolated `{0,2,3,4,5}`-valued mask silently
     fabricates nonexistent grade values.
3. `build_labels.py` (reworked): every positive lesion gets detection class `0`. Grade-resolved
   lesions (220 `human_expert` + audit-recovered `Pooch25`, §1.1) additionally get `grade ∈
   {2,3,4,5}`, `grade_source`, `grade_supervised: true`. Every other positive lesion gets
   `grade_supervised: false` and no grade field. Benign/GGG1 cases get `"instances": {}`.
4. After the crop, audit all 425 source-positive cases. Record source/resampled/in-plane/final
   lesion voxel and connected-component counts. A non-empty mask that becomes empty is excluded
   from the raw task and every fold under ADR 0002 D6 item 7; it is never relabelled as benign.
   Partial clips remain reported. Ground-truth-guided recentering is never a remedy.
5. `nnunet2nndet` → nnDetection task layout. **The task's own geometry (step 2) is the raw input
   — do not hand-plan spacing/patch size a second time.** Run `nndet_prep Task2201_PICAI_csPCa`;
   let its planner derive spacing and patch size, and record the resolved plan values (not your
   input geometry) in the dataset manifest.
   Record the planner's resolved `nonCT` normalization, target spacing, patch size, level count,
   per-level strides/kernels/channels, and decoder inputs. **Assert, do not merely record**, that
   `len(conv_kernels) == 5`: the count is planner-derived from patch size and spacing
   (`nndet/planning/architecture/boxes/c002.py:196-204`). If it resolves to six, stop and resolve it
   once at the planning level — adjust the raw-task geometry or pin `conv_kernels`/`strides` in the
   plan, record which in the manifest, and use that plan for all four arms. Never patch it per run.
   Note also that `nndet_prep` runs `crop_to_nonzero` (`nndet/io/crop.py:288`) first, so the padded
   raw-task array is not what the planner sees.
6. `prepare_picai install-splits` writes the official folds to `preprocessed/splits_final.pkl`.
7. `gcalf_data.sanity_checks --report-path docs/data_report.md`.

### 1.2.1 Crop-QC evidence and required disposition

**The audit is now code.** `gcalf_data/audit_crop_retention.py` emits the generated CSV and
exclusion manifest into the ignored task directory, while `sanity_checks` validates both and emits
the exception table into the generated report. The 128 mm audit still needs to be run on a clean
rebuilt task before M1 can close.

The 2026-09-05 read-only audit used the actual 1,500 T2W scans, Bosma22b glands, 220 non-empty
`human_expert/resampled` masks, and 205 Pooch25 masks. All Bosma22b gland masks were non-empty.
**It was run under the superseded 256-voxel crop rule.** 421/425 positive masks were fully
retained; three were partially clipped (`11174_1001197`: 40/32,365 voxels; `11280_1001303`:
58/6,406; `10956_1000975`: 469/107,178), and `11050_1001070` lost all 3,472 lesion voxels —
Guerbet23 lost it too. No additional depth clipping was observed among the 424 lesions that
survived the in-plane crop.

All four exceptions are fine-spacing scans, and none is at 0.5 mm or coarser:

| Case | In-plane spacing | FOV at 256 voxels | Outcome |
|---|---|---|---|
| `11280_1001303` | 0.234 mm | 60.0 mm | partial clip |
| `11050_1001070` | 0.281 mm | 72.0 mm | total loss |
| `10956_1000975` | 0.300 mm | 76.8 mm | partial clip |
| `11174_1001197` | 0.342 mm | 87.6 mm | partial clip |
| *modal case* | 0.500 mm | 128.0 mm | retained |

So they are a consequence of the crop rule, not four independent data defects. **Re-run the audit
under the 128 mm FOV before M1 closes** and report the result; the expectation is that the three
partial clips resolve. `11050_1001070` resolves only if its lesion lies within 64 mm of the gland
centroid — if it does not, that case is a genuine gland/lesion provenance conflict and D6 item 7's
predeclared exclusion rule disposes of it. Do not encode any known lesion location into
preprocessing to rescue a case.

## 1.3 Definition of done

- **L:** `Task2201_PICAI_csPCa/raw_splitted/` has `_0000/_0001/_0002` images and paired instance
  volumes/JSON files for every retained source case; the source cohort has 1,500 cases, the retained task has 1,499 cases, and any
  full-loss positive is named in `excluded_cases.json` and absent from every fold.
- **L:** `dataset.json["labels"] == {"0": "csPCa"}`; the planner derives `classifier_classes == 1`
  (nnDetection's own anchor head) and `in_channels == 3`.
- **L:** every instance's detection class is `0`; every instance with `grade_supervised: true` has
  `grade ∈ {2,3,4,5}` matching its mask provenance; instance-volume IDs exactly match `case.json` keys.
- **L:** retained official folds cover every retained case exactly once in validation, no
  patient-level leakage, and every held-out fold
  contains at least one grade-supervised lesion of every grade 2–5.
- **L:** `docs/data_report.md` is regenerated from a real run and states the final
  grade-supervised lesion count per grade (§1.1's audit result), the case-level `case_ISUP`
  distribution, this phase's cohort limitations (below), gland-mask QC, and the full crop-retention
  audit with the disposition of all four known exceptions.
- **L:** crop-centre calculation accepts no lesion annotation; train/validation/inference use the
  same gland/T2W-only resolver; saved spatial transforms can map predictions back to native space.
- **L:** raw inputs are not already z-scored; the resolved plan uses one `nonCT` normalization
  pass, and the resolved `use_mask_for_norm` value is recorded (the raw task's zero padding decides
  it — `nndet/planning/experiment/base.py:287-312`).
- **L:** the resolved model plan contains exactly five encoder outputs and five decoder inputs, by
  assertion rather than observation. Record the raw-task geometry *and* the planner's resolved
  patch size separately — the raw geometry is not the model input, since `crop_to_nonzero`,
  planner resampling, and patch extraction all sit in between.
- **L:** `gcalf_data/audit_crop_retention.py` is committed, run, and its output referenced by the
  generated data report.
- Sanity checks and planner assertions all pass.

## 1.4 Cohort limitations (record in the generated data report, not silently)

- Grade supervision covers 220 baseline + audit-recovered lesions out of the retained positive
  cohort — never the full detection-training cohort. State this explicitly wherever weighted F1 or
  the confusion matrix is reported.
- GGG4 (20) and GGG5 (18) are small even before folding — roughly 4 held-out cases per fold each
  at the baseline floor. Report per-grade counts and bootstrap CIs everywhere (Phase 6).
- The multi-component, multi-lesion Pooch25 cases that the audit could not resolve remain
  detection-positive but grade-unsupervised. Do not revisit this rule under schedule pressure —
  it is the boundary between recovered-and-defensible and inferred-and-not.
- A prostate-centred crop can expose gland/lesion provenance disagreements; it cannot defensibly
  resolve them using the target mask. Report repairs and exclusions, including their fold, before
  training and apply the same frozen case set to all four ablation arms.
