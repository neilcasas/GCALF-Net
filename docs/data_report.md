# PI-CAI M1 data report

- Cohort: 424 retained positive cases across 1,499 retained cases train csPCa detection (single foreground class).
- Source-positive cases audited: 425; excluded after target-independent crop: 1 (`11050_1001070`).
- Grade supervision: only grade-resolved lesions (human-expert masks directly, plus D3 rev. 2 audit-recovered homogeneous Pooch25 cases) train the separate GGG2--5 grade head; this is 441 of 458 positive lesions, never the full detection-training cohort.
- ISUP 0 and 1 cases remain zero-instance negatives; no benign or GGG1 foreground class is created.
- Modalities: T2W, ADC, HBV/high-b DWI (`_0000`, `_0001`, `_0002`).
- Gland-centred crop fallbacks: 0 (all non-gland strategies are listed in `crop_strategy_exceptions.json`).

## Grade-supervised instance counts

- GGG2: 253
- GGG3: 104
- GGG4: 37
- GGG5: 47
- Ungraded positive instances: 17 (the 13 heterogeneous Pooch25 cases contain 17 retained components and remain latent).

## Marksheet case ISUP counts

- ISUP 0: 847
- ISUP 1: 228
- ISUP 2: 234
- ISUP 3: 99
- ISUP 4: 40
- ISUP 5: 52

## Crop-retention audit

- excluded_no_retained_voxels: 1
- fully_retained: 424
- negative: 1,075

| Case | Status | Source voxels | Final voxels | Source components | Final components |
|---|---|---:|---:|---:|---:|
| 11050_1001070 | excluded_no_retained_voxels | 3,472 | 0 | 1 | 0 |

## Official fold sizes

- Fold 0: train=1,199, val=300
- Fold 1: train=1,201, val=298
- Fold 2: train=1,197, val=302
- Fold 3: train=1,196, val=303
- Fold 4: train=1,203, val=296

## Cohort limitations

- Grade supervision covers 441 of 458 retained positive instances, not the 1,499-case detection cohort. State both denominators with every grade metric.
- GGG4 and GGG5 remain small after recovery; report per-grade bootstrap confidence intervals and retain GGG4+5 only as a secondary analysis.
- The 13 Pooch25 cases with heterogeneous valid GGG2--5 marksheet grades remain detection-positive but grade-unsupervised. This is deliberate: no component linkage, largest-component choice, weak bag loss, or latent grade assignment fabricates their labels.
