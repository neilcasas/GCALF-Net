# GCALF-Net

GCALF-Net is research software for 3D prostate lesion detection in biparametric MRI. It is a
PyTorch/nnDetection fork used to evaluate frequency-filtering and cross-attention changes against
the unmodified PDHD-Net-style baseline. It is not clinically validated and must not be used for
clinical decision-making.

## Project status

Milestone 1 prepares the PI-CAI supervised cohort for nnDetection. The supported lesion target is
**GGG2--5**, not GGG1--5: PI-CAI provides spatial masks for clinically significant lesions only.
ISUP/GGG 0 and 1 cases are retained as zero-instance negatives; they are not foreground classes.
The task therefore has four foreground classifier classes and three input channels:

| nnDetection class | Grade group | Source mask value |
| --- | --- | --- |
| 0 | GGG2 | 2 |
| 1 | GGG3 | 3 |
| 2 | GGG4 | 4 |
| 3 | GGG5 | 5 |

The checked-in code creates and validates the task, but no PI-CAI data, preprocessing plan, or
training result is committed to this repository. M1 is not complete until the commands below run
successfully in the pinned M0 environment.

## Environment

The authoritative runtime is the CUDA 11.3/PyTorch 1.10.1 environment in
[environment.yml](environment.yml). It must successfully import `nndet._C`, `picai_prep`,
`picai_eval`, and `medcam` before data preparation. Set the nnDetection locations outside the
repository:

```bash
export det_data=/path/to/nnDet_raw
export det_models=/path/to/nnDet_models
export OMP_NUM_THREADS=8
export det_num_threads=8
```

Install the pinned auxiliary tools from [requirements-tools.txt](requirements-tools.txt) and the
local package only after a compatible PyTorch build is installed. See
[Phase 0](docs/phases/PHASE_0_environment.md) for the complete environment gate.

## PI-CAI task preparation

The PI-CAI public image release is an MHA archive, so M1 begins with `mha2nnunet`; it does not
convert DICOM. Use the original granular expert masks and the matching official 1,295-case
`picai_nnunet` split from `picai_baseline`:

```bash
python -m gcalf_data.prepare_picai build \
  --images-dir /path/to/picai_public_images \
  --labels-root /path/to/picai_labels \
  --splits-json /path/to/picai_baseline/src/picai_baseline/splits/picai_nnunet/splits.json \
  --task-dir "$det_data/Task2201_PICAI_GGG" \
  --work-dir /path/to/picai_m1_work

nndet_prep Task2201_PICAI_GGG

python -m gcalf_data.prepare_picai install-splits \
  --task-dir "$det_data/Task2201_PICAI_GGG" \
  --preprocessed-dir "$det_data/Task2201_PICAI_GGG/preprocessed"

python -m gcalf_data.sanity_checks \
  --task-dir "$det_data/Task2201_PICAI_GGG" \
  --marksheet /path/to/picai_labels/clinical_information/marksheet.csv \
  --plan-path "$det_data/Task2201_PICAI_GGG/preprocessed/D3V001_3d.pkl" \
  --report-path docs/data_report.md
```

The sanity gate verifies co-registered T2W/ADC/HBV geometry, instance-volume/JSON agreement,
class range, official fold coverage, and patient-level split separation. It also asserts that the
planner derived `in_channels=3` and `classifier_classes=4`; do not hand-edit these plan values.

Generated images, labels, plans, checkpoints, and reports containing local data remain outside the
repository. The task outputs are ignored through `det_data/`, `det_models/`, and model-artifact
rules in [.gitignore](.gitignore).

## Verification

Run the focused code checks before using data:

```bash
python -m pytest -q tests/gcalf/test_data_pipeline.py
python -m pytest -q tests/test_imports.py tests/test_encoder_cpu.py
```

The full milestone definition, data limitations, and exit criteria are in
[Phase 1](docs/phases/PHASE_1_data_pipeline.md). The roadmap is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Attribution

GCALF-Net builds on nnDetection, PDHD-Net, PI-CAI, and the PI-CAI preprocessing tools. Cite the
corresponding upstream work when using this repository or a derived result.
