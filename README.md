# GCALF-Net

GCALF-Net is research software for 3D prostate lesion detection and Gleason Grade Group (GGG)
grading in biparametric MRI. It is a PyTorch/nnDetection fork used to evaluate a built
frequency-filtering module (FDR/LFF) and a built cross-attention fusion module (WAF/CAF) against a
constructed baseline — not against the unmodified released PDHD-Net code. It is not clinically
validated and must not be used for clinical decision-making.

## Project status

The supported protocol is **GGG2–5 lesion-level grading**, not GGG1–5: PI-CAI encodes ISUP ≤1
(benign and GGG1) as spatial background, so no GGG1 mask exists to train on. The model has two
heads:

- **One detection class, `csPCa`** (clinically significant PCa, ISUP ≥2), trained on 1,499 retained
  cases — 424 positives (any lesion with a positive spatial mask, graded or not) and 1,075
  negatives (benign + GGG1). The source cohort has one declared exclusion.
- **A separate 4-logit grade head**, trained only on lesions with a resolved GGG2–5 grade
  (masked elsewhere). Of the 424 retained positives, 340 carry grade supervision. The remaining
  118 positive instances are grade-unsupervised.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and
[`docs/adr/0002-ggg2-5-masked-grade-head-and-built-fdr-waf.md`](docs/adr/0002-ggg2-5-masked-grade-head-and-built-fdr-waf.md)
for why it is shaped this way.

**Implementation gap, stated plainly:** `gcalf_data/{build_labels,prepare_picai,sanity_checks}.py`
and `tests/gcalf/{test_data_preparation,test_data_sanity_checks}.py` currently implement an
earlier, rejected native-4-class design (one detection class per grade, `classifier_classes=4`,
no separate grade head). They have not yet been reworked for the two-head contract above —
[Phase 1](docs/phases/PHASE_1_data_pipeline.md) tracks that work. Do not treat their current
passing tests as evidence the described protocol is implemented.

Neither the frequency module (FDR) nor the fusion module (WAF) this thesis's baseline requires
exists in the released code today — both must be built ([Phase 2](docs/phases/PHASE_2_baseline.md)).

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
[Phase 0](docs/phases/PHASE_0_environment.md) for the complete environment gate, including why
local development is CPU-only and where a separate modern-CUDA scratch environment fits.

## PI-CAI task preparation

The PI-CAI public image release is an MHA archive, so M1 begins with `mha2nnunet`; it does not
convert DICOM. Use the official 1,500-case `picai_nnunet` split from `picai_baseline`. The commands
below reflect the **target** two-head contract (`docs/ARCHITECTURE.md §3`); the underlying
`gcalf_data` scripts need the rework noted above before they implement it:

```bash
python -m gcalf_data.prepare_picai build \
  --images-dir /path/to/picai_public_images \
  --labels-root /path/to/picai_labels \
  --splits-json /path/to/picai_baseline/src/picai_baseline/splits/picai_nnunet/splits.json \
  --task-dir "$det_data/Task2201_PICAI_csPCa" \
  --work-dir /path/to/picai_m1_work

nndet_prep Task2201_PICAI_csPCa

python -m gcalf_data.prepare_picai install-splits \
  --task-dir "$det_data/Task2201_PICAI_csPCa" \
  --preprocessed-dir "$det_data/Task2201_PICAI_csPCa/preprocessed"

python -m gcalf_data.sanity_checks \
  --task-dir "$det_data/Task2201_PICAI_csPCa" \
  --marksheet /path/to/picai_labels/clinical_information/marksheet.csv \
  --plan-path "$det_data/Task2201_PICAI_csPCa/preprocessed/D3V001_3d.pkl" \
  --report-path docs/data_report.md
```

The sanity gate must verify co-registered T2W/ADC/HBV geometry, instance-volume/JSON agreement,
official fold coverage, patient-level split separation, and — once reworked — that the planner
derives `in_channels=3` and `classifier_classes=1` (one detection class; grade lives in instance
metadata, not the class count). Do not hand-edit plan values.

Generated images, labels, plans, checkpoints, and reports containing local data remain outside the
repository. The task outputs are ignored through `det_data/`, `det_models/`, and model-artifact
rules in [.gitignore](.gitignore).

## Verification

Run the focused code checks before using data:

```bash
python -m pytest -q tests/gcalf/ tests/test_imports.py tests/test_encoder_cpu.py
```

The full milestone definition, data limitations, and exit criteria are in
[Phase 1](docs/phases/PHASE_1_data_pipeline.md). The roadmap is in
[docs/ROADMAP.md](docs/ROADMAP.md); the full architecture is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Attribution

GCALF-Net builds on nnDetection, PDHD-Net, PI-CAI, and the PI-CAI preprocessing tools. Cite the
corresponding upstream work when using this repository or a derived result.
